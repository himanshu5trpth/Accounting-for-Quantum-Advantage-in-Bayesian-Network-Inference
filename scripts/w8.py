"""W8 of the quantum paper plan: supporting runs (public / structure-only inputs; cloud workspace).

  python scripts/w8.py kappa          # kappa for tuned IAE and maximum-likelihood AE on 50 benchmark instances (same criterion)
  python scripts/w8.py circuits       # write the V0/V1 circuits to QPY for the compiler comparison
  python scripts/w8.py compile        # count CX under this environment's qiskit at levels 1/2/3 (run in each environment)
  python scripts/w8.py tket           # pytket on the same circuits
  python scripts/w8.py ft             # fault-tolerant resource estimates (qsharp estimator, every hardware profile)
"""
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
OUT, SEED, REL, Z90 = Path("artifacts/quantum_paper"), 2026, 0.1, 1.645


# ------------------------------------------------------------------------------------- kappa
def mlae(a: float, K: int, N: int, rng) -> tuple[float, int]:
    """Maximum-likelihood AE (Suzuki et al. 2020), exponential schedule k = 0, 1, 2, 4, ..., 2^(K-1); N shots each.
    Outcomes from the amplification law; MLE by grid search plus a local refinement. -> (estimate, model evaluations)."""
    from scipy.optimize import minimize_scalar
    ks = np.array([0] + [2 ** j for j in range(K)])
    th = math.asin(math.sqrt(a))
    h = rng.binomial(N, np.sin((2 * ks + 1) * th) ** 2)

    def nll(t):
        q = np.clip(np.sin((2 * ks + 1) * t) ** 2, 1e-15, 1 - 1e-15)
        return -float(np.sum(h * np.log(q) + (N - h) * np.log(1 - q)))
    grid = np.linspace(1e-9, math.pi / 2 - 1e-9, int(max(20000, 40 * (2 * ks[-1] + 1))))
    q = np.clip(np.sin((2 * ks[None, :] + 1) * grid[:, None]) ** 2, 1e-15, 1 - 1e-15)
    ll = (h * np.log(q) + (N - h) * np.log(1 - q)).sum(1)
    t0 = grid[int(np.argmax(ll))]
    step = grid[1] - grid[0]
    t = minimize_scalar(nll, bounds=(max(t0 - step, 0), min(t0 + step, math.pi / 2)), method="bounded").x
    return math.sin(t) ** 2, int(N * np.sum(2 * ks + 1))


def mlae_fast(a: float, K: int, N: int, rng) -> tuple[float, int]:
    """Same estimator as mlae() (exponential schedule, maximum likelihood), with the maximum found by successive
    refinement: after each new power the likelihood is maximised on a local grid of +-6 standard errors (Fisher
    information of the powers so far). Exact for the schedules used here and fast at any amplitude."""
    from scipy.optimize import minimize_scalar
    ks = np.array([0] + [2 ** j for j in range(K)])
    th = math.asin(math.sqrt(a))
    h = rng.binomial(N, np.sin((2 * ks + 1) * th) ** 2)

    def ll(t, upto):
        kk, hh = ks[:upto], h[:upto]
        q = np.clip(np.sin((2 * kk[None, :] + 1) * np.atleast_1d(t)[:, None]) ** 2, 1e-15, 1 - 1e-15)
        return (hh * np.log(q) + (N - hh) * np.log(1 - q)).sum(1)
    grid = np.linspace(1e-9, math.pi / 2 - 1e-9, 4001)
    v = ll(grid, 1)
    beam = list(grid[np.argsort(v)[-5:]])  # keep the 5 best candidates through the refinement (aliasing guard)
    for j in range(2, len(ks) + 1):
        sd = 1 / math.sqrt(4 * N * np.sum((2 * ks[:j - 1] + 1) ** 2))
        cand = []
        for t0 in beam:
            g = np.linspace(max(t0 - 6 * sd, 1e-12), min(t0 + 6 * sd, math.pi / 2), 801)
            lv = ll(g, j)
            loc = np.flatnonzero((lv >= np.roll(lv, 1)) & (lv >= np.roll(lv, -1)))
            cand += [(lv[i], g[i]) for i in loc]
        cand.sort(reverse=True)
        beam = []
        for val, tt in cand:
            if all(abs(tt - b) > sd / 4 for b in beam):
                beam.append(tt)
            if len(beam) == 5:
                break
    t = beam[0]
    step = 12 * (1 / math.sqrt(4 * N * np.sum((2 * ks + 1) ** 2))) / 800
    t = minimize_scalar(lambda x: -ll(x, len(ks))[0], bounds=(max(t - step, 0), min(t + step, math.pi / 2)), method="bounded").x
    n_full = int(max(20000, 40 * (2 * ks[-1] + 1)))
    if n_full <= 400_000:  # also take the full-grid maximum when affordable, and keep the higher likelihood
        g = np.linspace(1e-9, math.pi / 2 - 1e-9, n_full)
        t2 = g[int(np.argmax(ll(g, len(ks))))]
        if ll(t2, len(ks))[0] > ll(t, len(ks))[0]:
            t = t2
    return math.sin(t) ** 2, int(N * np.sum(2 * ks + 1))


def mlae_cost(aE: float, aT: float, p: float, N: int, seed: int, repeats: int = 10, fast: bool = False):
    """Mean model evaluations for 90% of repeats within REL of p = aT / aE (the criterion LW and IAE are held to)."""
    def K_for(a, rel):
        th = math.asin(math.sqrt(a))
        for K in range(0, 40):
            ks = np.array([0] + [2 ** j for j in range(K)])
            sd = 1 / math.sqrt(4 * N * np.sum((2 * ks + 1) ** 2))
            if 2 * sd / math.tan(th) <= rel:
                return K
        return 40
    rng = np.random.default_rng(seed)
    for f in (2, 1, 0.5, 0.25, 0.125, 0.0625):
        kE, kT = K_for(aE, f * REL / Z90), K_for(aT, f * REL / Z90)
        errs, calls = [], []
        for _ in range(repeats):
            est = mlae_fast if fast else mlae
            e, c1 = est(aE, kE, N, rng)
            t, c2 = est(aT, kT, N, rng)
            errs.append(abs(t / e - p) / p if e > 0 else np.inf)
            calls.append(c1 + c2)
        if np.quantile(errs, 0.9, method="higher") <= REL:
            return float(np.mean(calls))
    return None


def kappa():
    import os
    import pandas as pd
    os.environ["AE_SHOTS"], os.environ["AE_ALPHA"] = "16", "0.2"
    import lw_check
    from benchmarks import NETWORKS, _frame
    _, ok = _frame(NETWORKS)
    ok = ok[(ok.a_T > 1e-7) & (ok.p > 1e-4)]
    pick = ok.sample(n=50, random_state=SEED)
    rows = []

    def _stop(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, _stop)
    for k, r in enumerate(pick.itertuples()):
        unit = math.pi / REL * (1 / math.sqrt(r.a_E) + 1 / math.sqrt(r.a_T))
        rec = {"network": r.network, "i": int(r.i), "p": float(r.p), "a_E": float(r.a_E), "a_T": float(r.a_T)}
        signal.alarm(60)
        try:
            c = lw_check.ae_cost(r.a_E, r.a_T, r.p, SEED + k)
            rec["kappa_iae_16_0.2"] = c / unit if c else None
        except TimeoutError:
            rec["kappa_iae_16_0.2"], rec["iae_stalled"] = None, True
        finally:
            signal.alarm(0)
        for N in (16, 100):
            c = mlae_cost(r.a_E, r.a_T, r.p, N, SEED + k)
            rec[f"kappa_mlae_N{N}"] = c / unit if c else None
        rows.append(rec)
        print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in rec.items() if a.startswith("kappa")}, flush=True)
    d = pd.DataFrame(rows)
    out = {"instances": len(d), "criterion": "90% of 10 repeats within 10% relative error of the exact posterior (same as LW, z = 1.645)",
           "unit": "kappa = model evaluations / (pi / eps (1/sqrt(a_E) + 1/sqrt(a_T)))",
           "frontier_kappa_tuned_iae_clinical": 11.685850819903646, "estimators": {}}
    for c in [c for c in d.columns if c.startswith("kappa")]:
        v = d[c].dropna()
        out["estimators"][c] = {"median": float(v.median()), "iqr": [float(v.quantile(.25)), float(v.quantile(.75))], "n": int(len(v)),
                                "failed_or_stalled": int(d[c].isna().sum())}
    (OUT / "w8_kappa.json").write_text(json.dumps({**out, "rows": rows}, indent=1))
    print(json.dumps(out, indent=1))


GRID_AE = tuple(float(x) for x in np.logspace(-6, 0, 13))      # tasks/plan_round3.md Q3
GRID_P = tuple(float(x) for x in np.logspace(-4, -0.3, 12))


def kappa_grid(part: str = "0", parts: str = "1"):
    """kappa as a function of (a_E, p) for tuned IAE (16 shots, alpha 0.2) and MLAE (16 shots), 3 seeds per point,
    each seed one 10-repeat calibration against the shared criterion. Checks first that mlae_fast == mlae."""
    import os
    os.environ["AE_SHOTS"], os.environ["AE_ALPHA"] = "16", "0.2"
    import lw_check
    chk = []
    for a in (0.3, 0.01, 1e-4):
        for K in (2, 5, 8):
            r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
            chk.append(abs(mlae(a, K, 16, r1)[0] - mlae_fast(a, K, 16, r2)[0]) / a)
    print("mlae_fast vs mlae, max relative difference:", max(chk), flush=True)
    pts = [(aE, p) for aE in GRID_AE for p in GRID_P if aE * p >= 1e-12][int(part)::int(parts)]

    def _stop(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, _stop)
    rows = []
    for aE, p in pts:
        aT = aE * p
        unit = math.pi / REL * (1 / math.sqrt(aE) + 1 / math.sqrt(aT))
        for sd in range(3):
            rec = {"a_E": aE, "p": p, "a_T": aT, "seed": sd}
            signal.alarm(120)
            try:
                c = lw_check.ae_cost(aE, aT, p, SEED + 97 * sd)
                rec["kappa_iae"] = c / unit if c else None
            except TimeoutError:
                rec["kappa_iae"], rec["iae_stalled"] = None, True
            except Exception as e:  # qiskit's IAE occasionally raises on emulated jobs; recorded as a failure
                rec["kappa_iae"], rec["iae_error"] = None, type(e).__name__
            finally:
                signal.alarm(0)
            c = mlae_cost(aE, aT, p, 16, SEED + 97 * sd, fast=True)
            rec["kappa_mlae16"] = c / unit if c else None
            rows.append(rec)
            print(rec, flush=True)
        json.dump(rows, open(OUT / f"w8_kappa_grid_{part}.json", "w"), indent=1)
    print("done", flush=True)


# ------------------------------------------------------------------------------------- compilers
def _cases():
    """(name, network, evidence, target, state): the v2.1 structure at 27 and 38 findings, and one full-network
    instance each of insurance, alarm and hepar2 (W3.2 protocol)."""
    from msbn import MSBN, from_pgmpy, instance
    import warnings
    warnings.filterwarnings("ignore")
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    names, parents = st["names"], [tuple(p) for p in st["parents"]]
    rr = np.random.default_rng(SEED)
    cpts = [np.stack([1 - p, p], 1).reshape((2,) * len(pa) + (2,)) for pa in parents for p in [rr.uniform(0.05, 0.95, 2 ** len(pa))]]
    bn = MSBN(names, [2] * len(names), parents, cpts)
    labs = [i for i, nm in enumerate(names) if nm.startswith("lab_")]
    dx = [i for i, nm in enumerate(names) if nm in ("heart_failure", "afib", "uti", "pneumonia", "aki", "anemia")]
    other = [i for i in range(len(names)) if i not in dx and i not in labs]
    cases = [(f"v2.1_findings{f}", bn, {i: 0 for i in other + labs[: f - len(other)]}, dx[0], 1) for f in (27, 38)]
    from pgmpy.utils import get_example_model
    rng = np.random.default_rng(SEED)
    for name in ("insurance", "alarm", "hepar2"):
        b = from_pgmpy(get_example_model(name))
        ev, t, s = instance(b, rng)
        cases.append((f"{name}_full", b, ev, t, s))
    return cases


def circuits():
    import pickle
    with open(OUT / "w8_cases.pkl", "wb") as f:
        pickle.dump(_cases(), f)
    print("cases written")


def _built():
    import pickle
    from msbn import baseline_A
    from oracle_v2 import grover
    with open(OUT / "w8_cases.pkl", "rb") as f:
        cases = pickle.load(f)
    out = []
    for name, bn, ev, t, s in cases:
        A, _ = baseline_A(bn, range(bn.n), ev, t, s)
        out += [(f"{name}|A", A), (f"{name}|Q_v0", grover(A, 0)), (f"{name}|Q_v1", grover(A, 1))]
    return out


def compile_():
    import qiskit
    from qiskit.transpiler import generate_preset_pass_manager
    res = {}
    built = _built()
    for lvl in (1, 2, 3):
        pm = generate_preset_pass_manager(optimization_level=lvl, basis_gates=["rz", "sx", "x", "cx"], seed_transpiler=SEED)
        for n, c in built:
            t0 = time.perf_counter()
            res[f"{n}|opt{lvl}"] = int(dict(pm.run(c).count_ops()).get("cx", 0))
            print(qiskit.__version__, lvl, n, res[f"{n}|opt{lvl}"], f"{time.perf_counter() - t0:.0f}s", flush=True)
    (OUT / f"w8_compile_qiskit{qiskit.__version__}.json").write_text(json.dumps(res, indent=1))


def tket():
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.passes import FullPeepholeOptimise, AutoRebase
    from pytket import OpType
    from qiskit import transpile
    res = {}
    for n, c in _built():
        base = transpile(c, basis_gates=["u", "cx"], optimization_level=0)  # qiskit's high-level synthesis, no optimisation
        tk = qiskit_to_tk(base)
        FullPeepholeOptimise().apply(tk)
        AutoRebase({OpType.CX, OpType.Rz, OpType.Rx}).apply(tk)
        res[n] = int(tk.n_gates_of_type(OpType.CX))
        print("tket", n, res[n], flush=True)
    (OUT / "w8_compile_tket.json").write_text(json.dumps(res, indent=1))


def compare():
    out = {}
    for f in sorted(OUT.glob("w8_compile_*.json")):
        d = json.loads(f.read_text())
        tool = f.stem.replace("w8_compile_", "")
        for k, v in d.items():
            parts = k.split("|")
            case, circ, lvl = parts[0], parts[1], parts[2] if len(parts) > 2 else "peephole"
            out.setdefault(f"{tool}|{lvl}", {}).setdefault(case, {})[circ] = v
    summ = {}
    for setting, cases in out.items():
        for case, c in cases.items():
            if all(k in c for k in ("A", "Q_v0", "Q_v1")):
                summ.setdefault(setting, {})[case] = {"ratio_v0": c["Q_v0"] / (2 * c["A"]), "ratio_v1": c["Q_v1"] / (2 * c["A"]),
                                                      "v1_below_v0": c["Q_v1"] < c["Q_v0"]}
    allpass = all(x["v1_below_v0"] for s in summ.values() for x in s.values())
    (OUT / "w8_compilers.json").write_text(json.dumps({"counts": out, "ratios": summ, "v1_below_v0_everywhere": allpass}, indent=1))
    for s, cs in summ.items():
        print(s, {c: (round(x["ratio_v0"], 2), round(x["ratio_v1"], 2)) for c, x in cs.items()})
    print("V1 < V0 everywhere:", allpass)


# ------------------------------------------------------------------------------------- fault tolerance
def _logical_counts(qc, reps: int, extra) -> dict:
    """Per-circuit rotation / T / CX counts after transpiling to rz/sx/x/cx; total = extra + reps x this circuit."""
    from qddx.quantum import _basis_pm
    t = _basis_pm().run(qc)
    rot = tg = 0
    depth = {q: 0 for q in range(t.num_qubits)}
    for inst in t.data:
        qs = [t.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "rz":
            ang = float(inst.operation.params[0]) % (2 * math.pi)
            k = ang / (math.pi / 4)
            if abs(k - round(k)) < 1e-9:
                if round(k) % 2 == 1:
                    tg += 1
                continue
            rot += 1
            depth[qs[0]] += 1
        elif len(qs) > 1:
            m = max(depth[q] for q in qs)
            for q in qs:
                depth[q] = m
    return {"rot": rot, "t": tg, "rot_depth": max(depth.values()), "cx": int(dict(t.count_ops()).get("cx", 0)), "qubits": t.num_qubits}


def ft():
    from qsharp.estimator import LogicalCounts, EstimatorParams, QubitParams, QECScheme
    from msbn import MSBN, baseline_A
    from oracle_v2 import grover, _pad
    import warnings
    warnings.filterwarnings("ignore")
    res_json = json.loads(Path("artifacts/v2.1/results/resources.json").read_text())
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    names, parents = st["names"], [tuple(p) for p in st["parents"]]
    rr = np.random.default_rng(SEED)
    cpts = [np.stack([1 - p, p], 1).reshape((2,) * len(pa) + (2,)) for pa in parents for p in [rr.uniform(0.05, 0.95, 2 ** len(pa))]]
    bn = MSBN(names, [2] * len(names), parents, cpts)
    labs = [i for i, nm in enumerate(names) if nm.startswith("lab_")]
    dx = {nm: i for i, nm in enumerate(names) if nm in ("heart_failure", "afib", "uti", "pneumonia", "aki", "anemia")}
    other = [i for i in range(len(names)) if i not in dx.values() and i not in labs]
    profiles = [p for p in res_json["profiles"] if p["quantile"] in (0.5, 0.001)]
    combos = [(qp, qec) for qp in ("GATE_NS_E3", "GATE_NS_E4", "GATE_US_E3", "GATE_US_E4") for qec in ("SURFACE_CODE",)] + \
             [(qp, qec) for qp in ("MAJ_NS_E4", "MAJ_NS_E6") for qec in ("SURFACE_CODE", "FLOQUET_CODE")]
    rows = []
    for p in profiles:
        A, _ = baseline_A(bn, range(bn.n), {i: 0 for i in other + labs[: p["findings"] - len(other)]}, dx[p["target"]], 1)
        for var, clean in (("v0", 0), ("v1", 1)):
            a = _logical_counts(A if clean == 0 else _pad(A, clean), 1, None)
            q = _logical_counts(grover(A, clean), 1, None)
            k = p["k_max"]
            counts = {"numQubits": q["qubits"], "tCount": a["t"] + k * q["t"], "rotationCount": a["rot"] + k * q["rot"],
                      "rotationDepth": a["rot_depth"] + k * q["rot_depth"], "cczCount": 0, "measurementCount": q["qubits"]}
            for qp, qec in combos:
                params = EstimatorParams()
                params.qubit_params.name = getattr(QubitParams, qp)
                params.qec_scheme.name = getattr(QECScheme, qec)
                r = LogicalCounts(counts).estimate(params)
                pc = r["physicalCounts"]
                rows.append({"profile_quantile": p["quantile"], "k_max": k, "variant": var, "qubit_params": qp, "qec": qec,
                             "logical_counts": counts, "cx_total": a["cx"] + k * q["cx"], "physical_qubits": pc["physicalQubits"],
                             "runtime_seconds": pc["runtime"] / 1e9, "code_distance": r["logicalQubit"]["codeDistance"]})
                print(p["quantile"], var, qp, qec, pc["physicalQubits"], f"{pc['runtime'] / 1e9:.3g} s", flush=True)
    summ = {}
    for p in profiles:
        for var in ("v0", "v1"):
            R = [r for r in rows if r["profile_quantile"] == p["quantile"] and r["variant"] == var]
            summ[f"q{p['quantile']}|{var}"] = {"physical_qubits_range": [min(r["physical_qubits"] for r in R), max(r["physical_qubits"] for r in R)],
                                               "runtime_seconds_range": [min(r["runtime_seconds"] for r in R), max(r["runtime_seconds"] for r in R)],
                                               "cx_total": R[0]["cx_total"]}
    (OUT / "w8_ft.json").write_text(json.dumps({"summary": summ, "rows": rows, "note": "plain AE on P(evidence) at the resources.json "
                                                "profiles (k_max from there); every hardware profile the estimator ships; "
                                                "default error budget 1e-3"}, indent=1))
    print(json.dumps(summ, indent=1))


def kappa_map_at(est: str, aE: float, p: float, path=OUT / "kappa_map.json") -> float:
    """The Q3 map (round3_q.py kappa-map) evaluated at (a_E, p): bilinear in log10, clamped to the grid."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    m = json.loads(Path(path).read_text())
    M = np.array(m["estimators"][est]["log10_kappa"], float)
    _, (ii, jj) = distance_transform_edt(~np.isfinite(M), return_indices=True)
    f = RegularGridInterpolator((np.log10(m["a_E"]), np.log10(m["p"])), M[ii, jj])
    x = [np.clip(np.log10(aE), np.log10(m["a_E"][0]), np.log10(m["a_E"][-1])), np.clip(np.log10(p), np.log10(m["p"][0]), np.log10(m["p"][-1]))]
    return float(10 ** f([x])[0])


def kappa_clinical():
    """Q2/Q3 on the 46 clinical calibration instances (lw_check rows, tuned IAE 16 shots alpha 0.2): measured kappa_IAE,
    kappa_MLAE (16 shots, same criterion), and the map's prediction for both. Resumable (one row per line)."""
    import time
    rows = [json.loads(x) for x in Path("artifacts/lw_check/rows.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("kb") == "artifacts/v2.1/kb" and r.get("shots") == 16 and r.get("alpha") == 0.2 and r.get("qlw")]
    outp = OUT / "kappa_clinical.jsonl"
    done = {json.loads(x)["k"] for x in outp.read_text().splitlines()} if outp.exists() else set()
    t0 = time.perf_counter()
    for k, r in enumerate(rows):
        if k in done:
            continue
        if time.perf_counter() - t0 > 140:
            print(f"paused at {k}/{len(rows)}")
            return
        aE = 10 ** (r["log10_pe"] - r["log10_W"])
        unit = math.pi / REL * (1 / math.sqrt(aE) + 1 / math.sqrt(aE * r["p"]))
        c = mlae_cost(aE, aE * r["p"], r["p"], 16, SEED + k, fast=True)
        rec = {"k": k, "a_E": aE, "p": r["p"], "kappa_iae": r["qlw"] / unit, "kappa_mlae16": c / unit if c else None,
               "map_iae": kappa_map_at("kappa_iae", aE, r["p"]), "map_mlae16": kappa_map_at("kappa_mlae16", aE, r["p"])}
        with open(outp, "a") as f:
            f.write(json.dumps(rec) + "\n")
    R = [json.loads(x) for x in outp.read_text().splitlines()]
    out = {"n": len(R)}
    for est in ("iae", "mlae16"):
        v = [r for r in R if r[f"kappa_{est}"]]
        err = [abs(math.log10(r[f"map_{est}"] / r[f"kappa_{est}"])) for r in v]
        out[est] = {"n_meeting_criterion": len(v), "measured_median": float(np.median([r[f"kappa_{est}"] for r in v])),
                    "map_median": float(np.median([r[f"map_{est}"] for r in v])),
                    "median_abs_log10_error": float(np.median(err))}
    (OUT / "kappa_clinical.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"kappa": kappa, "kappa-grid": lambda: kappa_grid(*sys.argv[2:]), "circuits": circuits, "compile": compile_, "tket": tket, "compare": compare, "ft": ft,
     "kappa-clinical": kappa_clinical}[sys.argv[1]]()
