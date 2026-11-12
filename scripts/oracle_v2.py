"""Step 6 of the quantum study: cheaper Grover steps for Bayesian-network amplitude estimation, measured exactly.

Variants (every one estimates the same quantity; correctness tests gate every reported number):
  V0  baseline: A = state preparation + one multi-controlled X over all evidence literals; Q = A S0 A' S_chi.
  V1  one clean ancilla available to the compiler: the evidence AND (in A) and the reflection S0 then use the cheap
      clean-ancilla synthesis instead of borrowing busy qubits / no-ancilla synthesis.        (plan items a and c)
  V2  observed nodes clamped inside state preparation: observed nodes are not qubits; an observed node with an
      unobserved parent gets one weight qubit rotated to sqrt(P(e | parents)); one with only observed parents is a
      known constant c_e. P(flag) = P(evidence [, target]) / prod c_e (exact; multiply back).  (plan item b)
  V3  V2 + one clean ancilla.
Reflecting about a *different* reference state than A|0> would change the estimated amplitude, so it is not used;
item (c) is realised as a cheaper implementation of the same reflection (V1, V3). Measurement-based uncomputation
of the evidence AND (item a) is reported analytically on top of V1: it replaces the uncompute half of each AND.

  python scripts/oracle_v2.py test           # correctness test 1 (statevector, random networks <= 16 qubits)
  python scripts/oracle_v2.py count          # CX per Grover step, step-5 grid + v2.1 structure (qiskit 2.6.1)
  python scripts/oracle_v2.py realtest       # correctness test 2 (>= 50 real held-out patients; needs the v2.1 KB)
  python scripts/oracle_v2.py report         # oracle_v2.json / .png from the saved counts and tests
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate, UCRYGate, grover_operator
from qiskit.quantum_info import Statevector

from qddx.bn import BN, Evidence, Struct, ancestral, posterior_exact, with_virtual
from qddx.quantum import _basis_pm, conj_oracle, problem_circuit, state_prep

OUT, REP, SEED = Path("artifacts/v2.1/results"), Path("reports/v2.1"), 2026


def cx(qc) -> int:
    return int(dict(_basis_pm().run(qc).count_ops()).get("cx", 0))


# ------------------------------------------------------------------------------------- circuits
def baseline_A(b: BN, obs: dict[int, int]) -> QuantumCircuit:
    return problem_circuit(state_prep(b), conj_oracle(b.struct.n, obs))


def clamped_A(b: BN, obs: dict[int, int], target: int | None) -> tuple[QuantumCircuit, float]:
    """Observed nodes clamped. -> (A, C) with P(flag = 1) = P(obs [, target = 1]) / C."""
    st = b.struct
    free = [j for j in range(st.n) if j not in obs]
    weighted = [j for j in obs if any(p not in obs for p in st.parents[j])]
    q = {j: i for i, j in enumerate(free)}
    w = {j: len(free) + i for i, j in enumerate(weighted)}
    n_q = len(free) + len(weighted) + 1
    qc = QuantumCircuit(n_q, name="A_clamped")
    C = 1.0

    def angles(j: int, want: int):
        """UCRY angles over j's unobserved parents (control i <-> i-th unobserved parent), observed parents fixed."""
        pa = list(st.parents[j])
        ctrl = [p for p in pa if p not in obs]
        out = []
        for cs in range(2 ** len(ctrl)):
            bits = {p: (cs >> i) & 1 for i, p in enumerate(ctrl)}
            row = sum(((obs[p] if p in obs else bits[p]) << (len(pa) - 1 - k)) for k, p in enumerate(pa))
            p1 = b.cpts[j][row, 1]
            pr = p1 if want == 1 else 1 - p1
            out.append(2 * math.asin(math.sqrt(min(max(pr, 0.0), 1.0))))
        return ctrl, out

    for j in range(st.n):  # topological order
        if j in obs and j not in w:
            row = sum((obs[p] << (len(st.parents[j]) - 1 - k)) for k, p in enumerate(st.parents[j]))
            p1 = b.cpts[j][row, 1]
            C *= p1 if obs[j] == 1 else 1 - p1
            continue
        tgt, want = (q[j], 1) if j in q else (w[j], obs[j])
        ctrl, ang = angles(j, want)
        if not ctrl:
            qc.ry(ang[0], tgt)
        else:
            qc.append(UCRYGate(ang), [tgt] + [q[p] for p in ctrl])
    lits = [w[j] for j in weighted] + ([q[target]] if target is not None else [])
    if lits:
        qc.append(MCXGate(len(lits)), lits + [n_q - 1])
    else:
        qc.x(n_q - 1)
    return qc, C


def grover(A: QuantumCircuit, clean: int = 0) -> QuantumCircuit:
    """Q = A S0 A' S_chi over A's qubits; `clean` extra qubits stay |0> and are free for the compiler to use."""
    n = A.num_qubits
    Ap = QuantumCircuit(n + clean)
    Ap.compose(A, range(n), inplace=True)
    orc = QuantumCircuit(n + clean)
    orc.z(n - 1)
    return grover_operator(orc, state_preparation=Ap, reflection_qubits=list(range(n)))


def flag_prob(qc: QuantumCircuit, flag: int) -> float:
    return float(Statevector(qc).probabilities([flag])[1])


# ------------------------------------------------------------------------------------- test 1
def random_bn(n, k_max, rng):
    parents = [tuple(sorted(rng.choice(j, size=int(rng.integers(0, min(j, k_max) + 1)), replace=False).tolist())) for j in range(n)]
    return BN(Struct([f"x{j}" for j in range(n)], parents), [np.stack([1 - p, p], 1) for p in (rng.uniform(0.05, 0.95, 2 ** len(pa)) for pa in parents)])


def test1(cases: int = 60):
    """Statevector, random networks: same marked amplitude (V2: times the known constant), and one Grover step gives
    sin^2(3 theta) for the baseline, the clean-ancilla reflection and the clamped circuit."""
    rng = np.random.default_rng(SEED)
    worst, worst_q, done = 0.0, 0.0, 0
    for c in range(cases):
        n = int(rng.integers(5, 11))
        b = random_bn(n, int(rng.integers(1, 5)), rng)
        L = int(rng.integers(1, n))
        idx = rng.permutation(n)
        obs = {int(j): int(rng.integers(0, 2)) for j in idx[:L]}
        t = int(idx[L]) if L < n else None
        lits = {**obs, **({t: 1} if t is not None else {})}
        A0 = baseline_A(b, lits)
        A2, C = clamped_A(b, obs, t)
        if max(A0.num_qubits, A2.num_qubits) + 1 > 16:
            continue
        a0, a2 = flag_prob(A0, A0.num_qubits - 1), flag_prob(A2, A2.num_qubits - 1)
        exact = posterior_exact(b, Evidence({f"x{j}": v for j, v in obs.items()}, {}), [f"x{t}"] if t is not None else [])
        a_exact = exact[1] * (exact[0][f"x{t}"] if t is not None else 1.0)
        worst = max(worst, abs(a0 - a_exact), abs(a2 * C - a_exact))
        for A, a, clean in ((A0, a0, 0), (A0, a0, 1), (A2, a2, 0), (A2, a2, 1)):
            nq = A.num_qubits
            qc = QuantumCircuit(nq + clean)
            qc.compose(A, range(nq), inplace=True)
            qc.compose(grover(A, clean), inplace=True)
            want = math.sin(3 * math.asin(math.sqrt(a))) ** 2
            worst_q = max(worst_q, abs(flag_prob(qc, nq - 1) - want))
        done += 1
    res = {"cases": done, "max_amplitude_error": worst, "max_grover_step_error": worst_q, "pass": worst < 1e-10 and worst_q < 1e-10}
    print(json.dumps(res))
    (OUT / "oracle_v2_test1.json").write_text(json.dumps(res))
    return res


# ------------------------------------------------------------------------------------- counts
def variants(b: BN, obs: dict[int, int], target: int | None) -> dict:
    lits = {**obs, **({target: 1} if target is not None else {})}
    A0 = baseline_A(b, lits)
    A2, C = clamped_A(b, obs, target)
    r = {"qubits_v0": A0.num_qubits, "qubits_v2": A2.num_qubits, "log10_C": float(math.log10(C)) if C > 0 else None}
    for name, A, clean in (("v0", A0, 0), ("v1", A0, 1), ("v2", A2, 0), ("v3", A2, 1)):
        r[f"A_{name}"] = cx(A) if clean == 0 else cx(_pad(A, clean))
        r[f"Q_{name}"] = cx(grover(A, clean))
    r["A_ref"] = r["A_v0"]  # unit: one baseline problem preparation (= one model evaluation)
    for name in ("v0", "v1", "v2", "v3"):
        r[f"ratio_{name}"] = r[f"Q_{name}"] / (2 * r["A_ref"])
    m = len(lits)
    r["mbu_saving_estimate"] = 4 * (m - 1)  # analytic: per step, the evidence AND in A and A'; measurement-based uncompute
    # replaces each AND's uncompute half (about 3 CX per control) by at most one CZ per control
    return r


def _pad(A, clean):
    qc = QuantumCircuit(A.num_qubits + clean)
    qc.compose(A, range(A.num_qubits), inplace=True)
    return qc


def count():
    rng = np.random.default_rng(SEED)  # the same networks and evidence as cost_model.py sweep
    rows, t0 = [], time.perf_counter()
    for n in (12, 16, 20, 24):
        for k in (1, 2, 3, 4):
            bpar = [tuple(sorted(rng.choice(j, size=int(rng.integers(0, min(j, k) + 1)), replace=False).tolist())) for j in range(n)]
            b = BN(Struct([f"x{j}" for j in range(n)], bpar), [np.stack([1 - p, p], 1) for p in (rng.uniform(0.05, 0.95, 2 ** len(pa)) for pa in bpar)])
            order = rng.permutation(n).tolist()
            for L in sorted({2, n // 4, n // 2, 3 * n // 4, n - 1, n}):
                for v in (0, 1, 2, 3):
                    if L + v > n:
                        continue
                    hard = {f"x{j}": int(rng.integers(0, 2)) for j in order[:L]}
                    soft = {f"x{j}": float(rng.uniform(0.2, 0.8)) for j in order[L:L + v]}
                    ext, obs = with_virtual(b, Evidence(hard, soft))
                    r = variants(ext, obs, None)
                    r.update(n=n, k_max=k, hard=L, virtual=v, source="grid")
                    rows.append(r)
        print(f"grid n={n} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    # v2.1 structure (parents only; CPT values do not change the counts), realistic evidence: age, history and drugs
    # observed, the first f labs measured, target = one diagnosis
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    names, parents = st["names"], [tuple(p) for p in st["parents"]]
    rr = np.random.default_rng(SEED)
    b = BN(Struct(names, parents), [np.stack([1 - p, p], 1) for p in (rr.uniform(0.05, 0.95, 2 ** len(pa)) for pa in parents)])
    dx = ["heart_failure", "afib", "uti", "pneumonia", "aki", "anemia"]
    labs = [i for i, nm in enumerate(names) if nm.startswith("lab_")]
    other = [i for i, nm in enumerate(names) if nm not in dx and not nm.startswith("lab_")]
    for findings in (27, 38, 39, 40):
        obs = {i: 0 for i in other + labs[: findings - len(other)]}
        for tname in dx:
            r = variants(b, obs, names.index(tname))
            r.update(n=len(names), findings=findings, target=tname, source="v2.1")
            rows.append(r)
        print(f"v2.1 findings={findings} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    (Path("artifacts") / "oracle_v2_counts.json").write_text(json.dumps(rows, indent=1))


# ------------------------------------------------------------------------------------- test 2 (real patients)
def realtest(n_cases: int = 50):
    """Real held-out admissions, 1-4 findings revealed, target = most probable diagnosis, ancestral sub-network small
    enough for statevector: baseline and clamped amplitudes agree (times C), and emulated IAE estimates from the two
    agree within their 95% intervals."""
    import pandas as pd
    from qddx.quantum import iae_emulated
    bn = BN.from_dict(json.loads(Path("artifacts/v2.1/kb/bn.json").read_text()))
    P = json.loads(Path("configs/vocab_v2.1.json").read_text())["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    ev = [k for k, p in P.items() if p["kind"] != "diagnosis"]
    df = pd.read_parquet("artifacts/v2.1/kb/cohort.parquet")
    subs = np.unique(df.subject_id.to_numpy())
    train = set(np.random.default_rng(SEED).permutation(subs)[: int(round(0.8 * len(subs)))].tolist())
    test = df[~df.subject_id.isin(train)].sample(n=4000, random_state=SEED)
    rng = np.random.default_rng(SEED)
    rows = []
    for _, r in test.iterrows():
        seen = [k for k in ev if r[k] >= 0]
        keep = list(rng.choice(seen, size=int(rng.integers(1, 5)), replace=False))
        E = Evidence({k: int(r[k]) for k in keep}, {})
        post, _, _ = posterior_exact(bn, E, dx)
        t = max(post, key=post.get)
        sub = ancestral(bn, keep + [t])
        if sub.struct.n + 1 > 16:
            continue
        obs = {sub.struct.index(k): E.hard[k] for k in keep}
        ti = sub.struct.index(t)
        A0 = baseline_A(sub, {**obs, ti: 1})
        A2, C = clamped_A(sub, obs, ti)
        a0, a2 = flag_prob(A0, A0.num_qubits - 1), flag_prob(A2, A2.num_qubits - 1)
        e0 = iae_emulated(a0, 0.01, 0.05, 100, SEED + len(rows))
        e2 = iae_emulated(a2, 0.01, 0.05, 100, SEED + 7 + len(rows))
        lo2, hi2 = e2["ci"][0] * C, e2["ci"][1] * C
        rows.append({"qubits_v0": A0.num_qubits, "qubits_v2": A2.num_qubits, "a0": a0, "a2_times_C": a2 * C,
                     "amp_err": abs(a0 - a2 * C), "est0": e0["estimate"], "est2_times_C": e2["estimate"] * C,
                     "ci_overlap": bool(max(e0["ci"][0], lo2) <= min(e0["ci"][1], hi2)),
                     "calls0": e0["oracle_queries"], "calls2": e2["oracle_queries"]})
        if len(rows) >= n_cases:
            break
    d = pd.DataFrame(rows)
    res = {"cases": int(len(d)), "max_amplitude_error": float(d.amp_err.max()), "ci_overlap_share": float(d.ci_overlap.mean()),
           "median_qubits_v0": float(d.qubits_v0.median()), "median_qubits_v2": float(d.qubits_v2.median()),
           "pass": bool(len(d) >= 50 and d.amp_err.max() < 1e-10 and d.ci_overlap.all())}
    print(json.dumps(res))
    (OUT / "oracle_v2_test2.json").write_text(json.dumps(res))


# ------------------------------------------------------------------------------------- report
def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads((Path("artifacts") / "oracle_v2_counts.json").read_text())
    t1 = json.loads((OUT / "oracle_v2_test1.json").read_text())
    t2p = OUT / "oracle_v2_test2.json"
    t2 = json.loads(t2p.read_text()) if t2p.exists() else None
    v21 = [r for r in rows if r["source"] == "v2.1"]
    grid = [r for r in rows if r["source"] == "grid"]
    summ = {}
    for name in ("v0", "v1", "v2", "v3"):
        summ[name] = {"v21_ratio_range": [min(r[f"ratio_{name}"] for r in v21), max(r[f"ratio_{name}"] for r in v21)],
                      "v21_Q_range": [min(r[f"Q_{name}"] for r in v21), max(r[f"Q_{name}"] for r in v21)],
                      "grid_ratio_median": float(np.median([r[f"ratio_{name}"] for r in grid])),
                      "grid_ratio_max": float(max(r[f"ratio_{name}"] for r in grid))}
    v21_c = [r["log10_C"] for r in v21 if r["log10_C"] is not None]
    # step-4 frontier redone in gate units for each variant: eps* in model evaluations / ratio (ratio = v2.1 range)
    fr = json.loads((OUT / "frontier.json").read_text())["rows"]
    front = {}
    for name in summ:
        for est in ("ideal", "tuned"):
            for end, rr in zip(("lo", "hi"), summ[name]["v21_ratio_range"]):
                eps = np.array([r[f"eps_star_qlw_{est}"] for r in fr]) / rr
                front[f"{name}_{est}_ratio_{end}"] = {"share_ae_wins_at_0.1": float((eps >= 0.1).mean()),
                                                      "median_orders_below_0.1": float(np.median(np.log10(0.1 / eps)))}
    out = {"tests": {"1_statevector_random": t1, "2_real_patients": t2, "3_test_quantum_unchanged": "run pytest tests/test_quantum.py"},
           "summary": summ, "v21_log10_C_range": [min(v21_c), max(v21_c)] if v21_c else None,
           "frontier_by_variant": front, "rows": rows,
           "unit": "ratio = CX of one Grover step / (2 x CX of the baseline problem preparation A_v0), i.e. per model evaluation"}
    (OUT / "oracle_v2.json").write_text(json.dumps(out, indent=1))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = {"v0": "V0 baseline", "v1": "V1 one clean ancilla", "v2": "V2 observed nodes clamped", "v3": "V3 clamped + ancilla"}
    colors = {"v0": "#6b6a64", "v1": "#2a78d6", "v2": "#eb6834", "v3": "#1baf7a"}
    for name in labels:
        vals = sorted(r[f"ratio_{name}"] for r in grid)
        axs[0].plot(np.linspace(0, 1, len(vals)), vals, color=colors[name], lw=2, label=labels[name])
    axs[0].axhline(1, color="#1f1f1e", lw=1, ls=":")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("grid instances, sorted (304 networks, 12-24 nodes)")
    axs[0].set_ylabel("CX per Grover step / 2 baseline preparations")
    axs[0].set_title("Grid: cost per Grover step, in model evaluations", loc="left", fontsize=9.5)
    axs[0].legend(frameon=False, fontsize=8)
    xs = np.arange(4)
    for i, f in enumerate((27, 38, 39, 40)):
        rs = [r for r in v21 if r["findings"] == f]
        for j, name in enumerate(labels):
            axs[1].bar(i + (j - 1.5) * 0.2, np.median([r[f"ratio_{name}"] for r in rs]), 0.2, color=colors[name],
                       label=labels[name] if i == 0 else None)
    axs[1].axhline(1, color="#1f1f1e", lw=1, ls=":")
    axs[1].set_xticks(xs)
    axs[1].set_xticklabels([f"{f} findings" for f in (27, 38, 39, 40)])
    axs[1].set_ylabel("CX per Grover step / 2 baseline preparations")
    axs[1].set_title("v2.1 network (46 nodes), median over the 6 targets", loc="left", fontsize=9.5)
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(REP / "oracle_v2.png", dpi=180)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    {"test": test1, "count": count, "realtest": realtest, "report": report}[sys.argv[1]]()
