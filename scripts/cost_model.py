"""Step 5 of the quantum study: gate cost of one Grover step vs one state preparation for Bayesian-network oracles.
Counts are exact (transpile to rz/sx/x/cx, optimization level 1, all-to-all, as in resources.json). Gate synthesis is
compiler-version dependent: the law is for qiskit 2.6.1, which reproduces resources.json exactly (qiskit 2.2.3 does not:
it cannot borrow busy qubits inside a circuit, so its Grover steps cost more). Needs artifacts/v2.1/kb/structure_only.json
(parents only) and artifacts/v2.1/results/resources.json.

  python scripts/cost_model.py sweep          # grid over size, in-degree, evidence literals, virtual qubits -> artifacts/cost_grid.json
  python scripts/cost_model.py fit            # fit the law, run the tests, write cost_model.json / .png / docs
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from qddx.bn import BN, Evidence, Struct, with_virtual
from qddx.quantum import _basis_pm, conj_oracle, problem_circuit, state_prep

OUT = Path("artifacts/v2.1/results")
GRID = Path("artifacts/cost_grid.json")
SEED = 2026


def cx(qc) -> int:
    return int(dict(_basis_pm().run(qc).count_ops()).get("cx", 0))


def grover(circ):
    from qiskit_algorithms import EstimationProblem
    return EstimationProblem(circ, objective_qubits=[circ.num_qubits - 1]).grover_operator


def random_bn(n: int, k_max: int, rng) -> BN:
    parents = [tuple(sorted(rng.choice(j, size=int(rng.integers(0, min(j, k_max) + 1)), replace=False).tolist())) for j in range(n)]
    cpts = [np.stack([1 - p, p], 1) for p in (rng.uniform(0.05, 0.95, 2 ** len(pa)) for pa in parents)]
    return BN(Struct([f"x{j}" for j in range(n)], parents), cpts)


def measure(b: BN, hard: dict, soft: dict) -> dict:
    ext, obs = with_virtual(b, Evidence(hard, soft))
    prep = state_prep(ext)
    A = problem_circuit(prep, conj_oracle(ext.struct.n, obs))
    c_prep, c_A, c_Q = cx(prep), cx(A), cx(grover(A))
    return {"nodes": ext.struct.n, "width": ext.struct.n + 1, "literals": len(obs), "free": ext.struct.n - len(obs),
            "prep_cx": c_prep, "A_cx": c_A, "Q_cx": c_Q, "ucry_cx": int(sum(2 ** len(p) for p in ext.struct.parents if p)),
            "in_degree_max": int(max(len(p) for p in ext.struct.parents))}


def sweep():
    rng = np.random.default_rng(SEED)
    rows, t0 = [], time.perf_counter()
    for n in (12, 16, 20, 24):
        for k in (1, 2, 3, 4):
            b = random_bn(n, k, rng)
            order = rng.permutation(n).tolist()
            for L in sorted({2, n // 4, n // 2, 3 * n // 4, n - 1, n}):
                for v in (0, 1, 2, 3):
                    if L + v > n:
                        continue
                    hard = {f"x{j}": int(rng.integers(0, 2)) for j in order[:L]}
                    soft = {f"x{j}": float(rng.uniform(0.2, 0.8)) for j in order[L:L + v]}
                    r = measure(b, hard, soft)
                    r.update(n=n, k_max=k, hard=L, virtual=v)
                    rows.append(r)
        print(f"n={n} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    GRID.write_text(json.dumps(rows, indent=1))


# ------------------------------------------------------------------------------------- the law
def mcx_noancilla_table(n_max: int = 64) -> dict[int, int]:
    """CX of a multi-controlled X with m controls and no other qubit in the circuit (qiskit's synthesis), measured."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate
    out = {}
    for m in range(1, n_max + 1):
        qc = QuantumCircuit(m + 1)
        qc.append(MCXGate(m), list(range(m + 1)))
        out[m] = cx(qc)
    return out


def m0(m: int, table: dict) -> int:
    """No-ancilla MCX cost: 4m^2 - 4m + 2 for 5 <= m <= 32 (exact); measured table otherwise (about 124 m above 32)."""
    if 5 <= m <= 32:
        return 4 * m * m - 4 * m + 2
    if m in table:
        return table[m]
    return int(round(124 * m))  # beyond the measured table (extrapolated)


def oracle_cx(m: int, free: int, table: dict) -> tuple[int, int, str]:
    """(CX of the evidence oracle inside A, CX saved where A-dagger meets the reflection, regime).
    m = marked literals (findings, virtual-evidence flags and target), free = qubits that are not literals or the flag."""
    if m <= 1:
        return m, 0, "trivial"
    if free == 0:
        return m0(m, table), 0, "no free qubit"
    if m == 2:
        return 6, 0, "toffoli"
    if m == 3:
        return 14, 2, "small"
    if free >= -(-m // 2) - 1:  # at least ceil(m/2) - 1 idle qubits to borrow
        return 8 * m - 12, 2 * m - 6, "8m-12 (many borrowed qubits)"
    return 12 * m - 18, 6 * m - 12, "12m-18 (few borrowed qubits)"


def predict(ucry_cx: int, nodes: int, literals: int, table: dict) -> dict:
    """The cost law. P = sum over nodes with k >= 1 parents of 2^k; A = P + oracle; one Grover step
    Q = A . S0 . A-dagger . S_chi costs 2A + M0(nodes) - saving, where M0(nodes) is the reflection about |0...0> on all
    nodes + flag (a multi-controlled gate with no qubit left to borrow). Ratio = Q / (2A)."""
    o, save, regime = oracle_cx(literals, nodes - literals, table)
    A = ucry_cx + o
    Q = 2 * A + m0(nodes, table) - save
    return {"A_cx": A, "Q_cx": Q, "ratio": Q / (2 * A), "oracle_cx": o, "reflection_cx": m0(nodes, table) - save, "regime": regime}


def fit():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.loads(GRID.read_text())
    table = mcx_noancilla_table()
    tests = {}
    # test 1: every grid point
    errs = []
    for r in rows:
        p = predict(r["ucry_cx"], r["nodes"], r["literals"], table)
        r["pred_A"], r["pred_Q"], r["pred_ratio"], r["regime"] = p["A_cx"], p["Q_cx"], p["ratio"], p["regime"]
        r["ratio"] = r["Q_cx"] / (2 * r["A_cx"])
        errs.append(max(abs(p["A_cx"] - r["A_cx"]) / r["A_cx"], abs(p["Q_cx"] - r["Q_cx"]) / r["Q_cx"]))
        assert r["prep_cx"] == r["ucry_cx"]
    tests["1_grid_points"] = {"n": len(rows), "tolerance": 0.01, "max_rel_error": float(max(errs)),
                              "exact": int(sum(e == 0 for e in errs)), "pass": bool(max(errs) <= 0.01)}
    # test 2: ratio monotone (non-increasing) in the number of evidence literals, same network / virtual count
    groups = {}
    for r in rows:
        groups.setdefault((r["n"], r["k_max"], r["virtual"]), []).append((r["literals"], r["ratio"]))
    viol = 0
    for g in groups.values():
        g.sort()
        viol += sum(b[1] > a[1] + 1e-12 for a, b in zip(g, g[1:]))
    tests["2_monotone_in_literals"] = {"groups": len(groups), "violations": int(viol), "pass": viol == 0}
    # tests 3-4: out-of-sample points already measured in resources.json
    res = json.loads((OUT / "resources.json").read_text())
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    P46 = sum(2 ** len(p) for p in st["parents"] if p)
    n46 = len(st["names"])
    t3 = []
    for p in res["profiles"]:
        q = predict(P46, n46, p["findings"] + 1, table)  # findings + the target diagnosis
        t3.append({"quantile": p["quantile"], "A_measured": p["problem_cx"], "A_predicted": q["A_cx"], "Q_measured": p["grover_cx"],
                   "Q_predicted": q["Q_cx"], "ratio_measured": p["grover_cx"] / (2 * p["problem_cx"]), "ratio_predicted": q["ratio"]})
    tests["3_resources_profiles"] = {"prep_cx_measured": res["kb_state_prep"]["two_qubit_gates"], "prep_cx_predicted": P46,
                                     "points": t3, "pass": P46 == res["kb_state_prep"]["two_qubit_gates"] and
                                     all(x["A_measured"] == x["A_predicted"] and x["Q_measured"] == x["Q_predicted"] for x in t3)}
    t4 = []
    for sc in res["scaling"]:
        q = predict(sc["prep_cx"], sc["propositions"], sc["evidence_literals"], table)
        t4.append({"propositions": sc["propositions"], "literals": sc["evidence_literals"], "Q_measured": sc["grover_cx"], "Q_predicted": q["Q_cx"]})
    tests["4_resources_scaling"] = {"points": t4, "pass": all(x["Q_measured"] == x["Q_predicted"] for x in t4)}
    ratios_v21 = [x["ratio_measured"] for x in t3]
    out = {"law": {
        "state_preparation": "P = sum over nodes with k >= 1 parents of 2^k CX (one uniformly controlled RY per node)",
        "oracle_in_A": "m literals, f free qubits: f = 0 -> M0(m); f >= ceil(m/2) - 1 -> 8m - 12; otherwise 12m - 18 "
                       "(m = 2: 6, m = 3: 14)",
        "grover_step": "Q = 2 (P + oracle) + M0(n) - s,  s = 2m - 6 (8m-12 regime), 6m - 12 (12m-18 regime), 0 if f = 0",
        "reflection": "M0(n) = 4n^2 - 4n + 2 for 5 <= n <= 32; measured table above (about 124 n)",
        "ratio": "Q / (2A) = 1 + (M0(n) - s) / (2 (P + oracle))"},
        "assumptions": res["assumptions"], "mcx_noancilla_table": table, "tests": tests,
        "ratio_range_v21_profiles": [float(min(ratios_v21)), float(max(ratios_v21))],
        "all_tests_pass": all(t["pass"] for t in tests.values())}
    (OUT / "cost_model.json").write_text(json.dumps(out, indent=1))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 0.8, 1.1]})
    meas = np.array([r["Q_cx"] for r in rows] + [x["Q_measured"] for x in t3 + t4])
    pred = np.array([r["pred_Q"] for r in rows] + [x["Q_predicted"] for x in t3 + t4])
    src = ["grid"] * len(rows) + ["v2.1 profiles"] * len(t3) + ["scaling"] * len(t4)
    cols = (("grid", "#2a78d6"), ("v2.1 profiles", "#eb6834"), ("scaling", "#1baf7a"))
    for lab, c in cols:
        k = [i for i, s_ in enumerate(src) if s_ == lab]
        axs[0].scatter(meas[k], pred[k], s=16, color=c, label=lab, edgecolors="white", linewidths=0.4)
        axs[1].scatter(meas[k], 100 * (pred[k] - meas[k]) / meas[k], s=16, color=c, edgecolors="white", linewidths=0.4)
    lim = [meas.min() * 0.8, meas.max() * 1.2]
    axs[0].plot(lim, lim, color="#6b6a64", lw=1)
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("measured CX per Grover step"); axs[0].set_ylabel("predicted by the law")
    axs[0].set_title(f"Measured vs predicted: {int((meas == pred).sum())}/{len(meas)} exact", loc="left", fontsize=9.5)
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].axhline(0, color="#6b6a64", lw=1)
    axs[1].axhspan(-1, 1, color="#6b6a64", alpha=0.08, lw=0)
    axs[1].set_xscale("log"); axs[1].set_ylim(-1.5, 1.5)
    from matplotlib.ticker import NullFormatter
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set_xlabel("measured CX per Grover step"); axs[1].set_ylabel("residual, % of measured")
    axs[1].set_title("Residuals (band = 1% tolerance)", loc="left", fontsize=9.5)
    R = np.array([r["ratio"] for r in rows]); L = np.array([r["literals"] / r["nodes"] for r in rows])
    sc_ = axs[2].scatter(L, R, c=[r["nodes"] for r in rows], cmap="Blues", s=16, edgecolors="#6b6a64", linewidths=0.3)
    axs[2].scatter([(p["findings"] + 1) / n46 for p in res["profiles"]], ratios_v21, marker="D", color="#eb6834", s=30,
                   label="v2.1 profiles (46 nodes)")
    axs[2].axhline(1, color="#6b6a64", lw=1, ls=":")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("share of nodes that are marked literals"); axs[2].set_ylabel("Grover step / 2 problem preparations (CX)")
    axs[2].set_title("The reflection, with no qubit to borrow, dominates", loc="left", fontsize=9.5)
    fig.colorbar(sc_, ax=axs[2], pad=0.01).set_label("nodes")
    axs[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(Path("reports/v2.1") / "cost_model.png", dpi=180)
    print(json.dumps({k: v for k, v in out.items() if k != "mcx_noancilla_table"}, indent=1))


if __name__ == "__main__":
    {"sweep": sweep, "fit": fit}[sys.argv[1]]()
