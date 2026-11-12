"""W2 of the quantum paper plan: the gate cost law for multi-state networks (qubit-register encoding, scripts/msbn.py).
Counts are exact (qiskit 2.6.1, rz/sx/x/cx, optimization level 1, all-to-all), as for the binary law.

  python scripts/cost_model_ms.py sweep      # random multi-state grid -> artifacts/quantum_paper/cost_grid_ms.json
  python scripts/cost_model_ms.py anchors    # v2.1 structure through the multi-state builder (binary): must equal resources.json
  python scripts/cost_model_ms.py fit        # law + tests -> cost_model_multistate.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from msbn import MSBN, baseline_A, nq, prep_circuit, random_msbn  # noqa: E402
from oracle_v2 import _pad, cx, grover  # noqa: E402

OUT, SEED = Path("artifacts/quantum_paper"), 2026


def measure(bn: MSBN, ev: dict, t: int, s: int) -> dict:
    prep, reg = prep_circuit(bn)
    A0, _ = baseline_A(bn, range(bn.n), ev, t, s)
    lits = sum(len(reg[j]) for j in ev) + len(reg[t])
    ctl = [sum(nq(bn.card[p]) for p in bn.parents[j]) + b for j in range(bn.n) for b in range(nq(bn.card[j]))]
    return {"qubits": prep.num_qubits, "literals": lits, "prep_cx": cx(prep), "ucry_cx_formula": int(sum(2 ** c for c in ctl if c)),
            "A_v0": cx(A0), "Q_v0": cx(grover(A0, 0)), "A_v1": cx(_pad(A0, 1)), "Q_v1": cx(grover(A0, 1)),
            "cards": bn.card, "max_card": max(bn.card), "max_parents": max(len(p) for p in bn.parents)}


def sweep():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, t0 = [], time.perf_counter()
    for n in (6, 9, 12):
        for cmax in (2, 3, 4, 5):
            for k in (1, 2, 3, 4):
                bn = random_msbn(n, cmax, k, rng)
                order = rng.permutation(n).tolist()
                for L in sorted({1, n // 4, n // 2, 3 * n // 4, n - 1}):
                    x = [int(rng.integers(c)) for c in bn.card]
                    ev = {j: x[j] for j in order[:L]}
                    t = order[L]
                    r = measure(bn, ev, t, int(rng.integers(bn.card[t])))
                    r.update(n=n, card_max=cmax, k_max=k, evidence_registers=L)
                    rows.append(r)
        print(f"n={n} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    (OUT / "cost_grid_ms.json").write_text(json.dumps(rows, indent=1))


def anchors():
    """The v2.1 structure (binary, parents only; CPT values do not change counts) through the multi-state builder,
    evidence = demographics, history, drugs and the first labs, target = one diagnosis (as oracle_v2.count)."""
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    names, parents = st["names"], [tuple(p) for p in st["parents"]]
    rr = np.random.default_rng(SEED)
    cpts = []
    for pa in parents:
        p = rr.uniform(0.05, 0.95, 2 ** len(pa))
        cpts.append(np.stack([1 - p, p], 1).reshape((2,) * len(pa) + (2,)))
    bn = MSBN(names, [2] * len(names), parents, cpts)
    labs = [i for i, nm in enumerate(names) if nm.startswith("lab_")]
    dx = [i for i, nm in enumerate(names) if nm in ("heart_failure", "afib", "uti", "pneumonia", "aki", "anemia")]
    other = [i for i in range(len(names)) if i not in dx and i not in labs]
    out = []
    for f in (27, 38, 39):
        ev = {i: 0 for i in other + labs[: f - len(other)]}
        r = measure(bn, ev, dx[0], 1)
        out.append({"findings": f, **{k: r[k] for k in ("prep_cx", "A_v0", "Q_v0", "A_v1", "Q_v1")}})
        print(out[-1], flush=True)
    (OUT / "cost_anchors_ms.json").write_text(json.dumps(out, indent=1))


def law(prep_cx: int, qubits: int, literals: int, table: dict) -> dict:
    """The generalised law. Nodes become qubits and literals become evidence-register qubits; otherwise identical to
    the binary law (cost_model.predict) for V0. V1 (one clean ancilla): oracle 6(L - 1) for L >= 2, reflection 6(n - 1)."""
    from cost_model import predict
    v0 = predict(prep_cx, qubits, literals, table)
    o1 = 6 * (literals - 1) if literals >= 2 else literals
    A1 = prep_cx + o1
    return {"A_v0": v0["A_cx"], "Q_v0": v0["Q_cx"], "A_v1": A1, "Q_v1": 2 * A1 + 6 * (qubits - 1),
            "ratio_v0": v0["Q_cx"] / (2 * v0["A_cx"]), "ratio_v1": (2 * A1 + 6 * (qubits - 1)) / (2 * v0["A_cx"])}


def fit():
    from cost_model import mcx_noancilla_table, predict
    rows = json.loads((OUT / "cost_grid_ms.json").read_text())
    table = mcx_noancilla_table()
    errs, prep_dev = {k: [] for k in ("A_v0", "Q_v0", "A_v1", "Q_v1")}, []
    for r in rows:
        prep_dev.append(r["prep_cx"] - r["ucry_cx_formula"])
        L = law(r["prep_cx"], r["qubits"], r["literals"], table)
        for k in errs:
            errs[k].append(abs(L[k] - r[k]) / r[k] if r[k] else float(L[k] != r[k]))
        r["pred"] = L
    tests = {"1_grid": {k: {"max_rel_error": float(max(v)), "exact": int(sum(e == 0 for e in v)), "n": len(v)} for k, v in errs.items()}}
    tests["1_grid"]["pass_within_1pct"] = all(max(v) <= 0.01 for v in errs.values())
    tests["state_prep_formula"] = {"rows_equal_to_sum_2^c": int(sum(d == 0 for d in prep_dev)), "n": len(prep_dev),
                                   "max_shortfall": int(-min(prep_dev)), "max_excess": int(max(prep_dev)),
                                   "note": "formula: sum over register qubits with c >= 1 controls of 2^c, c = parent qubits + earlier own qubits"}
    # the saving s (A-dagger's oracle partly cancelling against the reflection) is a peephole the compiler finds in
    # most circuits but not all: every grid point is exactly one of the two values Q = 2A + M0(n) - s or 2A + M0(n)
    from cost_model import m0, oracle_cx
    two, applied, eligible = 0, 0, 0
    for r in rows:
        o, save, _ = oracle_cx(r["literals"], r["qubits"] - r["literals"], table)
        smeas = m0(r["qubits"], table) + 2 * r["A_v0"] - r["Q_v0"]
        two += smeas in (0, save)
        eligible += save > 0
        applied += save > 0 and smeas == save
    tests["1b_two_valued_saving"] = {"points_exactly_one_of_two_values": two, "n": len(rows), "saving_applied": applied,
                                     "saving_eligible": eligible, "pass": two == len(rows)}
    b = [r for r in rows if r["card_max"] == 2]
    bm = [r["Q_v0"] == predict(r["prep_cx"], r["qubits"], r["literals"], table)["Q_cx"] for r in b]
    tests["2_binary_rows_vs_binary_law_measured"] = {"rows": len(b), "exact": int(sum(bm)),
                                                     "note": "binary rows of this grid (6 and 9 nodes, smaller than the step-5 grid); misses are all the saving term"}
    an = json.loads((OUT / "cost_anchors_ms.json").read_text())
    want = {27: (800, 7374), 38: (1038, 7678), 39: (1050, 7696)}
    tests["3_v21_anchors_through_multistate_builder"] = {"points": an, "pass": all(a["prep_cx"] == 588 and (a["A_v0"], a["Q_v0"]) == want[a["findings"]] for a in an),
                                                         "v1_law_matches": all((law(588, 46, a["findings"] + 1, table)["A_v1"], law(588, 46, a["findings"] + 1, table)["Q_v1"]) == (a["A_v1"], a["Q_v1"]) for a in an)}
    sc = [{"propositions": 16, "literals": 14, "prep": 142, "Q": 1474}, {"propositions": 20, "literals": 17, "prep": 208, "Q": 2220}]
    tests["4_scaling_entries"] = {"points": [{**x, "Q_law": law(x["prep"], x["propositions"], x["literals"], table)["Q_v0"]} for x in sc]}
    tests["4_scaling_entries"]["pass"] = all(x["Q_law"] == x["Q"] for x in tests["4_scaling_entries"]["points"])
    out = {"law": {
        "registers": "node with k states -> m = max(1, ceil(log2 k)) qubits; states k..2^m-1 have zero amplitude",
        "state_preparation": "P = sum over register qubits of 2^c (c >= 1), c = parent-register qubits + earlier qubits of the same register; "
                             "measured counts can fall below this when structural zero angles let the compiler cancel gates (see test)",
        "v0": "binary law with nodes -> qubits n and literals -> evidence-register qubits L (incl. the target register); "
              "the saving s is applied by the compiler in most circuits but not all, so Q_v0 is exactly 2A + M0(n) - s or 2A + M0(n)",
        "v1": "A1 = P + 6(L - 1); Q1 = 2 A1 + 6(n - 1)",
        "ratio": "Q / (2 A_v0)"},
        "tests": tests, "qiskit": "2.6.1", "grid_points": len(rows)}
    (OUT / "cost_model_multistate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(tests, indent=1))


if __name__ == "__main__":
    {"sweep": sweep, "anchors": anchors, "fit": fit}[sys.argv[1]]()
