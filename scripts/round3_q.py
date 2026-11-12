"""Round 3 Q5 (measurement-based uncomputation) and Q6 (cost-law exceptions); tasks/plan_round3.md. Cloud workspace,
structure-only / synthetic inputs.

  python scripts/round3_q.py mbu          # correctness on 30 random multi-state cases, expected CX on anchors + grid
  python scripts/round3_q.py exceptions   # what separates the no-saving grid circuits; the step-5 failure
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate, grover_operator
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).parent))
from msbn import MSBN, baseline_A, forward_sample, nq, random_msbn  # noqa: E402
from oracle_v2 import cx  # noqa: E402

OUT, SEED = Path("artifacts/quantum_paper"), 2026


def _literals_of(A: QuantumCircuit):
    """(control qubits, control state) of the oracle MCX: A's last instruction."""
    inst = A.data[-1]
    op = inst.operation
    ctrl = [A.find_bit(q).index for q in inst.qubits[:-1]]
    return ctrl, op.ctrl_state


def _mcz(n_q: int, ctrl: list, state: int) -> QuantumCircuit:
    """Phase (-1) on |ctrl = state>: X on the zero literals, a multi-controlled Z, X back."""
    qc = QuantumCircuit(n_q)
    zeros = [q for i, q in enumerate(ctrl) if not (state >> i) & 1]
    for q in zeros:
        qc.x(q)
    if len(ctrl) == 1:
        qc.z(ctrl[0])
    else:
        qc.h(ctrl[-1])
        qc.append(MCXGate(len(ctrl) - 1), ctrl[:-1] + [ctrl[-1]])
        qc.h(ctrl[-1])
    for q in zeros:
        qc.x(q)
    return qc


def grover_parts(A: QuantumCircuit, clean: int = 1):
    """The V1 Grover step written out: S_chi, A-dagger, S0, A (as grover_operator orders it), and its MBU branches.
    MBU: A-dagger's first gate (the oracle MCX, uncomputing the flag) becomes H(flag), measure; outcome 1 applies
    the multi-controlled Z on the literals and X(flag). Returns (Q_standard, Q_branch0, Q_branch1)."""
    n = A.num_qubits
    flag = n - 1
    ctrl, state = _literals_of(A)
    prep = A.copy()
    prep.data = prep.data[:-1]
    W = n + clean

    def pad(c):
        q = QuantumCircuit(W)
        q.compose(c, range(c.num_qubits), inplace=True)
        return q
    orc = QuantumCircuit(W)
    orc.z(flag)
    Ap = pad(A)
    std = grover_operator(orc, state_preparation=Ap, reflection_qubits=list(range(n)))
    refl = grover_operator(QuantumCircuit(W), state_preparation=QuantumCircuit(W), reflection_qubits=list(range(n)))  # S0 only
    out = []
    for b in (0, 1):
        q = QuantumCircuit(W)
        q.z(flag)
        q.h(flag)
        if b:
            q.compose(_mcz(W, ctrl, state), inplace=True)
            q.x(flag)
        q.compose(pad(prep).inverse(), inplace=True)
        q.compose(refl, inplace=True)
        q.compose(Ap, inplace=True)
        out.append(q)
    return std, out[0], out[1]


def mbu():
    rng = np.random.default_rng(SEED)
    worst_state, worst_prob, cases = 0.0, 0.0, 0
    while cases < 30:
        bn = random_msbn(int(rng.integers(3, 7)), 4, 3, rng)
        x = forward_sample(bn, rng)
        L = int(rng.integers(1, bn.n))
        order = rng.permutation(bn.n)
        ev = {int(j): x[j] for j in order[:L]}
        t = int(order[L])
        A, _ = baseline_A(bn, range(bn.n), ev, t, int(rng.integers(bn.card[t])))
        if A.num_qubits + 1 > 14:
            continue
        n, flag = A.num_qubits, A.num_qubits - 1
        W = n + 1
        ctrl, state = _literals_of(A)
        start = QuantumCircuit(W)
        start.compose(A, range(n), inplace=True)
        start.z(flag)
        psi = Statevector(start)
        # coherent uncompute (the oracle MCX)
        u = QuantumCircuit(W)
        u.append(A.data[-1].operation, [A.find_bit(q).index for q in A.data[-1].qubits])
        ref = psi.evolve(u)
        h = QuantumCircuit(W)
        h.h(flag)
        phi = psi.evolve(h)
        for b in (0, 1):
            proj = np.array(phi.data)
            idx = np.arange(len(proj))
            proj[((idx >> flag) & 1) != b] = 0
            pb = float(np.vdot(proj, proj).real)
            worst_prob = max(worst_prob, abs(pb - 0.5))
            st = Statevector(proj / math.sqrt(pb))
            if b:
                fix = _mcz(W, ctrl, state)
                fix.x(flag)
                st = st.evolve(fix)
            ov = abs(np.vdot(st.data, ref.data))
            worst_state = max(worst_state, 1 - ov)
        cases += 1
    print("correctness: max 1 - |overlap|", worst_state, "max |P(outcome) - 0.5|", worst_prob, flush=True)
    rows = []
    # v2.1 anchors (structure only)
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    names, parents = st["names"], [tuple(p) for p in st["parents"]]
    rr = np.random.default_rng(SEED)
    cpts = [np.stack([1 - p, p], 1).reshape((2,) * len(pa) + (2,)) for pa in parents for p in [rr.uniform(0.05, 0.95, 2 ** len(pa))]]
    b21 = MSBN(names, [2] * len(names), parents, cpts)
    labs = [i for i, nm in enumerate(names) if nm.startswith("lab_")]
    dx = [i for i, nm in enumerate(names) if nm in ("heart_failure", "afib", "uti", "pneumonia", "aki", "anemia")]
    other = [i for i in range(len(names)) if i not in dx and i not in labs]
    cases = [(f"v2.1_findings{f}", b21, {i: 0 for i in other + labs[: f - len(other)]}, dx[0], 1) for f in (27, 38, 39)]
    rng = np.random.default_rng(SEED)
    for k in range(12):  # multi-state grid sample (as cost_model_ms sweep sizes)
        bn = random_msbn(int(rng.choice([6, 9, 12])), int(rng.integers(2, 6)), int(rng.integers(1, 5)), rng)
        x = [int(rng.integers(c)) for c in bn.card]
        order = rng.permutation(bn.n).tolist()
        L = int(rng.integers(1, bn.n))
        cases.append((f"grid_{k}", bn, {j: x[j] for j in order[:L]}, order[L], int(rng.integers(bn.card[order[L]]))))
    for name, bn, ev, t, s in cases:
        A, _ = baseline_A(bn, range(bn.n), ev, t, s)
        A0 = cx(A)
        std, q0, q1 = grover_parts(A, 1)
        r = {"case": name, "literals": len(_literals_of(A)[0]), "A_v0": A0, "Q_v1": cx(std), "Q_mbu_branch0": cx(q0), "Q_mbu_branch1": cx(q1)}
        r["Q_mbu_expected"] = (r["Q_mbu_branch0"] + r["Q_mbu_branch1"]) / 2
        r["ratio_v1"], r["ratio_v1_mbu"] = r["Q_v1"] / (2 * A0), r["Q_mbu_expected"] / (2 * A0)
        r["saving_share"] = 1 - r["Q_mbu_expected"] / r["Q_v1"]
        rows.append(r)
        print(r, flush=True)
    out = {"correctness": {"cases": 30, "max_one_minus_overlap": worst_state, "max_abs_outcome_prob_minus_half": worst_prob,
                           "pass": worst_state < 1e-10 and worst_prob < 1e-10}, "rows": rows,
           "note": "expected CX = 1/2 (branch 0 + branch 1); one mid-circuit measurement and one classically controlled fix-up per Grover step"}
    (OUT / "mbu.json").write_text(json.dumps(out, indent=1))


def _grid_cases():
    """Replays cost_model_ms.sweep's random draws: the same 224 networks and instances, in order."""
    rng = np.random.default_rng(SEED)
    for n in (6, 9, 12):
        for cmax in (2, 3, 4, 5):
            for k in (1, 2, 3, 4):
                bn = random_msbn(n, cmax, k, rng)
                order = rng.permutation(n).tolist()
                for L in sorted({1, n // 4, n // 2, 3 * n // 4, n - 1}):
                    x = [int(rng.integers(c)) for c in bn.card]
                    ev = {j: x[j] for j in order[:L]}
                    t = order[L]
                    yield bn, ev, t, int(rng.integers(bn.card[t])), dict(n=n, card_max=cmax, k_max=k, evidence_registers=L)


def _segments(A: QuantumCircuit) -> dict:
    """CX of the V0 Grover step and of its pieces, to locate where the saving s is taken."""
    n = A.num_qubits
    orc = QuantumCircuit(n)
    orc.z(n - 1)
    empty = QuantumCircuit(n)
    S0 = grover_operator(empty, state_preparation=empty, reflection_qubits=list(range(n)))
    Ad = A.inverse()
    def seq(*parts):
        q = QuantumCircuit(n)
        for p_ in parts:
            q.compose(p_, inplace=True)
        return cx(q)
    return {"Q": cx(grover_operator(orc, state_preparation=A, reflection_qubits=list(range(n)))),
            "A": cx(A), "S0": cx(S0), "Ad_S0": seq(Ad, S0), "S0_A": seq(S0, A), "Ad_S0_A": seq(Ad, S0, A),
            "chi_Ad": seq(orc, Ad)}


def exceptions():
    rows = json.loads((OUT / "cost_grid_ms.json").read_text())
    from cost_model import mcx_noancilla_table, m0, oracle_cx
    table = mcx_noancilla_table()
    out = []
    for i, (bn, ev, t, s, meta) in enumerate(_grid_cases()):
        r = rows[i]
        assert all(r[k] == meta[k] for k in meta), (i, r, meta)
        o, save, regime = oracle_cx(r["literals"], r["qubits"] - r["literals"], table)
        smeas = m0(r["qubits"], table) + 2 * r["A_v0"] - r["Q_v0"]
        if save == 0:
            continue
        A, reg = baseline_A(bn, range(bn.n), ev, t, s)
        assert cx(A) == r["A_v0"]
        seg = _segments(A)
        ctrl, _ = _literals_of(A)
        nqb = A.num_qubits - 1
        rec = {"i": i, **meta, "qubits": r["qubits"], "literals": r["literals"], "regime": regime, "saving_applied": smeas == save,
               "s": save, "seg": seg,
               # candidate explanations
               "last_node_qubit_is_literal": (nqb - 1) in ctrl,
               "target_is_last_register": max(reg[t]) == nqb - 1,
               "first_qubit_is_literal": 0 in ctrl,
               "last_prep_gate_touches_literal": any(A.find_bit(q).index in ctrl for q in A.data[-2].qubits) if len(A.data) > 1 else None,
               "last_prep_gate_qubits": [A.find_bit(q).index for q in A.data[-2].qubits] if len(A.data) > 1 else None,
               "last_prep_gate": A.data[-2].operation.name if len(A.data) > 1 else None}
        out.append(rec)
        print(i, rec["saving_applied"], {k: v for k, v in rec.items() if k in ("qubits", "literals", "last_node_qubit_is_literal", "target_is_last_register", "last_prep_gate_touches_literal")},
              "Q-(2A+S0)", seg["Q"] - 2 * seg["A"] - seg["S0"], "AdS0-(A+S0)", seg["Ad_S0"] - seg["A"] - seg["S0"], "S0A-(A+S0)", seg["S0_A"] - seg["A"] - seg["S0"], flush=True)
    (OUT / "cost_exceptions.json").write_text(json.dumps(out, indent=1, default=str))




def in_context():
    """The Grover step as amplitude estimation executes it (after A, and after earlier steps), against the standalone
    Q the cost law was fitted to: (a) the v2.1 anchors, V0 and V1; (b) the step-5 grid point that missed (row 0)."""
    from qddx.bn import Evidence, with_virtual
    from qddx.quantum import conj_oracle, problem_circuit, state_prep
    import cost_model as CM
    from oracle_v2 import grover, _pad

    def ctx(A, clean):
        W = A.num_qubits + clean
        Q = grover(A, clean)
        a = QuantumCircuit(W)
        a.compose(A, range(A.num_qubits), inplace=True)
        aq = a.copy()
        aq.compose(Q, inplace=True)
        aqq = aq.copy()
        aqq.compose(Q, inplace=True)
        r = {"standalone_Q": cx(Q), "in_context_first_Q": cx(aq) - cx(a), "in_context_second_Q": cx(aqq) - cx(aq), "A": cx(a)}
        if clean:  # MBU, each outcome branch in context, averaged
            _, b0, b1 = grover_parts(A, 1)
            inc = []
            for q in (b0, b1):
                x1 = a.copy(); x1.compose(q, inplace=True)
                x2 = x1.copy(); x2.compose(q, inplace=True)
                inc.append(cx(x2) - cx(x1))
            r["mbu_in_context_second_Q"] = sum(inc) / 2
        return r
    out = {}
    st = json.loads(Path("artifacts/v2.1/kb/structure_only.json").read_text())
    names, parents = st["names"], [tuple(p) for p in st["parents"]]
    rr = np.random.default_rng(SEED)
    cpts = [np.stack([1 - p, p], 1).reshape((2,) * len(pa) + (2,)) for pa in parents for p in [rr.uniform(0.05, 0.95, 2 ** len(pa))]]
    b21 = MSBN(names, [2] * len(names), parents, cpts)
    labs = [i for i, nm in enumerate(names) if nm.startswith("lab_")]
    dx = [i for i, nm in enumerate(names) if nm in ("heart_failure", "afib", "uti", "pneumonia", "aki", "anemia")]
    other = [i for i in range(len(names)) if i not in dx and i not in labs]
    for f in (27, 38, 39):
        A, _ = baseline_A(b21, range(b21.n), {i: 0 for i in other + labs[: f - len(other)]}, dx[0], 1)
        for var, clean in (("v0", 0), ("v1", 1)):
            out[f"v2.1_findings{f}|{var}"] = r = ctx(A, clean)
            r["ratio_standalone"] = r["standalone_Q"] / (2 * cx(A))
            r["ratio_in_context"] = r["in_context_second_Q"] / (2 * cx(A))
            if "mbu_in_context_second_Q" in r:
                r["ratio_mbu_in_context"] = r["mbu_in_context_second_Q"] / (2 * cx(A))
            print(f, var, r, flush=True)
    rng = np.random.default_rng(SEED)
    b = CM.random_bn(12, 1, rng)
    order = rng.permutation(12).tolist()
    hard = {f"x{j}": int(rng.integers(0, 2)) for j in order[:2]}
    ext, obs = with_virtual(b, Evidence(hard, {}))
    A = problem_circuit(state_prep(ext), conj_oracle(ext.struct.n, obs))
    out["step5_grid_row0"] = r = ctx(A, 0)
    r["Ad"] = cx(A.inverse())
    print("step-5 row 0", r, flush=True)
    (OUT / "cost_in_context.json").write_text(json.dumps(out, indent=1))




def grid_in_context():
    """Steady-state Grover-step CX as executed (A, Q, Q: the second Q's increment) on the 224 multi-state grid points,
    V0 and V1, plus V1 with MBU (each outcome branch in context, averaged)."""
    from oracle_v2 import grover
    rows = json.loads((OUT / "cost_grid_ms.json").read_text())
    out = []
    for i, (bn, ev, t, s, meta) in enumerate(_grid_cases()):
        A, _ = baseline_A(bn, range(bn.n), ev, t, s)
        rec = {"i": i, **meta, "qubits": rows[i]["qubits"], "literals": rows[i]["literals"], "A_v0": rows[i]["A_v0"],
               "Q_v0_standalone": rows[i]["Q_v0"], "Q_v1_standalone": rows[i]["Q_v1"], "prep_cx": rows[i]["prep_cx"]}
        for var, clean in (("v0", 0), ("v1", 1)):
            W = A.num_qubits + clean
            a = QuantumCircuit(W)
            a.compose(A, range(A.num_qubits), inplace=True)
            Q = grover(A, clean)
            aq = a.copy(); aq.compose(Q, inplace=True)
            aqq = aq.copy(); aqq.compose(Q, inplace=True)
            rec[f"Q_{var}_in_context"] = cx(aqq) - cx(aq)
        _, b0, b1 = grover_parts(A, 1)
        a = QuantumCircuit(A.num_qubits + 1)
        a.compose(A, range(A.num_qubits), inplace=True)
        inc = []
        for q in (b0, b1):
            aq = a.copy(); aq.compose(q, inplace=True)
            aqq = aq.copy(); aqq.compose(q, inplace=True)
            inc.append(cx(aqq) - cx(aq))
        rec["Q_v1_mbu_in_context"] = sum(inc) / 2
        out.append(rec)
        if i % 20 == 0:
            print(i, rec, flush=True)
            (OUT / "cost_grid_in_context.json").write_text(json.dumps(out, indent=1))
    (OUT / "cost_grid_in_context.json").write_text(json.dumps(out, indent=1))
    print("done", flush=True)



def kappa_map(grid="w8_kappa_grid_0.json"):
    """Q3: the kappa map (median over seeds of log10 kappa per cell, bilinear in log a_E, log p; empty cells take the
    nearest filled cell) and its test on the 50 benchmark calibration instances."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    g = json.loads((OUT / grid).read_text())
    ae, pp = sorted({r["a_E"] for r in g}), sorted({r["p"] for r in g})
    out = {"grid": grid, "a_E": ae, "p": pp, "estimators": {}}
    rows = json.loads((OUT / "w8_kappa.json").read_text())["rows"]
    for est, col in (("kappa_iae", "kappa_iae_16_0.2"), ("kappa_mlae16", "kappa_mlae_N16")):
        M = np.full((len(ae), len(pp)), np.nan)
        S = np.full_like(M, np.nan)
        for i, a in enumerate(ae):
            for j, p in enumerate(pp):
                v = [np.log10(r[est]) for r in g if r["a_E"] == a and r["p"] == p and r.get(est)]
                if v:
                    M[i, j] = np.median(v)
                    S[i, j] = np.median(np.abs(np.array(v) - np.median(v))) if len(v) > 1 else np.nan
        filled = np.isfinite(M)
        _, (ii, jj) = distance_transform_edt(~filled, return_indices=True)
        Mf = M[ii, jj]
        f = RegularGridInterpolator((np.log10(ae), np.log10(pp)), Mf, bounds_error=False, fill_value=None)
        pred = lambda a, p: float(10 ** f([[np.log10(min(max(a, ae[0]), ae[-1])), np.log10(min(max(p, pp[0]), pp[-1]))]])[0])
        err, meas, prd = [], [], []
        for r in rows:
            if r.get(col):
                k = pred(r["a_E"], r["p"])
                err.append(abs(np.log10(k) - np.log10(r[col])))
                meas.append(r[col]); prd.append(k)
        # spread over the region the instances occupy (grid cells inside the instances' a_E, p hull box)
        A_lo, A_hi = min(r["a_E"] for r in rows), max(r["a_E"] for r in rows)
        P_lo, P_hi = min(r["p"] for r in rows), max(r["p"] for r in rows)
        box = Mf[np.ix_([i for i, a in enumerate(ae) if A_lo / 1.8 <= a <= A_hi * 1.8], [j for j, p in enumerate(pp) if P_lo / 1.5 <= p <= P_hi * 1.5])]
        out["estimators"][est] = {
            "log10_kappa": M.tolist(), "seed_mad_log10_median": float(np.nanmedian(S)), "empty_cells": int((~filled).sum()),
            "benchmark_test": {"n": len(err), "median_abs_log10_error": float(np.median(err)),
                               "measured_median": float(np.median(meas)), "predicted_median": float(np.median(prd)),
                               "median_ratio_pred_over_meas": float(np.median(prd) / np.median(meas))},
            "occupied_region_kappa_range": [float(10 ** box.min()), float(10 ** box.max())],
            "occupied_region_max_over_min": float(10 ** (box.max() - box.min()))}
        print(est, json.dumps({k: v for k, v in out["estimators"][est].items() if k != "log10_kappa"}), flush=True)
    (OUT / "kappa_map.json").write_text(json.dumps(out, indent=1))


def _map_fn(est):
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    m = json.loads((OUT / "kappa_map.json").read_text())
    M = np.array(m["estimators"][est]["log10_kappa"], float)
    _, (ii, jj) = distance_transform_edt(~np.isfinite(M), return_indices=True)
    f = RegularGridInterpolator((np.log10(m["a_E"]), np.log10(m["p"])), M[ii, jj])
    la, lp = np.log10(m["a_E"]), np.log10(m["p"])
    return lambda aE, p: 10 ** f(np.column_stack([np.clip(np.log10(aE), la[0], la[-1]), np.clip(np.log10(p), lp[0], lp[-1])]))


def w4_kappa():
    """Q2 (benchmarks): the W4 statistics with kappa_MLAE (7.1, benchmark calibration) and with instance-specific kappa
    from the Q3 map, against the published version (tuned IAE, clinical constant 11.69). R is linear in kappa."""
    import benchmarks as BM
    _, ok = BM._frame(BM.NETWORKS)
    k0 = BM.KAPPA
    kb = json.loads((OUT / "w8_kappa.json").read_text())["estimators"]
    variants = {"published: IAE, constant 11.69 (clinical)": np.full(len(ok), k0),
                "IAE, constant 24.0 (benchmark calibration)": np.full(len(ok), kb["kappa_iae_16_0.2"]["median"]),
                "MLAE16, constant 7.10 (benchmark calibration)": np.full(len(ok), kb["kappa_mlae_N16"]["median"]),
                "IAE, per instance (Q3 map)": _map_fn("kappa_iae")(ok.a_E.to_numpy(), ok.p.to_numpy()),
                "MLAE16, per instance (Q3 map; map fails its test)": _map_fn("kappa_mlae16")(ok.a_E.to_numpy(), ok.p.to_numpy()),
                "best of IAE/MLAE16 per instance (Q3 map)": np.minimum(_map_fn("kappa_iae")(ok.a_E.to_numpy(), ok.p.to_numpy()),
                                                                     _map_fn("kappa_mlae16")(ok.a_E.to_numpy(), ok.p.to_numpy()))}
    out = {}
    for name, k in variants.items():
        d = ok.copy()
        d[BM.R_COL] = d[BM.R_COL] * k / k0
        d["logR"] = np.log10(d[BM.R_COL])
        st = BM.w4_stats(d)
        out[name] = {"median_kappa": float(np.median(k)), "share_R_below_1": float((d[BM.R_COL] < 1).mean()),
                     "median_log10_R": float(d.logR.median()), "primary_pooled_spearman": st["primary_pooled_spearman"],
                     "within_network_rank_r": st["within_network_pooled_rank_correlation"],
                     "secondary_R_below_1": {a: st["secondary_R_below_1"][a] for a in ("instances", "share_cost_below_network_median", "pass")}}
        print(name, json.dumps(out[name]), flush=True)
    (OUT / "w4_kappa.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"mbu": mbu, "exceptions": exceptions, "in-context": in_context, "grid-in-context": grid_in_context, "kappa-map": kappa_map, "w4-kappa": w4_kappa}[sys.argv[1]]()
