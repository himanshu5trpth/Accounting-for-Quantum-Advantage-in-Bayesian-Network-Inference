"""Corrections before writing (tasks/correction_plan.md). Run from the repo root; each step writes to
artifacts/corrections/ and prints a short summary.

Order (the plan's): item 3, then item 2, then item 1.
  python scripts/corrections.py noevidence            # 3.1-3.3  benchmarks: |E| = 0 instances, stats with and without
  python scripts/corrections.py noevidence-clinical   # 3.4      W6: instances that lose all evidence (local VM)
  python scripts/corrections.py law-context           # 2.1      in-context counts against the standalone law, refits
  python scripts/corrections.py variants-context      # 2.2      V0-V3 per Grover step in context (needs pgmpy)
  python scripts/corrections.py compile-context qiskit   # 2.3   in-context compiler sweep, one run per qiskit install
  python scripts/corrections.py compile-context tket     # 2.3   ... and pytket
  python scripts/corrections.py compare-context       # 2.3      V1 < V0 in every combination, in context?
  python scripts/corrections.py sweep                 # 1.2      every occurrence of the kappa-sensitive numbers
  python scripts/corrections.py headline              # checkpoint: headline numbers, source, kappa and cost convention
"""
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.stdout.reconfigure(errors="replace")  # Windows consoles may not encode κ
QP, OUT = Path("artifacts/quantum_paper"), Path("artifacts/corrections")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026


def _save(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))
    print(f"-> {OUT / name}")


# ===================================================================================== item 3: no-evidence instances
def _kappa_map():
    """The Q3 IAE map as a vectorised function of (a_E, p)."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    m = json.loads((QP / "kappa_map.json").read_text())
    M = np.array(m["estimators"]["kappa_iae"]["log10_kappa"], float)
    _, (ii, jj) = distance_transform_edt(~np.isfinite(M), return_indices=True)
    f = RegularGridInterpolator((np.log10(m["a_E"]), np.log10(m["p"])), M[ii, jj])
    la, lp = np.log10(m["a_E"]), np.log10(m["p"])
    return lambda aE, p: 10 ** f(np.column_stack([np.clip(np.log10(aE), la[0], la[-1]), np.clip(np.log10(p), lp[0], lp[-1])]))


def _bench_stats(ok: pd.DataFrame) -> dict:
    """Every primary statistic the conclusions rest on (W4 + the p mechanism), network-bootstrap CIs."""
    import benchmarks as BM
    from scipy.stats import rankdata
    st = BM.w4_stats(ok)

    def partial_p(g):  # rank partial correlation of log R with log p, given log exact cost
        x, y, z = rankdata(g.logR), rankdata(np.log10(g.p)), rankdata(g.cost)
        Z = np.column_stack([np.ones(len(z)), z])
        rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
        ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        return float(np.corrcoef(rx, ry)[0, 1])

    def within_p(g):
        r = BM._within_ranks(g.assign(lp=np.log10(g.p)), ["lp", "logR"])
        return float(np.corrcoef(r.lp, r.logR)[0, 1])
    return {"instances": int(len(ok)), "networks": int(ok.network.nunique()),
            "share_R_below_1": float((ok[BM.R_COL] < 1).mean()), "median_log10_R": float(ok.logR.median()),
            "pooled_spearman_cost": st["primary_pooled_spearman"],
            "within_network_rank_r_cost": st["within_network_pooled_rank_correlation"],
            "secondary_R_below_1": {k: st["secondary_R_below_1"][k] for k in ("instances", "share_cost_below_network_median", "pass")},
            "pooled_spearman_p": BM._spearman(np.log10(ok.p), ok.logR),
            "within_network_rank_r_p": {"r": within_p(ok), "ci95_network_bootstrap": BM._net_boot(ok, within_p)},
            "partial_spearman_p_given_cost": {"r": partial_p(ok), "ci95_network_bootstrap": BM._net_boot(ok, partial_p)}}


def _conclusions(s: dict) -> dict:
    """The conclusions the paper states, as booleans (fixed before the numbers were looked at)."""
    return {"primary pooled Spearman negative, CI excludes 0": bool(s["pooled_spearman_cost"]["rho"] < 0 and s["pooled_spearman_cost"]["ci95_network_bootstrap"][1] < 0),
            "no within-network cost effect (CI contains 0)": bool(s["within_network_rank_r_cost"]["ci95_network_bootstrap"][0] <= 0 <= s["within_network_rank_r_cost"]["ci95_network_bootstrap"][1]),
            "secondary fails (< 0.9)": not s["secondary_R_below_1"]["pass"],
            "R tracks p given cost (partial >= 0.7)": bool(s["partial_spearman_p_given_cost"]["r"] >= 0.7),
            "quantum cheaper in a minority of instances": bool(s["share_R_below_1"] < 0.5)}


def noevidence():
    """3.1 count |E| = 0 as generated and after pruning, per network; 3.2 the primary statistics with and without;
    3.3 the decision rule. kappa: per-estimator calibrated constant (benchmark IAE, primary) and the IAE map."""
    import benchmarks as BM
    _, ok = BM._frame(BM.NETWORKS)
    kb = json.loads((QP / "w8_kappa.json").read_text())["estimators"]["kappa_iae_16_0.2"]["median"]
    count = {"pooled": {}, "by_network": {}}
    for name, g in [("pooled", ok)] + list(ok.groupby("network")):
        c = {"instances": int(len(g)), "evidence_given_0": int((g.evidence_given == 0).sum()),
             "evidence_used_0": int((g.evidence_used == 0).sum()), "share_evidence_used_0": float((g.evidence_used == 0).mean()),
             "a_E_equal_1": int((g.a_E >= 1 - 1e-12).sum()),
             "evidence_used_quantiles_10_25_50_75_90": [float(x) for x in np.quantile(g.evidence_used, [.1, .25, .5, .75, .9])],
             "evidence_given_median": float(g.evidence_given.median())}
        if name == "pooled":
            count["pooled"] = c
            count["evidence_used_histogram"] = {int(k): int(v) for k, v in g.evidence_used.value_counts().sort_index().items()}
        else:
            count["by_network"][name] = c
    out = {"count": count, "kappa_conventions": {}, "decision": {}}
    kmap = _kappa_map()
    for kname, k in (("benchmark IAE constant (primary)", np.full(len(ok), kb)),
                     ("IAE per instance, Q3 map (secondary)", kmap(ok.a_E.to_numpy(), ok.p.to_numpy()))):
        d = ok.copy()
        d[BM.R_COL] = d[BM.R_COL] * k / BM.KAPPA  # R is linear in kappa for a fixed instance
        d["logR"] = np.log10(d[BM.R_COL])
        full, sub = _bench_stats(d), _bench_stats(d[d.evidence_used > 0].reset_index(drop=True))
        cf, cs = _conclusions(full), _conclusions(sub)
        changed = [c for c in cf if cf[c] != cs[c]]
        out["kappa_conventions"][kname] = {"all_instances": full, "evidence_bearing_only": sub,
                                           "conclusions_all": cf, "conclusions_evidence_bearing": cs, "changed": changed}
        print(f"\n[{kname}]  kappa median {np.median(k):.2f}")
        for lab, s in (("all", full), ("evidence-bearing", sub)):
            print(f"  {lab:17s} n={s['instances']:6d}  R<1 {s['share_R_below_1']:.3f}  med log R {s['median_log10_R']:+.3f}  "
                  f"rho_cost {s['pooled_spearman_cost']['rho']:+.3f} {np.round(s['pooled_spearman_cost']['ci95_network_bootstrap'], 3)}  "
                  f"within_cost {s['within_network_rank_r_cost']['r']:+.3f} {np.round(s['within_network_rank_r_cost']['ci95_network_bootstrap'], 3)}  "
                  f"partial_p|cost {s['partial_spearman_p_given_cost']['r']:.3f}")
        print("  conclusions changed:", changed or "none")
    prim = out["kappa_conventions"]["benchmark IAE constant (primary)"]["changed"]
    out["decision"] = {"rule": "3.3: if every conclusion is unchanged, keep all instances and report the sensitivity in one line; "
                               "otherwise the evidence-bearing subset becomes primary. Applied to the primary kappa convention.",
                       "conclusions_changed": prim,
                       "primary_population": "all instances" if not prim else "evidence-bearing subset (evidence_used > 0)"}
    print("\nDECISION:", out["decision"]["primary_population"], "| changed:", prim or "none")
    print(f"|E| = 0 as generated: {count['pooled']['evidence_given_0']}; after pruning: {count['pooled']['evidence_used_0']} "
          f"({count['pooled']['share_evidence_used_0']:.1%})")
    _save("noevidence_benchmarks.json", out)


def noevidence_clinical():
    """3.4: W6 instances that lose all their evidence to pruning, and whether they carry the advantage."""
    import bands as B
    import round3_w6 as W
    X = W.frame()
    assert "evidence_used" in X, "round3_w6.frame must carry evidence_used (scripts/round3_w6.py, this commit)"
    k = json.loads((B.RES / "frontier.json").read_text())["kappa_tuned_iae"]
    X["lr"] = B._log_ratio(X, B.T_DEC, k, "v1")
    out = {"kappa": k, "metric": "decision t = 1%, V1 gates, pruned, both engines", "populations": {}}
    for pname, P in (("W6 primary (window p in [0.001, 0.04])", X[(X.p >= B.WIN[0]) & (X.p <= B.WIN[1])]),
                     ("all test instances", X)):
        P = P.reset_index(drop=True)
        zero = P.evidence_used == 0
        cheap = P.lr < 0
        rec = {"instances": int(len(P)), "evidence_used_0": int(zero.sum()), "share_evidence_used_0": float(zero.mean()),
               "share_lost_most_evidence": float((1 - P.evidence_used / P.evidence_given.clip(lower=1) > 0.5).mean()),
               "share_of_quantum_cheaper_instances_with_no_evidence": float(zero[cheap].mean()) if cheap.any() else None,
               "no_evidence": {"median_ratio": float(10 ** P.lr[zero].median()) if zero.any() else None,
                               "share_quantum_cheaper": float(cheap[zero].mean()) if zero.any() else None},
               "evidence_bearing": {"median_ratio": float(10 ** P.lr[~zero].median()), "share_quantum_cheaper": float(cheap[~zero].mean())},
               "by_band": {b: {"share_evidence_used_0": float(zero[P.band == b].mean()),
                               "evidence_bearing_median_ratio": float(10 ** P.lr[(P.band == b) & ~zero].median()),
                               "evidence_bearing_share_cheaper": float(cheap[(P.band == b) & ~zero].mean())} for b in B.PRIMARY}}
        if zero.any() and (~zero).sum() > 100:
            rec["evidence_bearing_endpoint"] = B._endpoint(P[~zero].reset_index(drop=True), "lr")
        out["populations"][pname] = rec
        print(f"[{pname}] n={rec['instances']:,}  no evidence left {rec['share_evidence_used_0']:.1%}  "
              f"of quantum-cheaper instances, no evidence {rec['share_of_quantum_cheaper_instances_with_no_evidence']:.1%}")
        print(f"   no evidence: ratio {rec['no_evidence']['median_ratio']}, cheaper {rec['no_evidence']['share_quantum_cheaper']}")
        print(f"   evidence-bearing: ratio {rec['evidence_bearing']['median_ratio']:.3f}, cheaper {rec['evidence_bearing']['share_quantum_cheaper']:.3f}")
    _save("noevidence_clinical.json", out)


# ===================================================================================== item 2: in-context cost law
def _prep_split(A):
    """CX of A's preparation gates whose target is an oracle literal (or the flag), and of the rest."""
    from round3_q import _literals_of
    ctrl, _ = _literals_of(A)
    flag = A.num_qubits - 1
    lit = free = 0
    for ins in A.data[:-1]:
        qs = [A.find_bit(q).index for q in ins.qubits]
        c = 2 ** (len(qs) - 1) if len(qs) > 1 else 0  # a UCRY with c controls compiles to 2^c CX
        if qs[0] in ctrl or qs[0] == flag:
            lit += c
        else:
            free += c
    return lit, free


def _fit_report(pred, y) -> dict:
    e = (np.asarray(pred, float) - y) / y
    return {"exact": int((np.round(pred) == y).sum()), "n": int(len(y)), "max_abs_rel_error": float(np.abs(e).max()),
            "median_abs_rel_error": float(np.median(np.abs(e))), "share_law_at_or_above_measured": float((np.round(pred) >= y).mean())}


def law_context():
    """2.1: the standalone law against the in-context counts on the same 224 grid points, and refits reported in the
    same form (exact on N of 224, max error). Decision rule fixed here: an in-context law replaces the standalone law
    in the population estimates (W4, W6) only if it is exact on all 224 points or within 1% everywhere, the bar the
    standalone law was held to. Otherwise the standalone law stays as a stated upper bound and in-context costs are
    quoted as measured."""
    from round3_q import _grid_cases, baseline_A
    g = json.loads((QP / "cost_grid_ms.json").read_text())
    c = json.loads((QP / "cost_grid_in_context.json").read_text())
    lit, free = [], []
    for i, (bn, ev, t, s, meta) in enumerate(_grid_cases()):
        A, _ = baseline_A(bn, range(bn.n), ev, t, s)
        a, b = _prep_split(A)
        lit.append(a)
        free.append(b)
    col = lambda rows, k: np.array([r[k] for r in rows], float)
    n, L, P, A0, A1 = col(g, "qubits"), col(g, "literals"), col(g, "prep_cx"), col(g, "A_v0"), col(g, "A_v1")
    Pf = np.array(free, float)
    out = {"rule": law_context.__doc__.split("Decision rule fixed here: ")[1].replace("\n    ", " "), "variants": {}}
    for v, ystand, yctx in (("v1", col(g, "Q_v1"), col(c, "Q_v1_in_context")), ("v0", col(g, "Q_v0"), col(c, "Q_v0_in_context"))):
        rec = {"in_context_over_standalone": {"median": float(np.median(yctx / ystand)), "min": float((yctx / ystand).min()),
                                              "max": float((yctx / ystand).max()), "share_lower": float((yctx < ystand).mean())}}
        law = 2 * A1 + 6 * (n - 1) if v == "v1" else ystand  # V0's standalone law is the two-valued measured form
        rec["standalone_law_vs_standalone"] = _fit_report(law, ystand)
        rec["standalone_law_vs_in_context"] = _fit_report(law, yctx)
        rec["cancellation_form_vs_in_context"] = _fit_report(law - 2 * Pf, yctx)  # A S_chi A-dagger: prep off the oracle cancels
        X = np.column_stack([P, L - 1, n - 1, Pf, np.ones_like(n)])
        beta = np.linalg.lstsq(X, yctx, rcond=None)[0]
        rec["linear_refit"] = {"terms": ["P", "L-1", "n-1", "P_off_oracle", "1"], "coef": beta.tolist(), **_fit_report(X @ beta, yctx)}
        rec["meets_rule"] = {k: bool(rec[k]["exact"] == len(yctx) or rec[k]["max_abs_rel_error"] <= 0.01)
                             for k in ("standalone_law_vs_in_context", "cancellation_form_vs_in_context", "linear_refit")}
        out["variants"][v] = rec
        print(f"{v}: in-context/standalone median {rec['in_context_over_standalone']['median']:.3f} "
              f"(range {rec['in_context_over_standalone']['min']:.2f}-{rec['in_context_over_standalone']['max']:.2f}, lower in {rec['in_context_over_standalone']['share_lower']:.0%})")
        for k in ("standalone_law_vs_standalone", "standalone_law_vs_in_context", "cancellation_form_vs_in_context", "linear_refit"):
            print(f"   {k:34s} exact {rec[k]['exact']:3d}/{rec[k]['n']}  max err {rec[k]['max_abs_rel_error']:.1%}  "
                  f"median {rec[k]['median_abs_rel_error']:.1%}  law >= measured {rec[k]['share_law_at_or_above_measured']:.0%}")
    ok = any(out["variants"]["v1"]["meets_rule"].values())
    up = out["variants"]["v1"]["standalone_law_vs_in_context"]["share_law_at_or_above_measured"]
    out["decision"] = ("an in-context law meets the bar: use it for W4/W6" if ok else
                       "no in-context closed form meets the bar: the standalone V1 law stays in W4/W6, as an upper bound "
                       f"(at or above the in-context V1 count at {up:.0%} of grid points), and in-context costs are quoted as measured")
    print("DECISION:", out["decision"])
    _save("law_in_context.json", out)


def _capped(count, cap: int = 50):
    """qiskit's optimisation loop iterates to a fixed point; on some long in-context circuits it oscillates and stops
    with 'Maximum iteration reached'. Then the same compile is re-run with the loop stopped after `cap` rounds instead
    of raising, and the circuit is recorded in NONCONVERGED.
    ponytail: patches DoWhileController for that one call; drop when qiskit exposes a loop cap."""
    from qiskit.passmanager import flow_controllers as fc

    def f(c):
        try:
            return count(c)
        except Exception as e:
            if "Maximum iteration reached" not in str(e):
                raise
        orig = fc.DoWhileController.iter_tasks

        def capped(self, state):
            for _ in range(cap):
                for task in self.tasks:
                    state = yield task
                if not self.do_while(state.property_set):
                    return
                state.workflow_status.completed_passes.difference_update(self.tasks)
        fc.DoWhileController.iter_tasks = capped
        try:
            NONCONVERGED.append(c.num_qubits)
            return count(c)
        finally:
            fc.DoWhileController.iter_tasks = orig
    return f


NONCONVERGED: list = []


def _in_context(A, clean, compile_cx):
    """CX of one Grover step as amplitude estimation runs it: after A and one earlier step (the second step's increment)."""
    from qiskit import QuantumCircuit
    from oracle_v2 import grover
    W = A.num_qubits + clean
    a = QuantumCircuit(W)
    a.compose(A, range(A.num_qubits), inplace=True)
    Q = grover(A, clean)
    aq = a.copy()
    aq.compose(Q, inplace=True)
    aqq = aq.copy()
    aqq.compose(Q, inplace=True)
    compile_cx = _capped(compile_cx)
    n0 = len(NONCONVERGED)
    ca, c1, c2 = compile_cx(a), compile_cx(aq), compile_cx(aqq)
    return {"A": ca, "Q_first": c1 - ca, "Q_steady": c2 - c1, "Q_standalone": compile_cx(Q),
            "optimisation_loop_capped": len(NONCONVERGED) > n0}


def variants_context():
    """2.2: V0-V3 per Grover step, in context, on the W1.3 instances (msbn.variants: 5 networks x 3 instances, full and
    pruned). Unit as before: CX per step / (2 x CX of the V0 problem circuit)."""
    from pgmpy.utils import get_example_model
    from msbn import from_pgmpy, instance, prune, baseline_A, clamped_A
    from oracle_v2 import cx
    rng = np.random.default_rng(SEED)  # same draws as msbn.variants
    outp = OUT / "variants_in_context.json"
    rows = json.loads(outp.read_text()).get("rows", []) if outp.exists() else []  # resume: finished rows are kept
    done = {(r["network"], r["instance"], r["scope"]) for r in rows}
    for name in ("insurance", "alarm", "hepar2", "child", "water"):
        bn = from_pgmpy(get_example_model(name))
        for i in range(3):
            ev, t, s = instance(bn, rng)
            for scope in ("full", "pruned"):
                if (name, i, scope) in done:
                    continue
                e2, nodes = (ev, list(range(bn.n))) if scope == "full" else prune(bn, t, ev)
                A0, _ = baseline_A(bn, nodes, e2, t, s)
                A2, _ = clamped_A(bn, nodes, e2, t, s)
                r = {"network": name, "instance": i, "scope": scope, "A_v0": cx(A0)}
                for v, A, clean in (("v0", A0, 0), ("v1", A0, 1), ("v2", A2, 0), ("v3", A2, 1)):
                    q = _in_context(A, clean, cx)
                    r[f"Q_{v}_standalone"], r[f"Q_{v}_in_context"] = q["Q_standalone"], q["Q_steady"]
                    r[f"loop_capped_{v}"] = q["optimisation_loop_capped"]
                    r[f"ratio_{v}_standalone"] = q["Q_standalone"] / (2 * r["A_v0"])
                    r[f"ratio_{v}_in_context"] = q["Q_steady"] / (2 * r["A_v0"])
                rows.append(r)
                print(name, i, scope, {v: (round(r[f"ratio_{v}_standalone"], 2), round(r[f"ratio_{v}_in_context"], 2)) for v in ("v0", "v1", "v2", "v3")}, flush=True)
                outp.write_text(json.dumps({"rows": rows}, indent=1))
    summ = {sc: {f"{v}_{u}": [float(f(np.array([r[f"ratio_{v}_{u}"] for r in rows if r["scope"] == sc]))) for f in (np.min, np.median, np.max)]
                 for v in ("v0", "v1", "v2", "v3") for u in ("standalone", "in_context")} for sc in ("full", "pruned")}
    anchors = json.loads((QP / "cost_in_context.json").read_text())
    out = {"unit": "CX per Grover step / (2 x CX of the V0 problem circuit); in context = second step after A",
           "loop_capped": {"circuits_with_capped_optimisation_loop": sum(r.get(f"loop_capped_{v}", False) for r in rows for v in ("v0", "v1", "v2", "v3")),
                           "note": "qiskit's level-1 optimisation loop did not reach a fixed point; stopped after 50 rounds (see _capped)"},
           "ratio_min_median_max": summ, "v21_anchors": anchors, "rows": rows}
    outp.write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))
    print(f"-> {outp}")


def compile_context(tool: str):
    """2.3: the 35-combination sweep (w8.compile_, w8.tket) redone with in-context Grover steps. One run per qiskit
    install (levels 1-3), one for pytket. Resumable: finished circuits are kept."""
    import pickle
    from msbn import baseline_A
    if not (QP / "w8_cases.pkl").exists():
        raise SystemExit("run `python scripts/w8.py circuits` first (needs pgmpy)")
    with open(QP / "w8_cases.pkl", "rb") as f:
        cases = pickle.load(f)
    if tool == "qiskit":
        import qiskit
        from qiskit.transpiler import generate_preset_pass_manager
        outp = OUT / f"compile_ctx_qiskit{qiskit.__version__}.json"
        settings = {f"opt{l}": generate_preset_pass_manager(optimization_level=l, basis_gates=["rz", "sx", "x", "cx"], seed_transpiler=SEED)
                    for l in (1, 2, 3)}
        counters = {k: (lambda pm: (lambda c: int(dict(pm.run(c).count_ops()).get("cx", 0))))(pm) for k, pm in settings.items()}
    elif tool == "tket":
        from pytket import OpType
        from pytket.extensions.qiskit import qiskit_to_tk
        from pytket.passes import AutoRebase, FullPeepholeOptimise
        from qiskit import transpile
        outp = OUT / "compile_ctx_tket.json"

        def tk_cx(c):
            tk = qiskit_to_tk(transpile(c, basis_gates=["u", "cx"], optimization_level=0))
            FullPeepholeOptimise().apply(tk)
            AutoRebase({OpType.CX, OpType.Rz, OpType.Rx}).apply(tk)
            return int(tk.n_gates_of_type(OpType.CX))
        counters = {"peephole": tk_cx}
    else:
        raise SystemExit("tool: qiskit | tket")
    res = json.loads(outp.read_text()) if outp.exists() else {}
    for name, bn, ev, t, s in cases:
        A, _ = baseline_A(bn, range(bn.n), ev, t, s)
        for setting, count in counters.items():
            for v, clean in (("v0", 0), ("v1", 1)):
                key = f"{name}|{v}|{setting}"
                if key in res:
                    continue
                t0 = time.perf_counter()
                res[key] = _in_context(A, clean, count)
                outp.write_text(json.dumps(res, indent=1))
                print(outp.stem, key, res[key], f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"-> {outp}")


def compare_context():
    """Ratios per compiler setting and case, in context and standalone; V1 < V0 everywhere?"""
    summ = {}
    for f in sorted(OUT.glob("compile_ctx_*.json")):
        tool = f.stem.replace("compile_ctx_", "")
        d = json.loads(f.read_text())
        for key, r in d.items():
            case, v, setting = key.split("|")
            summ.setdefault(f"{tool}|{setting}", {}).setdefault(case, {})[v] = r
    ratios, combos = {}, 0
    for setting, cases in summ.items():
        for case, c in cases.items():
            if "v0" in c and "v1" in c:
                A = c["v0"]["A"]  # unit: 2 x CX of the V0 problem circuit under the same compiler
                ratios.setdefault(setting, {})[case] = {
                    "ratio_v0_in_context": c["v0"]["Q_steady"] / (2 * A), "ratio_v1_in_context": c["v1"]["Q_steady"] / (2 * A),
                    "ratio_v0_standalone": c["v0"]["Q_standalone"] / (2 * A), "ratio_v1_standalone": c["v1"]["Q_standalone"] / (2 * A),
                    "v1_below_v0_in_context": c["v1"]["Q_steady"] < c["v0"]["Q_steady"]}
                combos += 1
    allpass = all(x["v1_below_v0_in_context"] for s in ratios.values() for x in s.values())
    capped = sorted(f"{s}|{c}|{v}" for s, cs in summ.items() for c, x in cs.items() for v, r in x.items() if r.get("optimisation_loop_capped"))
    out = {"combinations": combos, "v1_below_v0_everywhere_in_context": allpass, "loop_capped": capped, "ratios": ratios}
    for s, cs in ratios.items():
        print(s, {c: (round(x["ratio_v0_in_context"], 2), round(x["ratio_v1_in_context"], 2)) for c, x in cs.items()})
    print(f"{combos} combinations; V1 < V0 everywhere in context: {allpass}; optimisation loop capped in {len(capped)}")
    _save("compilers_in_context.json", out)


# ===================================================================================== item 1: consistency sweep
PATTERNS = {"27%": r"27\s?%|0\.27\d?\b", "11.7": r"11\.6\d|11\.7", "24 (kappa)": r"(κ|kappa|IAE)[^\n]{0,40}\b24(\.0)?\b|\b24(\.0)?\b[^\n]{0,40}(κ|kappa|IAE)",
            "7.1": r"\b7\.1\b|\b7\.09", "1.6x": r"1\.6\s?×|1\.6x|~1\.6|≈\s?1\.6", "0.32": r"\b0\.32\b", "0.34": r"\b0\.34\d?\b",
            "kappa": r"κ|\bkappa\b"}
SWEEP_FILES = ["docs/quantum_paper_results.md", "reports/v2.1/quantum_story.md", "docs/hardness_audit.md", "docs/quantum_cost_model.md",
               "docs/round3_results.md"]


def sweep():
    """Every occurrence of the kappa-sensitive numbers in the docs, reports and figure-producing code (captions,
    titles, axis labels, annotations), as a checklist to reconcile against the convention in item 1.1."""
    files = [Path(f) for f in SWEEP_FILES]
    files += sorted(set(Path("docs").glob("*.md")) | set(Path("reports").rglob("*.md")) | set(Path("reports").rglob("*.tex")))
    figcode = [p for p in Path("scripts").glob("*.py") if p.name != "corrections.py" and re.search(r"set_title|suptitle|xlabel|ylabel|annotate|caption|ax\.text", p.read_text(encoding="utf-8", errors="ignore"))]
    lines, missing, n = [], [], 0
    seen = set()
    for f in files + figcode:
        if f in seen:
            continue
        seen.add(f)
        if not f.exists():
            missing.append(str(f))
            continue
        hits = []
        for i, ln in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if f.suffix == ".py" and not re.search(r"set_title|suptitle|xlabel|ylabel|annotate|caption|label=|\.text\(", ln):
                continue
            tags = [k for k, rx in PATTERNS.items() if re.search(rx, ln)]
            if tags:
                hits.append(f"- [ ] L{i} `{', '.join(tags)}`: {ln.strip()[:220]}")
        if hits:
            lines += [f"\n## {f}  ({len(hits)})", *hits]
            n += len(hits)
    head = ["# Consistency sweep (item 1.2)", "",
            "Convention (item 1.1): W4 = benchmark-calibrated IAE constant (24.0); W6 = clinical IAE constant (11.7);",
            "per-instance map secondary (IAE only; MLAE map failed its test); a constant kappa cannot change a rank",
            "correlation. Tick each line once it agrees with the convention or is correct as a historical statement.", "",
            f"{n} occurrences. Missing files: {', '.join(missing) or 'none'}."]
    (OUT / "sweep.md").write_text("\n".join(head + lines) + "\n", encoding="utf-8")
    print(f"{n} occurrences in {len(seen) - len(missing)} files; missing: {missing or 'none'}\n-> {OUT / 'sweep.md'}")


# ===================================================================================== checkpoint
def headline():
    """One table: every headline number, its source file, its kappa and cost convention. A row that cannot be filled
    without ambiguity is printed as MISSING or AMBIGUOUS; the corrections are finished when there are none."""
    def j(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else None
    rows = []

    def add(claim, value, src, kappa, cost):
        rows.append((claim, "MISSING" if value is None else value, src, kappa, cost))
    w4 = j(QP / "w4_kappa.json")
    ne = j(OUT / "noevidence_benchmarks.json")
    pop = ne["decision"]["primary_population"] if ne else None
    k4 = "IAE 24.0 (benchmark calibration)"
    if ne:
        key = "evidence_bearing_only" if pop.startswith("evidence") else "all_instances"
        s = ne["kappa_conventions"]["benchmark IAE constant (primary)"][key]
        sm = ne["kappa_conventions"]["IAE per instance, Q3 map (secondary)"][key]
        add(f"W4 share R < 1 ({pop})", f"{s['share_R_below_1']:.1%}", "corrections/noevidence_benchmarks.json", k4, "standalone V1 law")
        add("W4 share R < 1, per-instance map (secondary)", f"{sm['share_R_below_1']:.1%}", "corrections/noevidence_benchmarks.json", "IAE map", "standalone V1 law")
        add("W4 pooled Spearman, cost", f"{s['pooled_spearman_cost']['rho']:+.2f} {np.round(s['pooled_spearman_cost']['ci95_network_bootstrap'], 2).tolist()}",
            "corrections/noevidence_benchmarks.json", "invariant to constant kappa", "standalone V1 law")
        add("W4 within-network rank r, cost", f"{s['within_network_rank_r_cost']['r']:+.2f} {np.round(s['within_network_rank_r_cost']['ci95_network_bootstrap'], 2).tolist()}",
            "corrections/noevidence_benchmarks.json", "invariant to constant kappa", "standalone V1 law")
        add("W4 partial Spearman log R ~ log p, given cost", f"{s['partial_spearman_p_given_cost']['r']:.2f}", "corrections/noevidence_benchmarks.json",
            "invariant to constant kappa", "standalone V1 law")
        add("Benchmark |E| = 0 after pruning", f"{ne['count']['pooled']['evidence_used_0']} ({ne['count']['pooled']['share_evidence_used_0']:.1%})",
            "corrections/noevidence_benchmarks.json", "-", "-")
    else:
        add("W4 (run `noevidence` first)", None, "corrections/noevidence_benchmarks.json", k4, "standalone V1 law")
    w6 = j(QP / "w6_kappa.json")
    if w6:
        v = w6["variants"]
        p, m = v["published: IAE, constant 11.69"], v["IAE, per instance (Q3 map)"]
        add("W6 primary ratio (abstract)", f"{p['ratio']:.3f}", "quantum_paper/w6_kappa.json", "IAE 11.7 (clinical calibration)", "standalone V1 law")
        add("W6 primary, quantum cheaper", f"{p['share_qlw_cheaper']:.1%}", "quantum_paper/w6_kappa.json", "IAE 11.7 (clinical calibration)", "standalone V1 law")
        add("W6 per-instance map (secondary)", f"{m['ratio']:.3f} ({m['share_qlw_cheaper']:.1%})", "quantum_paper/w6_kappa.json", "IAE map", "standalone V1 law")
    else:
        add("W6 primary", None, "quantum_paper/w6_kappa.json (local VM)", "IAE 11.7", "standalone V1 law")
    nc = j(OUT / "noevidence_clinical.json")
    if nc:
        r = nc["populations"]["W6 primary (window p in [0.001, 0.04])"]
        add("W6 primary: share with no evidence left", f"{r['share_evidence_used_0']:.1%}", "corrections/noevidence_clinical.json", "-", "-")
        add("W6 primary, evidence-bearing ratio", f"{r['evidence_bearing']['median_ratio']:.3f} ({r['evidence_bearing']['share_quantum_cheaper']:.1%})",
            "corrections/noevidence_clinical.json", "IAE 11.7", "standalone V1 law")
    else:
        add("W6 no-evidence check", None, "corrections/noevidence_clinical.json (local VM)", "IAE 11.7", "standalone V1 law")
    law = j(OUT / "law_in_context.json")
    add("Cost law decision (in-context)", law["decision"] if law else None, "corrections/law_in_context.json", "-", "in context")
    an = j(QP / "cost_in_context.json")
    if an:
        add("v2.1 V1 anchors, in context", " / ".join(f"{an[f'v2.1_findings{f}|v1']['ratio_in_context']:.2f}" for f in (27, 38, 39)),
            "quantum_paper/cost_in_context.json", "-", "in context (second step after A)")
        add("v2.1 V1 anchors, standalone", " / ".join(f"{an[f'v2.1_findings{f}|v1']['ratio_standalone']:.2f}" for f in (27, 38, 39)),
            "quantum_paper/cost_in_context.json", "-", "standalone (superseded)")
    vc = j(OUT / "variants_in_context.json")
    add("V0-V3 ratios, in context (full network)", json.dumps({k: np.round(x, 2).tolist() for k, x in vc["ratio_min_median_max"]["full"].items() if k.endswith("in_context")}) if vc and "ratio_min_median_max" in vc else None,
        "corrections/variants_in_context.json", "-", "in context")
    if vc and "loop_capped" in vc:
        add("V0-V3 in context: circuits with capped qiskit loop", vc["loop_capped"]["circuits_with_capped_optimisation_loop"],
            "corrections/variants_in_context.json", "-", "in context")
    cc = j(OUT / "compilers_in_context.json")
    add("V1 < V0 under every compiler, in context", f"{cc['v1_below_v0_everywhere_in_context']} ({cc['combinations']} combinations; loop capped in {len(cc.get('loop_capped', []))})" if cc else None,
        "corrections/compilers_in_context.json", "-", "in context")
    w5 = j(QP / "w5_factorial.json")
    if w5:
        t3 = w5["tests"]["iii_partial_spearman_logR_logp_given_cost"]
        add("Q1 partial Spearman log R ~ log p, given cost", f"{t3['value']:.3f} {np.round(t3['ci95_network_bootstrap'], 3).tolist()}",
            "quantum_paper/w5_factorial.json", "rank statistic, invariant to constant kappa", "standalone V1 law")
        add("Q1 shares of R < 1 by base rate", "0-0.2% / 98-99.9% / 100% / 100%", "quantum_paper/w5_factorial.json",
            "AMBIGUOUS: computed at IAE 11.7; synthetic instances have no calibration of their own (choose: 11.7, 24.0 or the map)",
            "standalone V1 law")
    else:
        add("Q1 factorial", None, "quantum_paper/w5_factorial.json", "-", "-")
    pk = j(QP / "w6_pauker_kassirer.json")
    if pk:
        t = pk["thresholds"]["H/B=0.01"]["error_0.1max(p,t)"]
        add("Pauker-Kassirer H/B = 1/100", f"{t['median_ratio']:.2f} ({t['share_qlw_cheaper']:.0%})", "quantum_paper/w6_pauker_kassirer.json", "IAE 11.7", "standalone V1 law")
    else:
        add("Pauker-Kassirer sweep", None, "quantum_paper/w6_pauker_kassirer.json (local VM)", "IAE 11.7", "standalone V1 law")
    bad = [r for r in rows if r[1] == "MISSING"]
    amb = [r for r in rows if "AMBIGUOUS" in r[3] or "AMBIGUOUS" in r[4]]
    md = ["# Headline numbers: source and conventions (checkpoint)", "",
          "| Claim | Value | Source (artifacts/) | κ convention | Cost convention |", "|---|---|---|---|---|"]
    md += [f"| {a} | {b} | `{c}` | {d} | {e} |" for a, b, c, d, e in rows]
    md += ["", f"Rows missing: {len(bad)}. Rows ambiguous: {len(amb)}." + ("" if bad or amb else " The table is complete.")]
    (OUT / "headline_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"-> {OUT / 'headline_table.md'}")


if __name__ == "__main__":
    cmds = {"noevidence": noevidence, "noevidence-clinical": noevidence_clinical, "law-context": law_context,
            "variants-context": variants_context, "compile-context": compile_context, "compare-context": compare_context,
            "sweep": sweep, "headline": headline}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit(__doc__)
    cmds[sys.argv[1]](*sys.argv[2:])
