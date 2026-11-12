"""Classical joint-diagnosis test: is P(A and B | findings) predicted better by the network's true joint than by
products of marginals or by classifiers trained directly on the pair label? Pre-specified in tasks/plan_pairs.md
(pass/fail rule copied below). Structured v2.1 data only; frozen network; aggregates only.

  python scripts/pairs.py            # full run: sanity checks, arms, bootstrap, verdict, outputs
  python scripts/pairs.py --quick    # 50 bootstrap resamples, no XGBoost tuning (pipeline check only)

Pass/fail (AUPRC, named pairs HF+AFib, UTI+AKI, AKI+anemia; "beats" = paired difference > 0 with 95% CI excluding 0):
  Fail first: arm 4a or 4b beats arm 1 in >= 2 of 3 named pairs.
  Pass: arm 1 beats arm 3 and 4a and 4b in >= 2 of 3 named pairs.
  Partial: neither, and arm 1 beats arm 2 in >= 2 of 3 named pairs.  Otherwise inconclusive.
"""
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qddx import kb
from qddx.bn import Evidence, marginal
from qddx.cli import load_config
from qddx.evaluate import bootstrap, fit_lr, lr_features, lr_predict
from qddx.vocab import inner_split

NAME = "v2.1"
KBDIR, RES, REP = Path(f"artifacts/{NAME}/kb"), Path(f"artifacts/{NAME}/results"), Path(f"reports/{NAME}")
NAMED = [("heart_failure", "afib"), ("uti", "aki"), ("aki", "anemia")]
LIFT_MIN, ALERT, SEED = 1.2, 0.10, 2026
ARMS = ["1 network joint", "2 network product", "3 logistic product", "4a logistic pair", "4b XGBoost pair"]


# ------------------------------------------------------------------------------------- metrics (weighted, tie-aware)
class Ranked:
    """One score vector, sorted once; weighted AUPRC / AUROC / recall-at-rate for any bootstrap weights in O(n)."""

    def __init__(self, s: np.ndarray, y: np.ndarray):
        self.o = np.argsort(-s, kind="mergesort")
        ss = s[self.o]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]  # tie groups in descending order
        self.y = y[self.o].astype(float)

    def all(self, w, rate: float) -> tuple[float, float, float]:
        """(AUPRC, AUROC, recall in the top `rate` of the weight) for bootstrap weights w."""
        ws = w[self.o]
        tp = np.add.reduceat(ws * self.y, self.starts)
        fp = np.add.reduceat(ws * (1 - self.y), self.starts)
        P, N = tp.sum(), fp.sum()
        if P <= 0 or N <= 0:
            return np.nan, np.nan, np.nan
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        prec = np.divide(ctp, ctp + cfp, out=np.zeros_like(ctp), where=(ctp + cfp) > 0)
        auprc = float((tp / P * prec).sum())  # sum over thresholds of (R_n - R_{n-1}) P_n, as sklearn
        auroc = float(1 - (tp * (cfp - fp + 0.5 * fp)).sum() / (P * N))  # cfp - fp = negatives scored strictly higher
        k = np.searchsorted(np.cumsum(ws), rate * ws.sum())  # rows in score order until `rate` of the weight
        recall = float((ws[: k + 1] * self.y[: k + 1]).sum() / P)
        return auprc, auroc, recall


def brier(p, y, w):
    return float(np.average((p - y) ** 2, weights=w)) if np.isfinite(p).all() else np.nan


# ------------------------------------------------------------------------------------- d-separation
def d_separated(parents: list[tuple], a: int, b: int, given: set[int]) -> bool:
    """a and b d-separated given `given`: moralised ancestral graph of {a, b} U given, with `given` removed."""
    anc, stack = set(), [a, b, *given]
    while stack:
        j = stack.pop()
        if j not in anc:
            anc.add(j)
            stack.extend(parents[j])
    adj = {j: set() for j in anc}
    for j in anc:
        pa = list(parents[j])
        for p in pa:
            adj[j].add(p)
            adj[p].add(j)
        for p, q in combinations(pa, 2):
            adj[p].add(q)
            adj[q].add(p)
    seen, stack = {a}, [a]
    while stack:
        j = stack.pop()
        for k in adj[j] - given:
            if k == b:
                return False
            if k not in seen:
                seen.add(k)
                stack.append(k)
    return True


# ------------------------------------------------------------------------------------- network arms
def network_tables(bn, df, ev, dx):
    """Exact P(dx pattern | findings) for each row (64 = 2^6 diagnosis settings), once per evidence pattern."""
    st = bn.struct
    keep = [st.index(d) for d in dx]
    E = df[ev].to_numpy(np.int8)
    pats, inv = np.unique(E, axis=0, return_inverse=True)
    tabs = np.zeros((len(pats), 2 ** len(dx)))
    for i, p in enumerate(pats):
        t = marginal(bn, Evidence({n: int(v) for n, v in zip(ev, p) if v >= 0}, {}), keep).reshape(-1)
        tabs[i] = t / t.sum()
    return tabs, pats, inv.reshape(-1)


def pair_from_table(tab, i, j, n=6):
    """P(d_i = 1, d_j = 1) and the marginals from a [rows, 2^n] table (d_0 = most significant bit)."""
    s = np.arange(2 ** n)
    bi, bj = (s >> (n - 1 - i)) & 1, (s >> (n - 1 - j)) & 1
    return tab[:, (bi & bj) == 1].sum(1), tab[:, bi == 1].sum(1), tab[:, bj == 1].sum(1)


# ------------------------------------------------------------------------------------- main
def main(quick: bool):
    t0 = time.perf_counter()
    cfg = load_config("configs/qddx.yaml", [], vocab=f"configs/vocab_{NAME}.json")
    dx, ev = kb.props(cfg, "diagnosis"), kb.evidence(cfg)
    bn, _ = kb.load(cfg, KBDIR)
    df = pd.read_parquet(KBDIR / "cohort.parquet")
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    inner, val, _ = inner_split(cfg, df)
    train, test = df[tr].reset_index(drop=True), df[~tr].reset_index(drop=True)
    pairs = list(combinations(range(len(dx)), 2))
    pname = [f"{dx[i]}+{dx[j]}" for i, j in pairs]
    Yd = test[dx].to_numpy(int)
    Yp = np.stack([Yd[:, i] & Yd[:, j] for i, j in pairs], 1)
    out = {"name": NAME, "n_train": int(len(train)), "n_test": int(len(test)), "n_test_patients": int(test.subject_id.nunique()),
           "pairs": pname, "named": [f"{a}+{b}" for a, b in NAMED], "arms": ARMS, "quick": quick}

    # 1. pair counts and lift (decided before any model score)
    cnt = []
    for k, (i, j) in enumerate(pairs):
        both, either = int(Yp[:, k].sum()), int((Yd[:, i] | Yd[:, j]).sum())
        lift = Yp[:, k].mean() / (Yd[:, i].mean() * Yd[:, j].mean())
        cnt.append({"pair": pname[k], "both": both, "either": either, "lift": float(lift), "informative": bool(lift >= LIFT_MIN)})
    out["counts"] = cnt
    print("pair counts (test):")
    for c in cnt:
        print(f"  {c['pair']:<26} both {c['both']:>6,}  either {c['either']:>7,}  lift {c['lift']:.2f}{'' if c['informative'] else '  (uninformative)'}")

    # 2. arms on the test set (and on validation rows, for the secondary calibrated Brier)
    tabs, pats, inv = network_tables(bn, test, ev, dx)
    print(f"network: {len(pats):,} distinct test evidence patterns ({time.perf_counter() - t0:.0f}s)")
    M_tr = train[ev].to_numpy(float, copy=True)
    M_te = test[ev].to_numpy(float, copy=True)
    M_tr[M_tr < 0], M_te[M_te < 0] = np.nan, np.nan
    kinds = [cfg["propositions"][k]["kind"] for k in ev]
    lr = fit_lr(cfg, KBDIR)  # the existing lr.json (all training admissions)
    L_te = lr_predict(lr, cfg, M_te)
    vmask = val[tr]  # validation admissions within the training rows
    vtab, _, vinv = network_tables(bn, train[vmask], ev, dx)
    L_va = lr_predict(lr, cfg, M_tr[vmask])

    # sanity 2: single-diagnosis AUROC reproduces eval-kb
    ek = json.loads((RES / "eval-kb.json").read_text())["models"]
    net_m = np.stack([pair_from_table(tabs[inv], i, i)[1] for i in range(len(dx))], 1)
    one = np.ones(len(test))
    mac = lambda P: float(np.nanmean([Ranked(P[:, i], Yd[:, i]).all(one, ALERT)[1] for i in range(len(dx))]))
    s2 = {"network": (mac(net_m), ek["learned network"]["macro_auroc"]["value"]),
          "logistic": (mac(L_te), ek["logistic regression"]["macro_auroc"]["value"])}
    out["sanity_single_auroc"] = {k: {"here": a, "eval_kb": b} for k, (a, b) in s2.items()}
    for k, (a, b) in s2.items():
        print(f"sanity: {k} single-diagnosis macro AUROC {a:.4f} vs eval-kb {b:.4f}")
        assert abs(a - b) <= 0.002, f"{k} AUROC does not reproduce eval-kb: pipeline is wrong"

    from sklearn.linear_model import LogisticRegression
    X_tr, X_te = lr_features(M_tr, kinds), lr_features(M_te, kinds)
    S, S_va = {a: np.zeros((len(test), len(pairs))) for a in ARMS}, {a: np.zeros((vmask.sum(), len(pairs))) for a in ARMS}
    dsep_checked, dsep_maxdiff, dsep_cache = 0, 0.0, {}
    Ytr_d = train[dx].to_numpy(int)
    xgb_cfg = {}
    for k, (i, j) in enumerate(pairs):
        pj, pi_, pj_ = pair_from_table(tabs, i, j)
        S[ARMS[0]][:, k], S[ARMS[1]][:, k] = pj[inv], (pi_ * pj_)[inv]
        vj, vi_, vj_ = pair_from_table(vtab, i, j)
        S_va[ARMS[0]][:, k], S_va[ARMS[1]][:, k] = vj[vinv], (vi_ * vj_)[vinv]
        S[ARMS[2]][:, k], S_va[ARMS[2]][:, k] = L_te[:, i] * L_te[:, j], L_va[:, i] * L_va[:, j]
        # sanity 1: independence given the findings => joint == product (per evidence pattern)
        for p in range(len(pats)):
            obs = frozenset(bn.struct.index(n) for n, v in zip(ev, pats[p]) if v >= 0)
            key = (i, j, obs)  # d-separation depends on which nodes are observed, not on their values
            if key not in dsep_cache:
                dsep_cache[key] = d_separated(bn.struct.parents, bn.struct.index(dx[i]), bn.struct.index(dx[j]), set(obs))
            if dsep_cache[key]:
                dsep_checked += 1
                dsep_maxdiff = max(dsep_maxdiff, abs(pj[p] - pi_[p] * pj_[p]))
        y_tr = Ytr_d[:, i] & Ytr_d[:, j]
        assert len(np.unique(y_tr)) == 2, f"{pname[k]}: no positive training admissions"
        m = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
        S[ARMS[3]][:, k], S_va[ARMS[3]][:, k] = m.predict_proba(X_te)[:, 1], m.predict_proba(X_tr[vmask])[:, 1]
        S[ARMS[4]][:, k], S_va[ARMS[4]][:, k], xgb_cfg[pname[k]] = fit_xgb(M_tr, y_tr, inner[tr], val[tr], M_te, quick)
        print(f"arms fitted: {pname[k]} ({time.perf_counter() - t0:.0f}s)", flush=True)
    out["sanity_independence"] = {"pattern_pairs_d_separated": dsep_checked, "max_abs_joint_minus_product": dsep_maxdiff}
    print(f"sanity: {dsep_checked:,} (pattern, pair) cases d-separated given the findings; max |joint - product| {dsep_maxdiff:.1e}")
    assert dsep_maxdiff <= 1e-9, "joint != product under independence: pipeline is wrong"
    out["xgboost"] = xgb_cfg

    # 3. metrics, bootstrap over patients, paired differences
    B = 50 if quick else 1000
    rank = {(a, k): Ranked(S[a][:, k], Yp[:, k]) for a in ARMS for k in range(len(pairs))}
    hits = {a: (np.sum(S[a][:, None, :] > S[a][:, :, None], axis=2) < 3) for a in ARMS}  # pair in the row's top 3

    def metrics(w):
        r = {}
        for a in ARMS:
            for k in range(len(pairs)):
                y = Yp[:, k]
                pos = y == 1
                ap, auc, rec = rank[(a, k)].all(w, ALERT)
                r[(a, k)] = {"auprc": ap, "auroc": auc, "brier": brier(S[a][:, k], y, w), "recall_at_10pct": rec,
                             "top3_pair_hit": float(np.average(hits[a][pos, k], weights=w[pos])) if w[pos].sum() > 0 else np.nan}
        return r

    point = metrics(one)
    boots = bootstrap(test.subject_id.to_numpy(), B, SEED, metrics)
    print(f"bootstrap: {B} patient resamples ({time.perf_counter() - t0:.0f}s)")

    def ci(vals):
        v = np.array(vals, float)
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [None, None]

    mets = ["auprc", "auroc", "brier", "recall_at_10pct", "top3_pair_hit"]
    table = {}
    for k in range(len(pairs)):
        t = {"arms": {}, "diff_vs_arm1": {}}
        for a in ARMS:
            t["arms"][a] = {m: {"value": point[(a, k)][m], "ci": ci([b[(a, k)][m] for b in boots])} for m in mets}
        for a in ARMS[1:]:
            t["diff_vs_arm1"][a] = {m: {"value": point[(ARMS[0], k)][m] - point[(a, k)][m],
                                        "ci": ci([b[(ARMS[0], k)][m] - b[(a, k)][m] for b in boots])} for m in mets}
        # secondary: Platt-calibrated Brier (fitted on validation admissions; in-sample for every arm alike)
        yv = Ytr_d[vmask][:, pairs[k][0]] & Ytr_d[vmask][:, pairs[k][1]]
        t["brier_calibrated"] = {a: brier(platt(S_va[a][:, k], yv, S[a][:, k]), Yp[:, k], one) for a in ARMS}
        table[pname[k]] = t
    out["table"] = table
    # secondary: both diagnoses in the top 3 by single-diagnosis scores (arms with single-diagnosis outputs)
    top3 = {"network (arms 1-2)": np.argsort(-net_m, 1)[:, :3], "logistic (arm 3)": np.argsort(-L_te, 1)[:, :3]}
    out["both_in_top3_single"] = {name: {pname[k]: float(np.mean([(i in r) and (j in r) for r in T[Yp[:, k] == 1]]))
                                         for k, (i, j) in enumerate(pairs)} for name, T in top3.items()}

    # 4. verdict
    def beats(k, a, b_arm, sign=1):
        d = table[pname[k]]["diff_vs_arm1"][b_arm]["auprc"]
        lo, hi = d["ci"]
        return (lo > 0) if sign > 0 else (hi < 0)
    nk = [pname.index(f"{a}+{b}") for a, b in NAMED]
    fail = sum(beats(k, ARMS[0], ARMS[3], -1) or beats(k, ARMS[0], ARMS[4], -1) for k in nk) >= 2
    passed = sum(all(beats(k, ARMS[0], a) for a in ARMS[2:]) for k in nk) >= 2
    partial = sum(beats(k, ARMS[0], ARMS[1]) for k in nk) >= 2
    verdict = "FAIL" if fail else "PASS" if passed else "PARTIAL" if partial else "INCONCLUSIVE"
    out["verdict"] = verdict
    out["arm1_beats_arm2_named"] = int(sum(beats(k, ARMS[0], ARMS[1]) for k in nk))
    out["seconds"] = time.perf_counter() - t0
    RES.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    (RES / "pairs.json").write_text(json.dumps(out, indent=1))
    write_md(out, REP / "pairs.md")
    figure(out, REP / "pairs.png")
    print(f"verdict: {verdict}  (arm 1 beats arm 2 in {out['arm1_beats_arm2_named']}/3 named pairs); wrote {RES / 'pairs.json'}, "
          f"{REP / 'pairs.md'}, {REP / 'pairs.png'} ({out['seconds']:.0f}s)")


def platt(s_fit, y_fit, s_apply):
    from sklearn.linear_model import LogisticRegression
    lg = lambda s: np.log(np.clip(s, 1e-9, 1 - 1e-9) / (1 - np.clip(s, 1e-9, 1 - 1e-9))).reshape(-1, 1)
    if len(np.unique(y_fit)) < 2:  # no positives to calibrate on (only in tiny synthetic data)
        return np.full(len(s_apply), np.nan)
    return LogisticRegression().fit(lg(s_fit), y_fit).predict_proba(lg(s_apply))[:, 1]


def fit_xgb(M_tr, y_tr, inner, val, M_te, quick):
    """Depth chosen and rounds early-stopped on inner-train -> validation, then refit on all training rows."""
    import xgboost as xgb
    base = {"objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "tree_method": "hist", "nthread": 2, "seed": SEED}
    best = None
    di, dv = xgb.DMatrix(M_tr[inner], label=y_tr[inner]), xgb.DMatrix(M_tr[val], label=y_tr[val])
    for depth in ([4] if quick else [3, 5, 7]):
        b = xgb.train({**base, "max_depth": depth}, di, 50 if quick else 2000, evals=[(dv, "val")],
                      early_stopping_rounds=50, verbose_eval=False)
        if best is None or b.best_score < best[0]:
            best = (b.best_score, depth, b.best_iteration + 1)
    final = xgb.train({**base, "max_depth": best[1]}, xgb.DMatrix(M_tr, label=y_tr), best[2])
    return (final.predict(xgb.DMatrix(M_te)), final.predict(xgb.DMatrix(M_tr[val])),
            {"max_depth": best[1], "rounds": best[2], "val_logloss": float(best[0])})


def write_md(o, path):
    f = lambda v, d=3: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"
    L = [f"# Joint-diagnosis test ({o['name']}, structured data)", "",
         f"Train {o['n_train']:,} admissions; test {o['n_test']:,} admissions of {o['n_test_patients']:,} patients. "
         f"Pre-specified in `tasks/plan_pairs.md`.{' QUICK RUN (pipeline check only).' if o['quick'] else ''}", "",
         f"**Verdict: {o['verdict']}** (named pairs: {', '.join(o['named'])}). Arm 1 beats arm 2 (network joint vs its own product) "
         f"in {o['arm1_beats_arm2_named']}/3 named pairs.", "", "## Sanity checks", ""]
    for k, v in o["sanity_single_auroc"].items():
        L.append(f"- {k}: single-diagnosis macro AUROC {v['here']:.4f} (eval-kb {v['eval_kb']:.4f})")
    si = o["sanity_independence"]
    L += [f"- independence: {si['pattern_pairs_d_separated']:,} (pattern, pair) cases d-separated given the findings; "
          f"max |joint - product| = {si['max_abs_joint_minus_product']:.1e}", "", "## Pairs (test, before any model score)", "",
          "| pair | both | either | lift |", "|---|---|---|---|"]
    L += [f"| {c['pair']} | {c['both']:,} | {c['either']:,} | {c['lift']:.2f} |" for c in o["counts"]]
    order = sorted(o["pairs"], key=lambda p: (p not in o["named"], -next(c["lift"] for c in o["counts"] if c["pair"] == p)))
    for p in order:
        t = o["table"][p]
        L += ["", f"## {p}{' (named)' if p in o['named'] else ''}", "",
              "| arm | AUPRC | AUROC | Brier | recall @10% | top-3 pair hit | Brier (calibrated) |", "|---|---|---|---|---|---|---|"]
        for a, m in t["arms"].items():
            L.append(f"| {a} | {f(m['auprc']['value'])} [{f(m['auprc']['ci'][0])}, {f(m['auprc']['ci'][1])}] | {f(m['auroc']['value'])} | "
                     f"{f(m['brier']['value'], 4)} | {f(m['recall_at_10pct']['value'])} | {f(m['top3_pair_hit']['value'])} | "
                     f"{f(t['brier_calibrated'][a], 4)} |")
        L += ["", "| arm 1 minus | AUPRC difference [95% CI] |", "|---|---|"]
        L += [f"| {a} | {f(d['auprc']['value'], 4)} [{f(d['auprc']['ci'][0], 4)}, {f(d['auprc']['ci'][1], 4)}] |" for a, d in t["diff_vs_arm1"].items()]
    path.write_text("\n".join(L) + "\n")


def figure(o, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = sorted(o["pairs"], key=lambda p: next(c["lift"] for c in o["counts"] if c["pair"] == p))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(order))
    for n, a in enumerate(o["arms"]):
        ax.plot(x + (n - 2) * 0.12, [o["table"][p]["arms"][a]["auprc"]["value"] for p in order], "o", ms=5, label=a)  # pairs are categories: dots, no lines
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p.replace('heart_failure', 'HF').replace('_', ' ')}\n(lift {next(c['lift'] for c in o['counts'] if c['pair'] == p):.1f})"
                        for p in order], rotation=60, ha="right", fontsize=7)
    for t in ax.get_xticklabels():
        if any(t.get_text().startswith(n.replace("heart_failure", "HF").replace("_", " ")) for n in o["named"]):
            t.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("AUPRC of P(both)")
    ax.set_title("Joint-diagnosis test: AUPRC by arm, pairs sorted by dependence lift")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main("--quick" in sys.argv)
