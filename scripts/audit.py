"""Statistical audit of the quantum cost paper and its clinical annex.

Plan: tasks/stat_audit_plan.md, frozen before this script ran (tag stat-audit-plan-frozen). Every test the script
runs is written to test_log.csv, including the ones that weaken a claim.

Run from the repo root:
    python scripts/audit.py              # everything (about 1-2 hours, most of it the clinical bootstrap)
    python scripts/audit.py --quick      # small resample counts, to check the run end to end (about 15 min)
    python scripts/audit.py --only quantum   |   --only clinical
Outputs (artifacts/audit/): claim_audit_report.md, test_log.csv, forest_plot.png, network_size_scatter.png,
results.json, requirements_used.txt. Clinical outputs are aggregates only.

statsmodels is optional. Without it the mixed models (S1.3, S2.3, S3.2) are replaced by the fallbacks the plan
names, and the report says which analyses were skipped.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

OUT = Path("artifacts/audit")
QP = Path("artifacts/quantum_paper")
SEED, Z90, REL, T_DEC, WIN = 2026, 1.645, 0.1, 0.01, (0.001, 0.04)
DELTA = 0.2          # S2.1/S2.2 equivalence margin on |Spearman| (the bar already used in the W5 factorial)
NEG_MARGIN = 0.02    # S2.4 equivalence margin on AUROC around 0.5
WIDTH_MARGIN = 0.05  # S3.2 equivalence margin on the width effect: share of R < 1, widest minus narrowest width
AC1_GAP = 0.2        # S4.3: AC1 this much above Fleiss kappa (medians) means the kappa reading is a prevalence artifact
ERR_BAR = 0.15       # S3.3: the kappa-map error bar from round 3
try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

R = {}        # results, written to results.json
LOG = []      # every test
SKIPPED = []  # analyses not run, with the reason
N = {}        # resample counts (set in main)


def say(*a):
    print(*a, flush=True)


def log_test(tid, claim, family, test, effect, lo=np.nan, hi=np.nan, p=np.nan, note="", prespecified=True):
    LOG.append({"id": tid, "claim": claim, "family": family, "prespecified": prespecified, "test": test,
                "effect": effect, "ci_low": lo, "ci_high": hi, "p_raw": p, "note": note})


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok):
        o = ok[np.argsort(p[ok])]
        adj = np.maximum.accumulate((len(o) - np.arange(len(o))) * p[o])
        out[o] = np.minimum(adj, 1.0)
    return out


def test_log_frame() -> pd.DataFrame:
    d = pd.DataFrame(LOG)
    d["p_adj_holm"] = np.nan
    for fam, idx in d.groupby("family").groups.items():
        if fam.startswith("sensitivity"):
            continue  # the plan: sensitivity family reported unadjusted
        d.loc[idx, "p_adj_holm"] = holm(d.loc[idx, "p_raw"].to_numpy())
    return d


# ------------------------------------------------------------------------------------------------ resampling helpers
def cluster_index(groups):
    """Row indices per cluster, for fast cluster bootstraps."""
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    cuts = np.flatnonzero(np.diff(inv[order])) + 1
    return np.split(order, cuts)


def cluster_boot(rows_of, stat, B, seed=SEED):
    rng = np.random.default_rng(seed)
    k = len(rows_of)
    out = np.empty(B)
    for b in range(B):
        out[b] = stat(np.concatenate([rows_of[j] for j in rng.integers(0, k, k)]))
    return out


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, a)) for a in q] if len(x) else [np.nan] * len(q)


def boot_p(boots, null=0.0):
    """Two-sided bootstrap p for H0: effect = null (percentile method; floor 1/B)."""
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if not len(b):
        return np.nan
    return float(min(1.0, max(2 * min((b <= null).mean(), (b >= null).mean()), 1 / len(b))))


def spearman(x, y):
    return float(stats.spearmanr(x, y).statistic)


def tost_corr(est, boots, delta=DELTA):
    """TOST for |rho| < delta on the Fisher-z scale, SE from the cluster bootstrap; 90% CI from percentiles."""
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    z, se = np.arctanh(np.clip(est, -0.999999, 0.999999)), np.std(np.arctanh(np.clip(b, -0.999999, 0.999999)), ddof=1)
    p_lo = stats.norm.sf((z - np.arctanh(-delta)) / se)
    p_hi = stats.norm.cdf((z - np.arctanh(delta)) / se)
    lo90, hi90 = pct(b, [0.05, 0.95])
    return {"estimate": float(est), "ci90": [lo90, hi90], "ci95": pct(b, [0.025, 0.975]), "p_tost": float(max(p_lo, p_hi)),
            "equivalent": bool(-delta < lo90 and hi90 < delta), "margin": delta}


def logistic_irls(X, y, ridge=1e-8, it=100):
    """Logistic regression by IRLS. -> coef, se, loglik, converged, separation flag."""
    X = np.column_stack([np.ones(len(X)), X])
    b = np.zeros(X.shape[1])
    conv = False
    for _ in range(it):
        eta = np.clip(X @ b, -35, 35)
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-12
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-9:
            conv = True
            break
    eta = np.clip(X @ b, -35, 35)
    mu = 1 / (1 + np.exp(-eta))
    ll = float(np.sum(y * np.log(mu + 1e-300) + (1 - y) * np.log(1 - mu + 1e-300)))
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (mu * (1 - mu) + 1e-12)[:, None]) + ridge * np.eye(X.shape[1]))))
    return b, se, ll, conv, bool(np.abs(b).max() > 15 or not conv)


def ols(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X, y, rcond=None)[0]


class AUC:
    """Weighted AUROC for fixed scores, O(n) per weight vector (ties count one half). Matches sklearn's
    roc_auc_score(y, s, sample_weight=w)."""
    def __init__(self, y, s):
        o = np.argsort(s, kind="mergesort")
        ss = np.asarray(s)[o]
        self.o, self.y = o.astype(np.int64), np.asarray(y, float)[o]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]

    def __call__(self, w):
        w = w[self.o]
        wp = w * self.y
        gp, gn = np.add.reduceat(wp, self.starts), np.add.reduceat(w - wp, self.starts)
        P, Nn = gp.sum(), gn.sum()
        if P <= 0 or Nn <= 0:
            return np.nan
        return float((gp * (np.cumsum(gn) - gn + 0.5 * gn)).sum() / (P * Nn))


# ================================================================================================ quantum data
def kappa_map_fn(est="kappa_iae", M=None):
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt
    m = json.loads((QP / "kappa_map.json").read_text())
    M = np.array(m["estimators"][est]["log10_kappa"], float) if M is None else M
    _, (ii, jj) = distance_transform_edt(~np.isfinite(M), return_indices=True)
    f = RegularGridInterpolator((np.log10(m["a_E"]), np.log10(m["p"])), M[ii, jj])
    la, lp = np.log10(m["a_E"]), np.log10(m["p"])
    return lambda aE, p: 10 ** f(np.column_stack([np.clip(np.log10(aE), la[0], la[-1]), np.clip(np.log10(p), lp[0], lp[-1])]))


def bench_frame():
    import benchmarks as BM
    _, ok = BM._frame(BM.NETWORKS)
    ok = ok.reset_index(drop=True)
    ok["V"] = ok.lw_samples / (Z90 / REL) ** 2       # LW relative variance
    ok["D"] = ok.V * ok.p / (1 - ok.p)                # D: that variance over plain Monte Carlo's (1 - p) / p
    ok["kappa_const"] = BM.KAPPA
    ok["kappa_map"] = kappa_map_fn()(ok.a_E.to_numpy(), ok.p.to_numpy())
    return ok, BM


def logR_rel(lp, laE, lD, lg, lk):
    """log10 R as in W4/W5: qLW (kappa pi/eps (1/sqrt a_E + 1/sqrt a_T)) x V1 gate ratio / LW samples (floored at 1),
    LW samples = (z/eps)^2 D (1 - p) / p."""
    p, aE = 10 ** lp, 10 ** laE
    q = 10 ** lk * math.pi / REL * (1 / np.sqrt(aE) + 1 / np.sqrt(aE * p)) * 10 ** lg
    return np.log10(q / np.maximum((Z90 / REL) ** 2 * 10 ** lD * (1 - p) / p, 1.0))


def logR_dec(lp, laE, lD, lg, lk):
    """log10 R as in W6: decision metric eps = 0.1 max(p, t) / p, no floor (bands._log_ratio)."""
    p, aE = 10 ** lp, 10 ** laE
    eps = REL * np.maximum(p, T_DEC) / p
    q = 10 ** lk * math.pi / eps * (1 / np.sqrt(aE) + 1 / np.sqrt(aE * p)) * 10 ** lg
    return np.log10(q / ((Z90 / REL) ** 2 * 10 ** lD * (1 - p) / p * (REL / eps) ** 2))


INPUTS = ["log p", "log a_E", "log D (LW variance over plain Monte Carlo)", "log gate ratio (V1)", "log kappa"]


def sobol(X, f, reps, seed):
    """First-order (Saltelli 2010) and total (Jansen) indices by pick-and-freeze, with every input drawn
    independently from its observed marginal (each column permuted on its own, for A and for B); AB_i = A with
    input i taken from B. Inputs with zero variance get 0."""
    rng = np.random.default_rng(seed)
    n, k = X.shape
    indep = lambda: np.column_stack([X[rng.permutation(n), j] for j in range(k)])
    num1, numT, var = np.zeros(k), np.zeros(k), 0.0
    for _ in range(reps):
        A, Bm = indep(), indep()
        yA, yB = f(*A.T), f(*Bm.T)
        var += np.var(np.r_[yA, yB])
        for i in range(k):
            ABi = A.copy()
            ABi[:, i] = Bm[:, i]
            yABi = f(*ABi.T)
            num1[i] += np.mean(yB * (yABi - yA))
            numT[i] += 0.5 * np.mean((yA - yABi) ** 2)
    V = var / reps
    return (num1 / reps) / V, (numT / reps) / V


def elasticities(X, f):
    x0 = np.median(X, 0)
    h = 1e-4
    out = []
    for i in range(X.shape[1]):
        a, b = x0.copy(), x0.copy()
        a[i] += h
        b[i] -= h
        out.append(float((f(*a[:, None]) - f(*b[:, None]))[0] / (2 * h)))
    return out


# ================================================================================================ stage 0
def stage0_zero_width():
    """0.1: why 0.343 (0.343-0.343). Checks the bootstrap unit and the tie structure of log R."""
    import bands as B
    import round3_w6 as W
    X = W.frame()
    W6 = X[(X.p >= WIN[0]) & (X.p <= WIN[1])].reset_index(drop=True)
    k = json.loads((B.RES / "frontier.json").read_text())["kappa_tuned_iae"]
    lr = np.round(B._log_ratio(W6, T_DEC, k, "v1").to_numpy(), 12)
    med = float(np.median(lr))
    rows_of = cluster_index(W6.subject_id.to_numpy())
    Bn = N["boot_small"]
    bmed = cluster_boot(rows_of, lambda i: np.median(lr[i]), Bn, SEED)
    bmean = cluster_boot(rows_of, lambda i: lr[i].mean(), Bn, SEED + 1)
    bshare = cluster_boot(rows_of, lambda i: (lr[i] < 0).mean(), Bn, SEED + 2)
    out = {"instances": int(len(W6)), "patients": int(len(rows_of)),
           "bootstrap_unit": "patients (bands.cluster_boot resamples subject_id groups with all their rows; checked: "
                             f"{len(rows_of)} clusters = {W6.subject_id.nunique()} patients)",
           "distinct_log_ratio_values": int(len(np.unique(lr))),
           "share_of_instances_tied_at_the_median_value": float((lr == med).mean()),
           "distribution": {"median_ratio": 10 ** med, "iqr_ratio": [10 ** q for q in pct(lr, [.25, .75])],
                            "p5_p95_ratio": [10 ** q for q in pct(lr, [.05, .95])], "share_below_1": float((lr < 0).mean())},
           "patient_bootstrap": {"median_ratio_ci95": [10 ** q for q in pct(bmed, [.025, .975])],
                                 "mean_log10_ratio": float(lr.mean()), "mean_log10_ratio_ci95": pct(bmean, [.025, .975]),
                                 "share_below_1_ci95": pct(bshare, [.025, .975])}}
    lo, hi = out["patient_bootstrap"]["median_ratio_ci95"]
    out["median_ci_relative_width"] = (hi - lo) / out["distribution"]["median_ratio"]
    tie = out["share_of_instances_tied_at_the_median_value"]
    out["median_rank_jitter_expected"] = 1 / math.sqrt(len(rows_of))  # sampling sd of the median's rank, as a share
    out["verdict"] = (
        "Neither (a) nor (b) alone. The bootstrap did resample patients (44,690 clusters), so (a) is ruled out. R is a "
        f"deterministic function of the instance, and {tie:.2%} of instances share exactly the median value, a block "
        f"wider than the median's sampling jitter under patient resampling (about {out['median_rank_jitter_expected']:.2%} "
        "of the distribution), so most resamples return the same median and the interval collapses "
        f"(relative width {out['median_ci_relative_width']:.1e}). The interval is not wrong, it is uninformative. Report the "
        "distribution (median, IQR, 5th/95th percentiles, share below 1), and patient-bootstrap CIs for the share below 1 "
        "and the mean log ratio, which do vary.")
    R["S0.1"] = out
    say("S0.1", out["verdict"][:120])
    return W6, lr, k


def stage0_partition(ok):
    """0.2: which R values are computed from formulas and which are measured, per experiment."""
    import bands as B
    rows = [json.loads(x) for x in Path("artifacts/lw_check/rows.jsonl").read_text().splitlines()]
    lwc = [r for r in rows if r.get("kb") == "artifacts/v2.1/kb" and r.get("shots") == 16 and r.get("alpha") == 0.2]
    c3 = []
    for b in B.PRIMARY:
        p = B.ART / b / "c3_sample.jsonl"
        if p.exists():
            c3 += [dict(json.loads(x), band=b) for x in p.read_text().splitlines()]
    w8 = json.loads((QP / "w8_kappa.json").read_text())["rows"]
    w5f = pd.concat([pd.read_parquet(q) for q in sorted(QP.glob("w5f_*.parquet"))])
    out = {
        "W4 benchmarks (13,477)": {"formula_derived": int(len(ok)), "measured": 0,
                                   "note": "LW from the exact variance formula, qLW from the IAE cost law"},
        "W8 kappa calibration (benchmarks)": {"qLW_emulated_LW_formula": int(sum(r.get("kappa_iae_16_0.2") is not None for r in w8)),
                                              "emulator_failed": int(sum(r.get("kappa_iae_16_0.2") is None for r in w8))},
        "W5 factorial": {"formula_derived": int((w5f.status == "ok").sum()), "measured": 0},
        "W6 clinical (window)": {"formula_derived": "all", "measured": 0},
        "lw_check clinical calibration": {"both_measured": int(sum(r.get("qlw") is not None and r.get("lw") is not None for r in lwc)),
                                          "note": "LW sampled on a coarse grid (smallest grid size that meets the criterion), "
                                                  "qLW emulated IAE; R in model evaluations (no gate ratio)"},
        "c3_sample (24 per band)": {"both_measured": int(sum(r.get("qlw_emulated") is not None for r in c3)),
                                    "emulator_stalled": int(sum(r.get("qlw_emulated") is None for r in c3)),
                                    "note": "LW empirical from 20 repeats, qLW emulated IAE; R in model evaluations"}}
    R["S0.2"] = out
    say("S0.2", {k: {a: b for a, b in v.items() if a != "note"} for k, v in out.items()})
    return lwc, c3, w8


# ================================================================================================ stage 1
def stage1(ok, W6, k6):
    # ---------------- S1.1 variance decomposition (primary): W4, and the clinical W6 window
    res = {}
    lg = np.log10(ok.ratio_v1.to_numpy())
    for kname in ("kappa_map", "kappa_const"):
        X = np.column_stack([np.log10(ok.p), np.log10(ok.a_E), np.log10(ok.D), lg, np.log10(ok[kname])])
        if kname == "kappa_const":
            assert np.allclose(logR_rel(*X.T), ok.logR, atol=1e-9), "cost expression does not reproduce R"
        S1, ST = sobol(X, logR_rel, N["sobol"], SEED)
        nets = cluster_index(ok.network.to_numpy())
        rng = np.random.default_rng(SEED + 5)
        bs1, bsT = [], []
        for b in range(N["sobol_boot"]):
            idx = np.concatenate([nets[j] for j in rng.integers(0, len(nets), len(nets))])
            a, t = sobol(X[idx], logR_rel, N["sobol_boot_reps"], SEED + 100 + b)
            bs1.append(a)
            bsT.append(t)
        bs1, bsT = np.array(bs1), np.array(bsT)
        res[f"W4 ({kname})"] = {"first_order": dict(zip(INPUTS, S1.tolist())), "total": dict(zip(INPUTS, ST.tolist())),
                                "input_spearman_correlations": pd.DataFrame(X, columns=INPUTS).corr("spearman").round(3).to_dict(),
                                "first_order_ci95": {i: pct(bs1[:, j], [.025, .975]) for j, i in enumerate(INPUTS)},
                                "total_ci95": {i: pct(bsT[:, j], [.025, .975]) for j, i in enumerate(INPUTS)},
                                "elasticity_at_median_instance": dict(zip(INPUTS, elasticities(X, logR_rel)))}
        if kname == "kappa_map":
            lo, hi = pct(bs1[:, 0], [.025, .975])
            log_test("S1.1", "C1", "primary", "Sobol first-order share of Var(log R) carried by log p, W4, kappa map",
                     float(S1[0]), lo, hi, note="descriptive sensitivity index, no p-value")
        say(f"S1.1 W4 {kname}: first-order", np.round(S1, 3), "total", np.round(ST, 3))
    # clinical W6 window (decision metric), on a random subsample for speed
    rng = np.random.default_rng(SEED)
    sub = W6.iloc[rng.choice(len(W6), min(len(W6), N["w6_sub"]), replace=False)].reset_index(drop=True)
    km = kappa_map_fn()(sub.aE.to_numpy(), sub.p.to_numpy())
    for kname, kv in (("kappa_map", km), ("kappa_const", np.full(len(sub), k6))):
        V = sub.lw_samples.to_numpy() / (Z90 / REL) ** 2
        Dd = V * sub.p.to_numpy() / (1 - sub.p.to_numpy())
        X = np.column_stack([np.log10(sub.p), np.log10(sub.aE), np.log10(Dd), np.log10(sub.ratio_v1), np.log10(kv)])
        if kname == "kappa_const":
            import bands as B
            assert np.allclose(logR_dec(*X.T), B._log_ratio(sub, T_DEC, k6, "v1"), atol=1e-9), "W6 cost expression"
        S1, ST = sobol(X, logR_dec, max(N["sobol"] // 5, 5), SEED)
        pats = cluster_index(sub.subject_id.to_numpy())
        bs1 = []
        rr = np.random.default_rng(SEED + 7)
        for b in range(max(N["sobol_boot"] // 2, 5)):
            idx = np.concatenate([pats[j] for j in rr.integers(0, len(pats), len(pats))])
            bs1.append(sobol(X[idx], logR_dec, 3, SEED + 300 + b)[0])
        bs1 = np.array(bs1)
        res[f"W6 window ({kname}, {len(sub):,} instance subsample)"] = {
            "first_order": dict(zip(INPUTS, S1.tolist())), "total": dict(zip(INPUTS, ST.tolist())),
            "first_order_ci95": {i: pct(bs1[:, j], [.025, .975]) for j, i in enumerate(INPUTS)},
            "elasticity_at_median_instance": dict(zip(INPUTS, elasticities(X, logR_dec)))}
        if kname == "kappa_map":
            lo, hi = pct(bs1[:, 0], [.025, .975])
            log_test("S1.1", "C1", "primary", "Sobol first-order share of Var(log R) carried by log p, W6 window, kappa map",
                     float(S1[0]), lo, hi, note="descriptive sensitivity index, patient-bootstrap CI, no p-value")
        say(f"S1.1 W6 {kname}: first-order", np.round(S1, 3), "total", np.round(ST, 3))
    res["note"] = ("Indices are computed under independent resampling of each input from its observed marginal; the observed "
                   "inputs are correlated, so first-order indices need not sum to 1 and the gap to the total indices is "
                   "interaction plus dependence. They describe the sensitivity of R to its inputs over the observed instances, "
                   "not a mechanism.")
    R["S1.1"] = res
    return res


def stage1_measured(ok, lwc, c3, w8):
    """S1.2 (sensitivity): rho(log R, log p) on the measured instances only."""
    out = {}
    rng_B = N["boot"]
    # c3_sample: both engines measured
    d = pd.DataFrame([r for r in c3 if r.get("qlw_emulated") and r.get("lw_empirical")])
    if len(d):
        lr, lp = np.log10(d.qlw_emulated / d.lw_empirical), np.log10(d.p)
        rows = [np.array([i]) for i in range(len(d))]
        b = cluster_boot(rows, lambda i: spearman(lp.to_numpy()[i], lr.to_numpy()[i]), rng_B)
        rho = spearman(lp, lr)
        out["c3_sample (both measured, clinical window)"] = {"n": int(len(d)), "rho": rho, "ci95": pct(b, [.025, .975]),
                                                             "within_band": {bd: spearman(np.log10(g.p), np.log10(g.qlw_emulated / g.lw_empirical))
                                                                             for bd, g in d.groupby("band")}}
        log_test("S1.2", "C1", "sensitivity", "Spearman(log R, log p), measured c3 sample", rho, *pct(b, [.025, .975]), p=boot_p(b))
    # lw_check: both measured (LW on a coarse grid)
    d = pd.DataFrame([r for r in lwc if r.get("qlw") and r.get("lw")])
    if len(d):
        lr, lp = np.log10(d.qlw / d.lw), np.log10(d.p)
        rows = [np.array([i]) for i in range(len(d))]
        b = cluster_boot(rows, lambda i: spearman(lp.to_numpy()[i], lr.to_numpy()[i]), rng_B)
        rho = spearman(lp, lr)
        out["lw_check (both measured, clinical calibration)"] = {"n": int(len(d)), "rho": rho, "ci95": pct(b, [.025, .975]),
                                                                 "distinct_lw_values": int(d.lw.nunique())}
        log_test("S1.2", "C1", "sensitivity", "Spearman(log R, log p), measured lw_check rows", rho, *pct(b, [.025, .975]), p=boot_p(b))
    # w8 benchmarks: qLW emulated, LW formula
    w = pd.DataFrame(w8).dropna(subset=["kappa_iae_16_0.2"])
    m = w.merge(ok[["network", "i", "a_E", "p", "D", "ratio_v1", "logR"]], on=["network", "i"], suffixes=("_w8", ""))
    if len(m):
        X = np.column_stack([np.log10(m.p), np.log10(m.a_E), np.log10(m.D), np.log10(m.ratio_v1), np.log10(m["kappa_iae_16_0.2"])])
        m["logR_meas"] = logR_rel(*X.T)
        nets = cluster_index(m.network.to_numpy())
        b = cluster_boot(nets, lambda i: spearman(np.log10(m.p.to_numpy()[i]), m.logR_meas.to_numpy()[i]), rng_B)
        rho = spearman(np.log10(m.p), m.logR_meas)
        out["w8 benchmarks (qLW emulated, LW formula)"] = {"n": int(len(m)), "networks": int(m.network.nunique()), "rho": rho,
                                                           "ci95_network_bootstrap": pct(b, [.025, .975]),
                                                           "rho_formula_R_same_instances": spearman(np.log10(m.p), m.logR)}
        log_test("S1.2", "C1", "sensitivity", "Spearman(log R, log p), w8 instances with emulated qLW", rho, *pct(b, [.025, .975]), p=boot_p(b))
    R["S1.2"] = out
    say("S1.2", {k: (v["n"], round(v["rho"], 3)) for k, v in out.items()})


def icc_anova(y, g):
    d = pd.DataFrame({"y": y, "g": g})
    grp = d.groupby("g").y
    n_i, m_i = grp.size().to_numpy(), grp.mean().to_numpy()
    Nn, k = len(d), len(n_i)
    msb = (n_i * (m_i - d.y.mean()) ** 2).sum() / (k - 1)
    msw = ((d.y - grp.transform("mean")) ** 2).sum() / (Nn - k)
    n0 = (Nn - (n_i ** 2).sum() / Nn) / (k - 1)
    sb = max((msb - msw) / n0, 0.0)
    return sb / (sb + msw), float(n_i.mean())


def stage1_clustering(ok):
    """S1.3 (primary): ICC, effective n, and pooled / within / mixed estimates side by side."""
    d = pd.DataFrame({"logR": ok.logR, "lp": np.log10(ok.p), "laE": np.log10(ok.a_E), "lD": np.log10(ok.D), "network": ok.network})
    icc, nbar = icc_anova(d.logR, d.network)
    nets = cluster_index(d.network.to_numpy())
    b_icc = cluster_boot(nets, lambda i: icc_anova(d.logR.to_numpy()[i], d.network.to_numpy()[i])[0], N["boot_small"])
    deff = 1 + (nbar - 1) * icc
    X = d[["lp", "laE", "lD"]].to_numpy()
    resid = d.logR - np.column_stack([np.ones(len(d)), X]) @ ols(X, d.logR.to_numpy())
    icc_res, _ = icc_anova(resid, d.network)
    pooled = ols(X, d.logR.to_numpy())
    b_pool = np.array([ols(X[i], d.logR.to_numpy()[i]) for i in
                       [np.concatenate([nets[j] for j in r]) for r in np.random.default_rng(SEED).integers(0, len(nets), (N["boot_small"], len(nets)))]])
    dm = d.copy()
    for c in ("logR", "lp", "laE", "lD"):
        dm[c] = d[c] - d.groupby("network")[c].transform("mean")
    Xw = dm[["lp", "laE", "lD"]].to_numpy()
    within = np.linalg.lstsq(Xw, dm.logR.to_numpy(), rcond=None)[0]
    b_with = np.array([np.linalg.lstsq(Xw[i], dm.logR.to_numpy()[i], rcond=None)[0] for i in
                       [np.concatenate([nets[j] for j in r]) for r in np.random.default_rng(SEED + 1).integers(0, len(nets), (N["boot_small"], len(nets)))]])
    names = ["log p", "log a_E", "log D"]
    out = {"icc_unconditional": icc, "icc_ci95_network_bootstrap": pct(b_icc, [.025, .975]), "icc_residual_after_fixed_effects": icc_res,
           "mean_cluster_size": nbar, "design_effect": deff, "effective_n": len(d) / deff, "n": int(len(d)), "networks": int(len(nets)),
           "pooled_ols": {n: {"coef": float(pooled[j + 1]), "ci95_network_bootstrap": pct(b_pool[:, j + 1], [.025, .975])} for j, n in enumerate(names)},
           "within_network_ols": {n: {"coef": float(within[j]), "ci95_network_bootstrap": pct(b_with[:, j], [.025, .975])} for j, n in enumerate(names)}}
    log_test("S1.3", "C1", "primary", "ICC of log R across networks (ANOVA estimator)", icc, *pct(b_icc, [.025, .975]),
             note=f"design effect {deff:.0f}, effective n {len(d) / deff:.0f}")
    lo, hi = out["within_network_ols"]["log p"]["ci95_network_bootstrap"]
    se = (hi - lo) / 3.92
    log_test("S1.3", "C1", "primary", "log p coefficient, within-network OLS (network bootstrap)", float(within[0]), lo, hi,
             p=float(2 * stats.norm.sf(abs(within[0]) / se)) if se > 0 else np.nan)
    if HAVE_SM:
        try:
            m = smf.mixedlm("logR ~ lp + laE + lD", d, groups=d["network"]).fit(reml=True)
            ci = m.conf_int()
            out["mixed_model"] = {n: {"coef": float(m.params[v]), "ci95": [float(ci.loc[v, 0]), float(ci.loc[v, 1])], "p": float(m.pvalues[v])}
                                  for n, v in zip(names, ["lp", "laE", "lD"])}
            out["mixed_model"]["network_variance"] = float(m.cov_re.iloc[0, 0])
            out["mixed_model"]["residual_variance"] = float(m.scale)
            out["mixed_model"]["icc_conditional"] = float(m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale))
            v = out["mixed_model"]["log p"]
            log_test("S1.3", "C1", "primary", "log p coefficient, mixed model (1 | network)", v["coef"], *v["ci95"], p=v["p"])
        except Exception as e:  # report, never hide
            out["mixed_model"] = f"failed: {e}"
            SKIPPED.append(f"S1.3 mixed model failed to fit: {e}")
    else:
        SKIPPED.append("S1.3 mixed model (statsmodels not installed): ICC by the ANOVA estimator, pooled and within-network "
                       "estimates with network-bootstrap CIs reported instead; no variance-components model fitted")
    R["S1.3"] = out
    say(f"S1.3 ICC {icc:.3f}, design effect {deff:.0f}, effective n {len(d) / deff:.0f}")


# ================================================================================================ stage 2
def stage2(ok, BM):
    # ---------------- S2.1 TOST on structure, within networks
    def within(g):
        r = BM._within_ranks(g, ["cost", "logR"])
        return float(np.corrcoef(r.cost, r.logR)[0, 1]) if r.cost.std() > 0 else np.nan
    est = within(ok)
    nets = cluster_index(ok.network.to_numpy())
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N["boot"]):
        idx = np.concatenate([nets[j] for j in rng.integers(0, len(nets), len(nets))])
        boots.append(within(ok.iloc[idx]))  # as benchmarks._net_boot: a network drawn twice keeps one label
    t = tost_corr(est, boots)
    R["S2.1"] = t
    log_test("S2.1", "C2", "primary", f"TOST |within-network rank r(cost, log R)| < {DELTA}", est, *t["ci90"], p=t["p_tost"],
             note="CI is 90% (TOST); equivalent" if t["equivalent"] else "CI is 90% (TOST); NOT equivalent")
    say("S2.1", t)

    # ---------------- S2.2 TOST on treewidth, per magnitude (W5 factorial)
    w5 = pd.concat([pd.read_parquet(q) for q in sorted(QP.glob("w5f_*.parquet"))], ignore_index=True)
    w5 = w5[w5.status == "ok"].copy()
    w5["logR"] = np.log10(w5[BM.R_COL])
    s22 = {}
    for b, g in w5.groupby("base_rate"):
        g = g.reset_index(drop=True)
        est = spearman(g.cost, g.logR)
        nets_b = cluster_index(g.network.to_numpy())
        bs = cluster_boot(nets_b, lambda i: spearman(g.cost.to_numpy()[i], g.logR.to_numpy()[i]), N["boot"])
        s22[str(b)] = tost_corr(est, bs)
        s22[str(b)]["by_width"] = {int(w): spearman(h.cost, h.logR) for w, h in g.groupby("width_param")}
        log_test("S2.2", "C3", "primary", f"TOST |Spearman(cost, log R)| < {DELTA}, b = {b}", est, *s22[str(b)]["ci90"],
                 p=s22[str(b)]["p_tost"], note="90% CI; " + ("equivalent" if s22[str(b)]["equivalent"] else "NOT equivalent"))
    R["S2.2"] = s22
    say("S2.2", {b: (round(v["estimate"], 3), [round(x, 3) for x in v["ci90"]], v["equivalent"]) for b, v in s22.items()})

    # ---------------- S2.3 nested models for network size
    d = pd.DataFrame({"logR": ok.logR, "cost": ok.cost, "lnodes": np.log10(ok.network_nodes), "network": ok.network})
    out = {}
    b1 = ols(d[["cost"]].to_numpy(), d.logR.to_numpy())
    b2 = ols(d[["cost", "lnodes"]].to_numpy(), d.logR.to_numpy())
    rng = np.random.default_rng(SEED + 3)
    bb1, bb2 = [], []
    for _ in range(N["boot"]):
        idx = np.concatenate([nets[j] for j in rng.integers(0, len(nets), len(nets))])
        bb1.append(ols(d[["cost"]].to_numpy()[idx], d.logR.to_numpy()[idx])[1])
        bb2.append(ols(d[["cost", "lnodes"]].to_numpy()[idx], d.logR.to_numpy()[idx])[1])
    out["pooled_ols_network_bootstrap"] = {"cost_coef_unadjusted": float(b1[1]), "ci95": pct(bb1, [.025, .975]),
                                           "cost_coef_adjusted_for_log_nodes": float(b2[1]), "ci95_adjusted": pct(bb2, [.025, .975])}
    per = ok.groupby("network").apply(lambda g: pd.Series({"rho": spearman(g.cost, g.logR) if g.cost.nunique() > 1 else np.nan,
                                                            "nodes": int(g.network_nodes.iloc[0]), "cost_sd": float(g.cost.std()),
                                                            "mean_logR": float(g.logR.mean()), "mean_cost": float(g.cost.mean())}))
    out["per_network"] = per.reset_index().to_dict("records")
    out["spearman_per_network_rho_vs_nodes"] = spearman(per.nodes, per.rho.fillna(0)) if per.rho.notna().sum() > 2 else np.nan
    lo, hi = pct(bb2, [.025, .975])
    se = (hi - lo) / 3.92
    log_test("S2.3", "C2", "primary", "cost coefficient adjusted for log n_nodes (pooled OLS, network bootstrap)", float(b2[1]), lo, hi,
             p=float(2 * stats.norm.sf(abs(b2[1]) / se)) if se > 0 else np.nan)
    if HAVE_SM:
        try:
            m1 = smf.mixedlm("logR ~ cost", d, groups=d["network"]).fit(reml=False)
            m2 = smf.mixedlm("logR ~ cost + lnodes", d, groups=d["network"]).fit(reml=False)
            lrt = 2 * (m2.llf - m1.llf)
            c1, c2 = m1.conf_int(), m2.conf_int()
            out["mixed_models"] = {"cost_unadjusted": {"coef": float(m1.params["cost"]), "ci95": [float(c1.loc["cost", 0]), float(c1.loc["cost", 1])], "p": float(m1.pvalues["cost"])},
                                   "cost_adjusted": {"coef": float(m2.params["cost"]), "ci95": [float(c2.loc["cost", 0]), float(c2.loc["cost", 1])], "p": float(m2.pvalues["cost"])},
                                   "lrt_chi2_df1": float(lrt), "lrt_p": float(stats.chi2.sf(lrt, 1))}
            v = out["mixed_models"]["cost_adjusted"]
            log_test("S2.3", "C2", "primary", "cost coefficient adjusted for log n_nodes, mixed model (1 | network)", v["coef"], *v["ci95"], p=v["p"])
            log_test("S2.3", "C2", "primary", "LRT: adding log n_nodes to log R ~ cost + (1 | network)", float(lrt), p=float(stats.chi2.sf(lrt, 1)))
        except Exception as e:
            out["mixed_models"] = f"failed: {e}"
            SKIPPED.append(f"S2.3 mixed models failed to fit: {e}")
    else:
        SKIPPED.append("S2.3 mixed-model likelihood-ratio test (statsmodels not installed): pooled OLS with and without "
                       "log n_nodes, network-bootstrap CIs, reported instead")
    R["S2.3"] = out
    say("S2.3", out["pooled_ols_network_bootstrap"])
    return w5, per


# ================================================================================================ stage 3
def jonckheere(groups):
    """JT for a decreasing trend across ordered groups (first group = highest b). -> J, z, one-sided p, concordance."""
    J, pairs = 0.0, 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            J += stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided").statistic  # pairs x_i > x_j (+ ties/2)
            pairs += len(groups[i]) * len(groups[j])
    n = np.array([len(g) for g in groups])
    Nn = n.sum()
    mean = (Nn ** 2 - (n ** 2).sum()) / 4
    var = (Nn ** 2 * (2 * Nn + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    z = (J - mean) / math.sqrt(var)
    return float(J), float(z), float(stats.norm.sf(z)), float(J / pairs)


def cochran_armitage(counts, totals, scores):
    counts, totals, scores = map(lambda a: np.asarray(a, float), (counts, totals, scores))
    pbar = counts.sum() / totals.sum()
    T = (scores * (counts - totals * pbar)).sum()
    V = pbar * (1 - pbar) * ((totals * scores ** 2).sum() - (totals * scores).sum() ** 2 / totals.sum())
    if V <= 0:
        return np.nan, np.nan
    z = T / math.sqrt(V)
    return float(z), float(stats.norm.sf(z))


def stage3(w5, BM):
    bs = sorted(w5.base_rate.unique(), reverse=True)  # 0.3, 0.03, ... : lower b = higher score
    w5 = w5.reset_index(drop=True)
    w5["below"] = (w5[BM.R_COL] < 1).astype(int)
    # ---------------- S3.1 trend at every width
    per = {}
    for w, g in w5.groupby("width_param"):
        grp = [g[g.base_rate == b].logR.to_numpy() for b in bs]
        J, z, p, conc = jonckheere(grp)
        cz, cp = cochran_armitage([g[g.base_rate == b].below.sum() for b in bs], [(g.base_rate == b).sum() for b in bs], range(len(bs)))
        netmed = g.groupby(["network", "base_rate"]).logR.median().reset_index()
        Jn, zn, pn, _ = jonckheere([netmed[netmed.base_rate == b].logR.to_numpy() for b in bs])
        per[int(w)] = {"jt_z": z, "jt_p_one_sided": p, "concordance": conc, "ca_z": cz, "ca_p_one_sided": cp,
                       "jt_network_medians_z": zn, "jt_network_medians_p": pn,
                       "share_R_below_1_by_b": [float(g[g.base_rate == b].below.mean()) for b in bs]}
    jt_max = max(v["jt_p_one_sided"] for v in per.values())
    ca_vals = [v["ca_p_one_sided"] for v in per.values() if np.isfinite(v["ca_p_one_sided"])]
    jtn_max = max(v["jt_network_medians_p"] for v in per.values())
    R["S3.1"] = {"by_width": per, "b_order": bs,
                 "jt_z_min_median_max": [float(np.min([v["jt_z"] for v in per.values()])), float(np.median([v["jt_z"] for v in per.values()])),
                                         float(np.max([v["jt_z"] for v in per.values()]))],
                 "intersection_union_p_jt": jt_max, "intersection_union_p_ca": max(ca_vals) if ca_vals else np.nan,
                 "intersection_union_p_jt_network_medians": jtn_max,
                 "note": "'at every width' is an intersection of 8 hypotheses: the claim holds at level alpha if the largest "
                         "of the 8 one-sided p-values is below alpha (intersection-union test, no adjustment needed). The "
                         "instance-level JT ignores clustering by network; the network-median version is the clustered check."}
    log_test("S3.1", "C4", "primary", "Jonckheere-Terpstra, decreasing log R as b falls, every width (max p of 8)",
             R["S3.1"]["jt_z_min_median_max"][0], p=jt_max, note="effect = smallest z across widths")
    log_test("S3.1", "C4", "primary", "Cochran-Armitage, rising share R < 1 as b falls, every width (max p of 8)",
             float(np.nanmin([v["ca_z"] for v in per.values()])), p=R["S3.1"]["intersection_union_p_ca"], note="effect = smallest z")
    log_test("S3.1", "C4", "sensitivity", "JT on network medians (clustered), every width (max p of 8)", np.nan, p=jtn_max)
    say("S3.1 IUT p: JT", jt_max, "CA", R["S3.1"]["intersection_union_p_ca"], "JT network medians", jtn_max)

    # ---------------- S3.2 width main effect and b x width interaction (equivalence), cluster bootstrap fallback
    widths = sorted(w5.width_param.unique())
    cells = {k: cluster_index(g.network.to_numpy()) for k, g in w5.groupby(["width_param", "base_rate"])}
    cell_rows = {k: g.index.to_numpy() for k, g in w5.groupby(["width_param", "base_rate"])}
    below = w5.below.to_numpy()

    def summaries(pick):
        sh = {k: below[cell_rows[k][np.concatenate([v[j] for j in pick[k]])]].mean() for k, v in cells.items()}
        width_eff = [sh[(widths[-1], b)] - sh[(widths[0], b)] for b in bs]
        slopes = [np.polyfit(range(len(bs)), [sh[(w, b)] for b in bs], 1)[0] for w in widths]
        return width_eff, max(slopes) - min(slopes), slopes
    point = summaries({k: np.arange(len(v)) for k, v in cells.items()})
    rng = np.random.default_rng(SEED + 9)
    boots = [summaries({k: rng.integers(0, len(v), len(v)) for k, v in cells.items()}) for _ in range(N["boot"])]
    we = np.array([b[0] for b in boots])
    spread = np.array([b[1] for b in boots])
    out = {"width_effect_widest_minus_narrowest": {}, "margin": WIDTH_MARGIN}
    for j, b in enumerate(bs):
        lo90, hi90 = pct(we[:, j], [.05, .95])
        eq = bool(-WIDTH_MARGIN < lo90 and hi90 < WIDTH_MARGIN)
        shares_b = [below[cell_rows[(w, b)]].mean() for w in widths]
        floor_ceiling = bool(all(x in (0.0, 1.0) for x in np.round(shares_b, 6)))
        out["width_effect_widest_minus_narrowest"][str(b)] = {"effect": float(point[0][j]), "ci90": [lo90, hi90], "equivalent": eq,
                                                              "floor_or_ceiling": floor_ceiling,
                                                              "note": "share is 0 or 1 in every width: the test is uninformative here" if floor_ceiling else ""}
        p_t = float(max((we[:, j] <= -WIDTH_MARGIN).mean(), (we[:, j] >= WIDTH_MARGIN).mean()))
        log_test("S3.2", "C3", "primary", f"equivalence of width effect on share R < 1 (|d| < {WIDTH_MARGIN}), b = {b}",
                 float(point[0][j]), lo90, hi90, p=max(p_t, 1 / len(we)),
                 note="90% CI; bootstrap TOST p; " + ("equivalent" if eq else "NOT equivalent") + ("; floor/ceiling, uninformative" if floor_ceiling else ""))
    out["trend_slope_by_width"] = dict(zip(map(int, widths), map(float, point[2])))
    out["slope_spread_max_minus_min"] = {"value": float(point[1]), "ci95": pct(spread, [.025, .975])}
    log_test("S3.2", "C3", "primary", "b x width interaction: spread of the trend slope across widths", float(point[1]),
             *pct(spread, [.025, .975]), note="share-per-step slopes; spread near 0 = no interaction (no p-value)")
    if HAVE_SM:
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            dd = w5.assign(lb=-np.log10(w5.base_rate), wz=(w5.width_param - w5.width_param.mean()) / w5.width_param.std())
            m = BinomialBayesMixedGLM.from_formula("below ~ lb * wz", {"network": "0 + C(network)"}, dd).fit_vb()
            out["glmm_variational"] = {n: {"posterior_mean": float(mu), "posterior_sd": float(sd)}
                                       for n, mu, sd in zip(m.model.exog_names, m.fe_mean, m.fe_sd)}
            out["glmm_note"] = "variational Bayes fit (statsmodels BinomialBayesMixedGLM); near-separation at b <= 0.003 inflates the coefficients"
        except Exception as e:
            out["glmm_variational"] = f"failed: {e}"
            SKIPPED.append(f"S3.2 logistic GLMM failed: {e}")
    else:
        SKIPPED.append("S3.2 logistic GLMM (statsmodels not installed): stratified Cochran-Armitage per width (S3.1) plus "
                       "network-bootstrap equivalence of the width effect and of the slope spread, reported instead")
    R["S3.2"] = out
    say("S3.2", out["width_effect_widest_minus_narrowest"])


def stage3_kappa():
    """S3.3 (sensitivity) map calibration and out-of-sample errors; S3.4 (primary) informative missingness."""
    m = json.loads((QP / "kappa_map.json").read_text())
    grid = pd.DataFrame(json.loads((QP / "w8_kappa_grid_0.json").read_text()))
    w8 = pd.DataFrame(json.loads((QP / "w8_kappa.json").read_text())["rows"])
    clin_p = QP / "kappa_clinical.jsonl"
    clin = pd.DataFrame([json.loads(x) for x in clin_p.read_text().splitlines()]) if clin_p.exists() else pd.DataFrame()
    out = {}
    f_iae = kappa_map_fn("kappa_iae")
    sets = {"benchmarks (50)": (w8.a_E, w8.p, w8["kappa_iae_16_0.2"])}
    if len(clin):
        sets["clinical (46)"] = (clin.a_E, clin.p, clin.kappa_iae)
    for name, (aE, p, k) in sets.items():
        ok = k.notna().to_numpy()
        obs = np.log10(k[ok].to_numpy(float))
        pred = np.log10(f_iae(aE[ok].to_numpy(), p[ok].to_numpy()))
        err = np.abs(pred - obs)
        coef = ols(pred[:, None], obs)
        rng = np.random.default_rng(SEED)
        bc = np.array([ols(pred[i][:, None], obs[i]) for i in [rng.integers(0, len(obs), len(obs)) for _ in range(N["boot"])]])
        out[name] = {"n": int(ok.sum()), "calibration_intercept": float(coef[0]), "intercept_ci95": pct(bc[:, 0], [.025, .975]),
                     "calibration_slope": float(coef[1]), "slope_ci95": pct(bc[:, 1], [.025, .975]),
                     "abs_log10_error_quantiles_10_25_50_75_90": pct(err, [.1, .25, .5, .75, .9]),
                     "share_error_above_bar": float((err > ERR_BAR).mean()), "max_error": float(err.max())}
        log_test("S3.3", "C5", "sensitivity", f"kappa map calibration slope (target 1), {name}", float(coef[1]),
                 *pct(bc[:, 1], [.025, .975]), p=boot_p(bc[:, 1], 1.0))
        log_test("S3.3", "C5", "sensitivity", f"kappa map calibration intercept (target 0), {name}", float(coef[0]),
                 *pct(bc[:, 0], [.025, .975]), p=boot_p(bc[:, 0], 0.0))
    # out of sample: leave one seed out, and leave one cell out, on the grid itself
    ae, pp = m["a_E"], m["p"]
    for est in ("kappa_iae", "kappa_mlae16"):
        g = grid.dropna(subset=[est])
        loso = []
        for s in sorted(grid.seed.unique()):
            rest = g[g.seed != s].groupby(["a_E", "p"])[est].apply(lambda v: np.median(np.log10(v)))
            held = g[g.seed == s]
            for _, r in held.iterrows():
                if (r.a_E, r.p) in rest.index:
                    loso.append(abs(rest[(r.a_E, r.p)] - np.log10(r[est])))
        M = np.array(m["estimators"][est]["log10_kappa"], float)
        loco = []
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isfinite(M[i, j]):
                    continue
                nb = [M[a, b] for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                      if 0 <= a < M.shape[0] and 0 <= b < M.shape[1] and np.isfinite(M[a, b])]
                if nb:
                    loco.append(abs(np.mean(nb) - M[i, j]))
        out[f"out_of_sample_{est}"] = {
            "note": "The map was fitted on the grid, and the calibration instances above were not used to fit it (they are "
                    "out of sample for the map). On the grid: leave-one-seed-out = the held-out seed against the median of "
                    "the other two at the same cell; leave-one-cell-out = the cell against the mean of its 4 neighbours.",
            "leave_one_seed_out_median_abs_error": float(np.median(loso)) if loso else np.nan,
            "leave_one_seed_out_share_above_bar": float(np.mean(np.array(loso) > ERR_BAR)) if loso else np.nan,
            "leave_one_cell_out_median_abs_error": float(np.median(loco)) if loco else np.nan,
            "leave_one_cell_out_share_above_bar": float(np.mean(np.array(loco) > ERR_BAR)) if loco else np.nan}
    R["S3.3"] = out
    say("S3.3", {k: (v.get("calibration_slope"), v.get("slope_ci95")) for k, v in out.items() if "slope_ci95" in v})

    # ---------------- S3.4 informative missingness
    miss = {}
    data = [("grid", "IAE", grid.a_E, grid.p, grid.kappa_iae), ("grid", "MLAE16", grid.a_E, grid.p, grid.kappa_mlae16),
            ("benchmarks", "IAE", w8.a_E, w8.p, w8["kappa_iae_16_0.2"]), ("benchmarks", "MLAE16", w8.a_E, w8.p, w8["kappa_mlae_N16"])]
    if len(clin):
        data.append(("clinical", "MLAE16", clin.a_E, clin.p, clin.kappa_mlae16))
    for ds, est, aE, p, k in data:
        y = k.isna().to_numpy(float)
        X = np.column_stack([np.log10(aE), np.log10(p)])
        rec = {"n": int(len(y)), "failed": int(y.sum())}
        if 0 < y.sum() < len(y):
            b, se, ll, conv, sep = logistic_irls(X, y)
            pbar = y.mean()
            ll0 = float(len(y) * (pbar * np.log(pbar) + (1 - pbar) * np.log(1 - pbar)))
            lrt = 2 * (ll - ll0)
            rec.update({"coef_log_aE": float(b[1]), "se_log_aE": float(se[1]), "coef_log_p": float(b[2]), "se_log_p": float(se[2]),
                        "lrt_chi2_df2": float(lrt), "lrt_p": float(stats.chi2.sf(lrt, 2)), "converged": conv, "separation_suspected": sep})
            kk = k.dropna().to_numpy(float)
            rec["median_kappa_successes"] = float(np.median(kk))
            rec["median_kappa_failures_imputed_at_worst"] = float(np.median(np.r_[kk, np.full(int(y.sum()), kk.max())]))
            rec["bias_direction"] = ("failures concentrate at small amplitudes; the kappa summary is conditional on convergence "
                                     "and biased toward the easier instances (optimistic)"
                                     if (b[1] < 0 or b[2] < 0) else "failures do not concentrate at small amplitudes")
            log_test("S3.4", "C5", "primary", f"failure depends on instance location (LRT, 2 df): {est} on {ds}", float(lrt),
                     p=float(stats.chi2.sf(lrt, 2)), note=f"{int(y.sum())}/{len(y)} failed" + ("; separation suspected" if sep else ""))
        miss[f"{est} | {ds}"] = rec
    R["S3.4"] = miss
    say("S3.4", {k: (v["failed"], v["n"], round(v.get("lrt_p", np.nan), 4)) for k, v in miss.items()})


# ================================================================================================ stage 4 (clinical)
def ac1_kappa(X, K=4):
    """X: items x raters, categories 0..K-1. -> observed agreement, Fleiss kappa, Gwet AC1, PABAK, marginals."""
    n = X.shape[1]
    cnt = np.stack([(X == c).sum(1) for c in range(K)], 1)
    Pi = (cnt * (cnt - 1)).sum(1) / (n * (n - 1))
    pa = Pi.mean()
    pi = cnt.sum(0) / cnt.sum()
    pe_k = (pi ** 2).sum()
    pe_g = (pi * (1 - pi)).sum() / (K - 1)
    kappa = (pa - pe_k) / (1 - pe_k) if pe_k < 1 - 1e-12 else np.nan
    ac1 = (pa - pe_g) / (1 - pe_g)
    return float(pa), float(kappa), float(ac1), float((K * pa - 1) / (K - 1)), pi.tolist()


def stage4_agreement():
    import clinical as C
    cfg = C.cfg21()
    P = cfg["propositions"]
    props = [k for k, p in P.items() if p["kind"] not in ("diagnosis", "demographic")]
    T = pd.read_parquet(C.ART / "sample_texts.parquet", columns=["note_id"])
    cat = {"present": 0, "possible": 1, "absent": 2, "unknown": 3}
    mats = []
    for model in C.models():
        G = C._ground(model)
        A = np.full((len(T), len(props)), 3)
        for i, nid in enumerate(T.note_id):
            f = G.get(nid, {"findings": {}})["findings"]
            for j, k in enumerate(props):
                A[i, j] = cat.get(f.get(k, {}).get("status", "unknown"), 3)
        mats.append(A)
    X = np.stack(mats, 2)  # notes x props x models
    per = {}
    for j, k in enumerate(props):
        pa, kap, ac1, pabak, marg = ac1_kappa(X[:, j, :])
        per[k] = {"kind": P[k]["kind"], "observed_agreement": pa, "fleiss_kappa": kap, "gwet_ac1": ac1, "pabak": pabak,
                  "marginals": dict(zip(cat, marg))}

    def med(D, key):
        return float(np.nanmedian([v[key] for v in D.values()]))
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N["boot_agree"]):
        idx = rng.integers(0, X.shape[0], X.shape[0])
        st = [ac1_kappa(X[idx, j, :]) for j in range(len(props))]
        boots.append([np.nanmedian([s[i] for s in st]) for i in range(4)])
    boots = np.array(boots)
    keys = ["observed_agreement", "fleiss_kappa", "gwet_ac1", "pabak"]
    summ = {k: {"median": med(per, k), "ci95_note_bootstrap": pct(boots[:, i], [.025, .975])} for i, k in enumerate(keys)}
    gap = summ["gwet_ac1"]["median"] - summ["fleiss_kappa"]["median"]
    gb = boots[:, 2] - boots[:, 1]
    marg = {c: float(np.mean([v["marginals"][c] for v in per.values()])) for c in cat}
    out = {"per_item": per, "medians": summ, "mean_category_marginals": marg, "ac1_minus_kappa": gap,
           "ac1_minus_kappa_ci95": pct(gb, [.025, .975]), "by_kind": {kd: {k: float(np.nanmedian([v[k] for v in per.values() if v["kind"] == kd]))
                                                                         for k in keys} for kd in ("history", "medication", "lab")},
           "rule": f"AC1 median at least {AC1_GAP} above the kappa median means the 'models do not agree' reading is partly a "
                   "prevalence artifact of kappa",
           "kappa_artifact": bool(gap >= AC1_GAP)}
    log_test("S4.3", "C9", "primary", "Gwet AC1 minus Fleiss kappa (medians over 39 items)", gap, *pct(gb, [.025, .975]), p=boot_p(gb),
             note="prevalence adjustment; " + ("kappa reading is partly an artifact" if gap >= AC1_GAP else "kappa reading stands"))
    R["S4.3"] = out
    say("S4.3", {k: round(v["median"], 3) for k, v in summ.items()}, "artifact:", out["kappa_artifact"])


def stage4_predictions():
    """Every clinical score matrix the audit needs, reproducing clinical.py's settings (same seeds)."""
    import clinical as C
    import yaml
    from qddx import kb
    from qddx.engine import platt
    from qddx.evaluate import fit_lr, lr_predict
    cfg, df, tr, dx, ev = C._setup()
    te = df[~tr].reset_index(drop=True)
    bn = C.BN_load()
    lr = fit_lr(cfg, C.KB)
    labs = np.array([cfg["propositions"][k]["kind"] == "lab" for k in ev])
    cal = cfg["engine"].get("calibration") or {}

    def net(t, b=bn):
        return kb.posteriors_by_pattern(cfg, b, t)[0]

    def lrp(t):
        M = t[ev].to_numpy(float)
        M[M < 0] = np.nan
        return lr_predict(lr, cfg, M)
    S = {}
    t0 = time.perf_counter()
    S["base|network"] = net(te)
    S["base|network_platt"] = np.stack([platt(S["base|network"][:, i], *cal[d]) if d in cal else S["base|network"][:, i]
                                        for i, d in enumerate(dx)], 1)
    S["base|lr"] = lrp(te)
    for scope_key, scope in (("all", "all findings"), ("labs", "labs only")):
        for m in (0.1, 0.3, 0.5, 0.7, 0.9):
            t = te.copy()
            rng = np.random.default_rng(C.SEED_X + int(m * 100) + (0 if scope == "all findings" else 1000))
            E = t[ev].to_numpy()
            hide = (rng.random(E.shape) < m) & (E >= 0) & (labs[None, :] if scope == "labs only" else True)
            E[hide] = -1
            t[ev] = E
            S[f"mask_{scope_key}_{m:g}|network"], S[f"mask_{scope_key}_{m:g}|lr"] = net(t), lrp(t)
            say(f"  predictions: mask {scope_key} {m:g} ({time.perf_counter() - t0:.0f}s)")
    rng = np.random.default_rng(C.SEED_X + 99)  # negative controls, as clinical.negctrl
    trd = df[tr].copy()
    trd[dx] = trd[dx].to_numpy()[rng.permutation(len(trd))]
    S["negctrl_a|network"] = net(te, kb.learn_bn(cfg, trd))
    rng = np.random.default_rng(C.SEED_X + 99)
    t = te.copy()
    t[ev] = t[ev].to_numpy()[rng.permutation(len(t))]
    S["negctrl_b|network"], S["negctrl_b|lr"] = net(t), lrp(t)
    say(f"  predictions: negative controls ({time.perf_counter() - t0:.0f}s)")
    S = {k: np.round(v, 12) for k, v in S.items()}  # as clinical._score: elimination round-off would rank ties at random
    # subgroups, as clinical.subgroups
    root = Path(yaml.safe_load(open("configs/qddx.yaml"))["paths"]["mimic_root"]) / "hosp"
    pth = lambda n: (root / n / n) if (root / n).is_dir() else root / n
    adm = pd.read_csv(pth("admissions.csv"), usecols=["hadm_id", "race"])
    pat = pd.read_csv(pth("patients.csv"), usecols=["subject_id", "gender"])
    tg = te[["hadm_id", "subject_id", "age_ge_65"]].merge(adm, on="hadm_id", how="left").merge(pat, on="subject_id", how="left")
    assert len(tg) == len(te)
    tg["race"] = tg.race.fillna("UNKNOWN")
    vc = tg.race.value_counts()
    small = set(vc[vc < 500].index)
    groups = {"sex": tg.gender.to_numpy(), "age": np.where(tg.age_ge_65 == 1, ">=65", "<65"),
              "race": np.where(tg.race.isin(small), "OTHER (merged, < 500 each)", tg.race).astype(str)}
    return S, te[dx].to_numpy(), te.subject_id.to_numpy(), dx, groups


def stage4(S, Y, g, dx, groups):
    """S4.1, S4.2 (primary), S2.4 and S4.4 (sensitivity): one patient-clustered bootstrap shared by all comparisons."""
    aucs = {k: [AUC(Y[:, i], v[:, i]) for i in range(len(dx))] for k, v in S.items()}
    sub_masks = {(col, val): (arr == val).astype(float) for col, arr in groups.items() for val in np.unique(arr)}
    sub_scores = [AUC(Y[:, i], S["base|network_platt"][:, i]) for i in range(len(dx))]

    def macro(objs, w):
        return float(np.nanmean([a(w) for a in objs]))
    w1 = np.ones(len(Y))
    point = {k: macro(v, w1) for k, v in aucs.items()}
    point_sub = {key: macro(sub_scores, m) for key, m in sub_masks.items()}
    u, inv = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(SEED)
    Bn = N["boot_patient"]
    boot = {k: np.empty(Bn) for k in aucs}
    boot_sub = {k: np.empty(Bn) for k in sub_masks}
    t0 = time.perf_counter()
    for b in range(Bn):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)
        for k, v in aucs.items():
            boot[k][b] = macro(v, w)
        for key, m in sub_masks.items():
            boot_sub[key][b] = macro(sub_scores, w * m)
        if b % max(Bn // 10, 1) == 0:
            say(f"  patient bootstrap {b}/{Bn} ({time.perf_counter() - t0:.0f}s)")
    # ---------------- S4.1 network minus LR, unmasked and at every masking level
    s41 = {}
    diffs = {}
    for scope in ("all", "labs"):
        ms = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        for m in ms:
            kn, kl = ("base|network", "base|lr") if m == 0 else (f"mask_{scope}_{m:g}|network", f"mask_{scope}_{m:g}|lr")
            d, db = point[kn] - point[kl], boot[kn] - boot[kl]
            diffs[(scope, m)] = (d, db)
            if m == 0 and scope == "labs":
                continue  # the unmasked comparison is logged once
            s41[f"{scope}|{m:g}"] = {"network": point[kn], "lr": point[kl], "difference": d, "ci95": pct(db, [.025, .975]), "p": boot_p(db)}
            log_test("S4.1", "C7", "primary", f"macro AUROC network - LR, mask {scope} m = {m:g}", d, *pct(db, [.025, .975]), p=boot_p(db))
    R["S4.1"] = s41
    say("S4.1", {k: (round(v["difference"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in s41.items()})
    # ---------------- S4.2 the crossover as an interaction: slope of the difference in m
    s42 = {}
    for scope in ("all", "labs"):
        ms = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        pts = np.array([diffs[(scope, m)][0] for m in ms])
        bts = np.stack([diffs[(scope, m)][1] for m in ms], 1)
        slope = float(np.polyfit(ms, pts, 1)[0])
        bslope = np.array([np.polyfit(ms, r, 1)[0] for r in bts])
        first = next((m for m in ms if pct(diffs[(scope, m)][1], [.025])[0] > 0), None)
        s42[scope] = {"slope_of_difference_per_unit_m": slope, "ci95": pct(bslope, [.025, .975]), "p": boot_p(bslope),
                      "first_m_where_ci_excludes_0_network_better": first,
                      "last_m_where_ci_excludes_0_lr_better": max((m for m in ms if pct(diffs[(scope, m)][1], [.975])[0] < 0), default=None)}
        log_test("S4.2", "C8", "primary", f"model x m interaction (slope of network - LR in m), mask {scope}", slope,
                 *pct(bslope, [.025, .975]), p=boot_p(bslope))
    R["S4.2"] = s42
    say("S4.2", s42)
    # ---------------- S2.4 negative controls as equivalence (sensitivity)
    s24 = {}
    for k in ("negctrl_a|network", "negctrl_b|network", "negctrl_b|lr"):
        lo90, hi90 = pct(boot[k], [.05, .95])
        eq = bool(0.5 - NEG_MARGIN < lo90 and hi90 < 0.5 + NEG_MARGIN)
        distinct = int(np.mean([len(np.unique(S[k][:, i])) for i in range(len(dx))]))
        ptost = float(max((boot[k] <= 0.5 - NEG_MARGIN).mean(), (boot[k] >= 0.5 + NEG_MARGIN).mean(), 1 / Bn))
        s24[k] = {"auroc": point[k], "ci90": [lo90, hi90], "equivalent": eq, "p_tost": ptost,
                  "mean_distinct_predicted_values_per_diagnosis": distinct,
                  "degenerate": distinct <= 1}
        log_test("S2.4", "C6", "sensitivity", f"TOST |AUROC - 0.5| < {NEG_MARGIN}: {k}", point[k], lo90, hi90, p=ptost,
                 note=("degenerate: constant predictions, AUROC is 0.5 by construction" if distinct <= 1 else "90% CI"))
    R["S2.4"] = s24
    say("S2.4", {k: (round(v["auroc"], 4), v["ci90"], v["equivalent"], v["degenerate"]) for k, v in s24.items()})
    # ---------------- S4.4 subgroups: differences with CIs, Holm within the subgroup family
    s44 = {"auroc": {f"{c}={v}": {"value": point_sub[(c, v)], "ci95": pct(boot_sub[(c, v)], [.025, .975]),
                                  "n": int(sub_masks[(c, v)].sum())} for c, v in sub_masks}}
    comps = []
    vals = {c: sorted({v for cc, v in sub_masks if cc == c}) for c in groups}
    if {"F", "M"} <= set(vals["sex"]):
        comps.append((("sex", "F"), ("sex", "M")))
    comps.append((("age", ">=65"), ("age", "<65")))
    ref = next((v for v in vals["race"] if v.upper() == "WHITE"), None)
    if ref:
        comps += [(("race", v), ("race", ref)) for v in vals["race"] if v != ref]
    s44["differences"] = {}
    for a, b in comps:
        d, db = point_sub[a] - point_sub[b], boot_sub[a] - boot_sub[b]
        s44["differences"][f"{a[0]}: {a[1]} - {b[1]}"] = {"difference": d, "ci95": pct(db, [.025, .975]), "p": boot_p(db)}
        log_test("S4.4", "C7", "secondary-subgroup", f"macro AUROC (network + Platt) {a[0]}: {a[1]} - {b[1]}", d, *pct(db, [.025, .975]),
                 p=boot_p(db), prespecified=True)
    # calibration: slope against 1 from the round-3 calibration run (its 100 patient-bootstrap resamples)
    cal_p = Path("artifacts/clinical/structured_calibration.json")
    if cal_p.exists():
        cal = json.loads(cal_p.read_text())
        s44["calibration"] = {}
        for model, per in cal.items():
            for d_, st in per.items():
                lo, hi = st["ci95"]["slope"]
                se = (hi - lo) / 3.92
                p = float(2 * stats.norm.sf(abs(st["slope"] - 1) / se)) if se > 0 else np.nan
                s44["calibration"][f"{model} | {d_}"] = {"ece": st["ece_10"], "ece_ci95": st["ci95"]["ece_10"], "slope": st["slope"],
                                                         "slope_ci95": [lo, hi], "p_slope_eq_1": p}
                log_test("S4.4", "C7", "sensitivity", f"calibration slope = 1: {model}, {d_}", st["slope"], lo, hi, p=p,
                         note="CI from the round-3 run (100 patient-bootstrap resamples)")
    else:
        SKIPPED.append("S4.4 calibration tests: artifacts/clinical/structured_calibration.json not found")
    R["S4.4"] = s44
    say("S4.4", {k: (round(v["difference"], 4), [round(x, 4) for x in v["ci95"]]) for k, v in s44["differences"].items()})


# ================================================================================================ report
def verdicts():
    V = {}
    s11 = R.get("S1.1", {}).get("W4 (kappa_map)")
    if s11:
        share = s11["first_order"]["log p"]
        meas = [v for v in R.get("S1.2", {}).values()]
        meas_ok = all((v.get("ci95") or v.get("ci95_network_bootstrap"))[0] > 0 for v in meas) if meas else False
        V["C1"] = ("Supported with caveats" if share >= 0.5 and meas_ok else "Overstated",
                   f"Over the observed instances, log p carries {share:.0%} of the variance of log R (first-order Sobol index, "
                   f"network-bootstrap CI {s11['first_order_ci95']['log p'][0]:.0%}-{s11['first_order_ci95']['log p'][1]:.0%}); "
                   "R is most sensitive to the queried posterior. Because R is computed from p by the cost law, this is a "
                   "sensitivity statement, not evidence that p drives R." +
                   ("" if meas_ok else " The correlation does not hold with a CI above 0 on every measured subset; say so."))
    t = R.get("S2.1")
    if t:
        V["C2"] = ("Supported", f"Within networks, we exclude an association between exact-inference cost and R larger than "
                                f"|r| = {DELTA} (r = {t['estimate']:.2f}, 90% CI {t['ci90'][0]:.2f} to {t['ci90'][1]:.2f}).") if t["equivalent"] else \
                  ("Overstated", f"Within networks we did not detect an association between exact-inference cost and R "
                                 f"(r = {t['estimate']:.2f}, 90% CI {t['ci90'][0]:.2f} to {t['ci90'][1]:.2f}); an association of "
                                 f"|r| = {DELTA} cannot be excluded.")
    s22, s32 = R.get("S2.2"), R.get("S3.2")
    if s22 and s32:
        fail = [b for b, v in s22.items() if not v["equivalent"]] + \
               [f"width effect at b = {b}" for b, v in s32["width_effect_widest_minus_narrowest"].items() if not v["equivalent"]]
        fc = [b for b, v in s32["width_effect_widest_minus_narrowest"].items() if v.get("floor_or_ceiling")]
        fc_txt = (f" At b = {', '.join(fc)} the share of R < 1 is 0 or 1 at every width, so the width-effect test is uninformative "
                  "there." if fc else "")
        V["C3"] = ("Supported", f"At every magnitude, we exclude |Spearman(cost, log R)| > {DELTA} and a width effect on the share of "
                                f"R < 1 larger than {WIDTH_MARGIN:.0%}." + fc_txt) if not fail else \
                  ("Overstated", "Equivalence fails for: " + ", ".join(f"b = {x}" if not str(x).startswith("width") else str(x) for x in fail) +
                                 ". Write 'we did not detect' for those levels, and keep the equivalence statement only where it holds." + fc_txt)
    s31 = R.get("S3.1")
    if s31:
        okp = s31["intersection_union_p_jt"] < 0.05 and s31["intersection_union_p_jt_network_medians"] < 0.05
        V["C4"] = ("Supported" if okp else "Supported with caveats",
                   f"log R falls as b falls at every width (Jonckheere-Terpstra, largest one-sided p over the 8 widths "
                   f"{s31['intersection_union_p_jt']:.1g}; on network medians {s31['intersection_union_p_jt_network_medians']:.1g}).")
    s33 = R.get("S3.3")
    if s33:
        sets = {k: v for k, v in s33.items() if "slope_ci95" in v}
        good = all(v["slope_ci95"][0] <= 1 <= v["slope_ci95"][1] and v["intercept_ci95"][0] <= 0 <= v["intercept_ci95"][1] for v in sets.values())
        V["C5"] = ("Supported" if good else "Supported with caveats",
                   "; ".join(f"{k}: slope {v['calibration_slope']:.2f} [{v['slope_ci95'][0]:.2f}, {v['slope_ci95'][1]:.2f}], "
                             f"{v['share_error_above_bar']:.0%} of instances above the 0.15 error bar" for k, v in sets.items()) +
                   ". Kappa summaries are conditional on estimator convergence (S3.4).")
    s24 = R.get("S2.4")
    if s24:
        deg = [k for k, v in s24.items() if v["degenerate"]]
        noneq = [k for k, v in s24.items() if not v["equivalent"] and not v["degenerate"]]
        V["C6"] = ("Supported with caveats" if not noneq else "Overstated",
                   ("Control (a) is degenerate: the permuted-label network predicts a constant, so its AUROC is 0.5 by "
                    "construction and is not evidence. " if deg else "") +
                   ("Control (b) is equivalent to 0.5 within ±0.02." if not noneq else f"Not equivalent within ±0.02: {', '.join(noneq)}."))
    s41, s44 = R.get("S4.1"), R.get("S4.4")
    if s41:
        sig = [k for k, v in s41.items() if v["ci95"][0] > 0 or v["ci95"][1] < 0]
        age = s44["differences"].get("age: >=65 - <65") if s44 else None
        V["C7"] = ("Supported with caveats", f"{len(sig)} of {len(s41)} network-minus-LR differences have a CI excluding 0. " +
                   (f"Age gap (>=65 minus <65): {age['difference']:.3f} [{age['ci95'][0]:.3f}, {age['ci95'][1]:.3f}]." if age else ""))
    s42 = R.get("S4.2")
    if s42:
        a = s42["all"]
        good = a["ci95"][0] > 0 and a["first_m_where_ci_excludes_0_network_better"] is not None
        V["C8"] = ("Supported" if good else "Overstated",
                   f"The network-minus-LR difference grows with m (slope {a['slope_of_difference_per_unit_m']:.3f} per unit m, CI "
                   f"{a['ci95'][0]:.3f} to {a['ci95'][1]:.3f}); the network is better with a CI excluding 0 from m = "
                   f"{a['first_m_where_ci_excludes_0_network_better']}." if good else
                   "The crossover is not established by an interaction with a CI excluding 0; report the point estimates as such.")
    s43 = R.get("S4.3")
    if s43:
        m = s43["medians"]
        V["C9"] = ("Overstated" if s43["kappa_artifact"] else "Supported with caveats",
                   f"Observed agreement {m['observed_agreement']['median']:.2f}, Fleiss kappa {m['fleiss_kappa']['median']:.3f}, Gwet "
                   f"AC1 {m['gwet_ac1']['median']:.2f}, PABAK {m['pabak']['median']:.2f} (medians over 39 items). " +
                   ("Kappa is deflated by the dominant category; write that agreement is low after adjusting for chance, and "
                    "report AC1 alongside." if s43["kappa_artifact"] else "AC1 confirms low agreement; the kappa reading stands."))
    return V


CLAIMS = {"C1": "R is governed by the magnitude of the queried posterior alone (Spearman 0.95 pooled, 0.92 within networks), "
                "with D_T and a_E barely mattering.",
          "C2": "Exact-inference cost does not predict R within networks (r = -0.02, CI -0.09 to 0.05).",
          "C3": "Within-magnitude |rho(log R, log cost)| of 0.03-0.18 across widths 2-16 (no width effect at any magnitude).",
          "C4": "The share of R < 1 rises as b falls, at every width.",
          "C5": "The kappa map reproduces the 2x gap from instance location alone.",
          "C6": "Negative controls: permuted-label and shuffled-finding AUROC 0.500.",
          "C7": "Clinical AUROC comparisons (missing-data table, subgroups, age >= 65 gap 0.724 vs 0.799).",
          "C8": "The network overtakes LR from m = 0.3 and degrades more slowly.",
          "C9": "Median Fleiss kappa 0.017: the models do not agree."}


def forest(d: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        SKIPPED.append("figures (matplotlib not installed)")
        return
    panels = [("Correlations and indices", d[d.id.isin(["S1.1", "S1.2", "S2.1", "S2.2"])]),
              ("AUROC differences", d[d.id.isin(["S4.1", "S4.4"]) & d.test.str.contains("AUROC")]),
              ("Agreement and calibration", d[d.id.isin(["S4.3", "S3.3"])])]
    panels = [(t, p.dropna(subset=["ci_low", "ci_high"])) for t, p in panels]
    panels = [(t, p) for t, p in panels if len(p)]
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 0.32 * sum(len(p) for _, p in panels) + 1.6 * len(panels)))
    axes = np.atleast_1d(axes)
    for ax, (title, p) in zip(axes, panels):
        y = np.arange(len(p))[::-1]
        ax.errorbar(p.effect, y, xerr=[p.effect - p.ci_low, p.ci_high - p.effect], fmt="o", ms=3.5, color="#1f5f8b", ecolor="#7fa7c4", capsize=2)
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([t[:70] for t in p.test], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "forest_plot.png", dpi=200)
    plt.close(fig)


def scatter(per):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    p = per.reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(p.nodes, p.rho, s=20 + 2 * p.cost_sd ** 2, color="#1f5f8b")
    for _, r in p.iterrows():
        ax.annotate(r.network, (r.nodes, r.rho), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("network nodes")
    ax.set_ylabel("per-network Spearman(cost, log R)")
    ax.set_title("S2.3: per-network correlation against size (marker area ~ cost variance)", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "network_size_scatter.png", dpi=200)
    plt.close(fig)


def write_report(d: pd.DataFrame):
    V = verdicts()
    L = ["# Claim audit: quantum cost paper and clinical annex", "",
         "Lens: computational-methods paper (cost-ratio measurement claims) with a clinical-epidemiology annex. Reporting "
         "standards: methods/resource reporting for the quantum claims, TRIPOD for the clinical prediction claims. Plan: "
         "`tasks/stat_audit_plan.md` (tag `stat-audit-plan-frozen`). Verdicts below are generated by rules written into "
         "`scripts/audit.py` before the run; read them as a first draft.", "",
         f"With 13,477 benchmark instances and about {R.get('S0.1', {}).get('instances', 0):,} clinical instances nearly every "
         "test is 'significant'; effect sizes, equivalence margins and intervals carry the argument, not p-values.", "",
         "## Stage 0", "", f"- **0.1 Zero-width intervals:** {R.get('S0.1', {}).get('verdict', 'not run')}"]
    if "S0.1" in R:
        s = R["S0.1"]
        L.append(f"  Distribution of the W6 ratio: median {s['distribution']['median_ratio']:.3f}, IQR "
                 f"{s['distribution']['iqr_ratio'][0]:.3f}-{s['distribution']['iqr_ratio'][1]:.3f}, 5th-95th "
                 f"{s['distribution']['p5_p95_ratio'][0]:.3f}-{s['distribution']['p5_p95_ratio'][1]:.3f}, share below 1 "
                 f"{s['distribution']['share_below_1']:.1%} (patient-bootstrap CI {s['patient_bootstrap']['share_below_1_ci95'][0]:.1%}-"
                 f"{s['patient_bootstrap']['share_below_1_ci95'][1]:.1%}).")
    if "S0.2" in R:
        L.append("- **0.2 Computed vs measured R:** " + "; ".join(
            f"{k}: " + ", ".join(f"{a} {b}" for a, b in v.items() if a != "note") for k, v in R["S0.2"].items()))
    L += ["", "## Summary", "", "| Claim | Verdict | Suggested wording |", "|---|---|---|"]
    for c, txt in CLAIMS.items():
        v = V.get(c, ("Not run", "-"))
        L.append(f"| **{c}** {txt} | {v[0]} | {v[1]} |")
    L += ["", "## Per-claim detail", ""]
    sections = {"C1": ["S1.1", "S1.2", "S1.3"], "C2": ["S2.1", "S2.3"], "C3": ["S2.2", "S3.2"], "C4": ["S3.1"], "C5": ["S3.3", "S3.4"],
                "C6": ["S2.4"], "C7": ["S4.1", "S4.4"], "C8": ["S4.2"], "C9": ["S4.3"]}
    for c, ids in sections.items():
        L += [f"### {c}", "", f"Claim: {CLAIMS[c]}", ""]
        rows = d[d.id.isin(ids)]
        if len(rows):
            L += ["| Test | Family | Effect | CI | p (raw) | p (Holm) | Note |", "|---|---|---|---|---|---|---|"]
            for _, r in rows.iterrows():
                ci = f"[{r.ci_low:.3g}, {r.ci_high:.3g}]" if np.isfinite(r.ci_low) else "-"
                pr = ("<1e-300" if r.p_raw == 0 else f"{r.p_raw:.2g}") if np.isfinite(r.p_raw) else "-"
                pa = f"{r.p_adj_holm:.2g}" if np.isfinite(r.p_adj_holm) else "-"
                L.append(f"| {r.test} | {r.family} | {r.effect:.4g} | {ci} | {pr} | {pa} | {r.note} |")
        v = V.get(c)
        L += ["", f"**Verdict:** {v[0]}. {v[1]}" if v else "**Verdict:** not run.", ""]
    L += ["## Multiplicity and test log", "",
          f"{len(d)} tests are logged in `test_log.csv`. Holm adjustment within the primary family "
          f"({int((d.family == 'primary').sum())} tests) and within the subgroup family "
          f"({int((d.family == 'secondary-subgroup').sum())}); the sensitivity family ({int((d.family == 'sensitivity').sum())}) is "
          "reported unadjusted, as the plan states. Descriptive indices (Sobol shares, ICC, slope spread) carry no p-value.", "",
          "## Analyses skipped or replaced", ""]
    L += [f"- {s}" for s in SKIPPED] or ["- none"]
    L += ["", "## What would strengthen this further", "",
          "- More measured instances (emulated qLW and sampled LW on the same instances), so C1 rests on measurement rather "
          "than on the cost law.",
          "- Estimator calibration that reports failures as censored observations instead of dropping them (S3.4).",
          "- A second clinical site for the clinical annex (TRIPOD external validation)."]
    (OUT / "claim_audit_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def requirements_used():
    import importlib.metadata as md
    pk = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pyarrow", "PyYAML", "statsmodels"]
    lines = []
    for p in pk:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            lines.append(f"# {p} not installed")
    (OUT / "requirements_used.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small resample counts, to check the run end to end")
    ap.add_argument("--only", choices=["quantum", "clinical"])
    a = ap.parse_args()
    N.update(sobol=1000, sobol_boot=200, sobol_boot_reps=20, w6_sub=50000, boot=1000, boot_small=200, boot_patient=2000, boot_agree=200)
    if a.quick:
        N.update(sobol=20, sobol_boot=10, sobol_boot_reps=3, w6_sub=10000, boot=50, boot_small=30, boot_patient=40, boot_agree=20)
    OUT.mkdir(parents=True, exist_ok=True)
    R["resamples"], R["statsmodels"] = dict(N), HAVE_SM
    t0 = time.perf_counter()
    if not HAVE_SM:
        say("statsmodels not installed: mixed models replaced by the plan's fallbacks (see the report)")
    stages = [] if a.only == "clinical" else ["quantum"]
    stages += [] if a.only == "quantum" else ["clinical"]
    W6 = k6 = None
    if "quantum" in stages:
        say("== Stage 0")
        W6, _, k6 = stage0_zero_width()
        ok, BM = bench_frame()
        lwc, c3, w8 = stage0_partition(ok)
        say("== Stage 1")
        stage1(ok, W6, k6)
        stage1_measured(ok, lwc, c3, w8)
        stage1_clustering(ok)
        say("== Stage 2")
        w5, per = stage2(ok, BM)
        scatter(per)
        say("== Stage 3")
        stage3(w5, BM)
        stage3_kappa()
    if "clinical" in stages:
        say("== Stage 4 (clinical; local data, aggregates only)")
        stage4_agreement()
        S, Y, g, dx, groups = stage4_predictions()
        stage4(S, Y, g, dx, groups)
    d = test_log_frame()
    d.to_csv(OUT / "test_log.csv", index=False)
    forest(d)
    write_report(d)
    requirements_used()
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    say(f"\ndone in {(time.perf_counter() - t0) / 60:.0f} min -> {OUT}/ (claim_audit_report.md, test_log.csv, forest_plot.png)")


if __name__ == "__main__":
    main()
