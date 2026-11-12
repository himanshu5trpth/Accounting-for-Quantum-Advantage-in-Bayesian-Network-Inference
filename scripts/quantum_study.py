"""Quantum study, v2.1 (plan: tasks/plan_quantum_study.md). Reads frozen v2.1 results; writes reports/v2.1/* and
artifacts/v2.1/results/*. Aggregates only.

  python scripts/quantum_study.py baseline     # step 2: evaluations vs P(evidence), three opponents
  python scripts/quantum_study.py lwmech       # step 3: why likelihood weighting gets stronger as evidence gets rarer
  python scripts/quantum_study.py frontier     # step 4: precision frontier (oracle calls and CX)
  python scripts/quantum_study.py prevalence   # E-U1 Part A (tasks/unified_experiments.md): crossover in the target posterior
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES, REP = Path("artifacts/v2.1/results"), Path("reports/v2.1")
KB, VOC = Path("artifacts/v2.1/kb"), Path("configs/vocab_v2.1.json")
SEED = 2026
C_REJ, C_LW, C_AE, INK, MUTED = "#eb6834", "#1baf7a", "#2a78d6", "#1f1f1e", "#6b6a64"  # validated categorical slots 1-3


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ------------------------------------------------------------------------------------- step 2
def baseline():
    a = json.loads((RES / "advantage.json").read_text())
    d = pd.DataFrame(a["profiles"])
    cap = float(a["max_samples"])
    d["rejection_censored"] = d.rejection.isna()
    d.to_csv(REP / "baseline_choice.csv", index=False)
    x = 1 / d.p_evidence
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fits = {}
    for col, c, lab in (("rejection", C_REJ, "Rejection sampling"), ("lw", C_LW, "Likelihood weighting"), ("iae", C_AE, "Amplitude estimation (IAE)")):
        ok = d[col].notna()
        ax.scatter(x[ok], d.loc[ok, col], s=22, color=c, edgecolors="white", linewidths=0.6, zorder=3, label=lab)
        b, a0 = np.polyfit(np.log10(x[ok]), np.log10(d.loc[ok, col].astype(float)), 1)
        fits[col] = (b, a0)
        xs = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
        ax.plot(xs, 10 ** (a0 + b * np.log10(xs)), color=c, lw=2, alpha=0.9, zorder=2)
        ax.annotate(f"slope {b:.2f}", (xs[-1], 10 ** (a0 + b * np.log10(xs[-1]))), xytext=(4, 0), textcoords="offset points",
                    color=INK, fontsize=8, va="center")
    cen = d.rejection_censored
    if cen.any():  # did not reach 10% error within the sample cap: plotted at the cap, marked, not dropped
        ax.scatter(x[cen], np.full(cen.sum(), cap), marker="^", s=46, facecolors="none", edgecolors=C_REJ, linewidths=1.4, zorder=4,
                   label=f"Rejection, not reached by {cap:,.0f} samples")
    cross = a["crossover_p_evidence"]
    ax.axvline(1 / cross, color=MUTED, lw=1, ls="--", zorder=1)
    from matplotlib.transforms import blended_transform_factory
    ax.text(1 / cross, 0.66, f"rejection cheaper ← → AE cheaper\nthan AE         than rejection\n(crossover P(findings) = {cross:.2f})", color=INK, fontsize=8, ha="center",
            va="top", transform=blended_transform_factory(ax.transData, ax.transAxes))
    rare = d.p_evidence <= d.p_evidence.quantile(0.1)
    lw_vs_ae = float((d.loc[rare, "iae"] / d.loc[rare, "lw"]).median())
    ax.text(0.56, 0.98, f"Rarest 10% of patients:\nAE {a['rarest_decile_ratio']:.0f}× cheaper than rejection,\n"
                        f"but {lw_vs_ae:.0f}× costlier than likelihood weighting", transform=ax.transAxes, ha="center", va="top",
            fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="#f4f3ef", ec="none"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rarity of the patient's findings, 1 / P(findings)  →  rarer")
    ax.set_ylabel("Model evaluations to reach 10% relative error")
    ax.set_title("The verdict flips with the opponent", loc="left", fontsize=11, color=INK)
    ax.grid(alpha=0.25, which="major")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(REP / "baseline_choice.png", dpi=180)
    out = {"slopes_refit": {k: v[0] for k, v in fits.items()}, "slopes_stored": {k: v["slope"] for k, v in a["slopes"].items()},
           "crossover_p_evidence": cross, "rarest_decile_ae_vs_rejection": a["rarest_decile_ratio"],
           "rarest_decile_ae_vs_lw": lw_vs_ae, "censored_rejection": int(cen.sum()), "profiles": int(len(d))}
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- step 3
def _bn_load():
    from qddx.bn import BN
    return BN.from_dict(json.loads((KB / "bn.json").read_text()))


def _test_split():
    P = json.loads(VOC.read_text())["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    ev = [k for k, p in P.items() if p["kind"] != "diagnosis"]
    df = pd.read_parquet(KB / "cohort.parquet")
    subs = np.unique(df.subject_id.to_numpy())
    train = set(np.random.default_rng(SEED).permutation(subs)[: int(round(0.8 * len(subs)))].tolist())  # = kb.split_mask
    return df[~df.subject_id.isin(train)].reset_index(drop=True), dx, ev


def lw_exact(bn, E, t):
    """Exact likelihood-weighting quantities for evidence E and target t (no sampling):
    P(E); p = P(t|E); weight dispersion 1 + CV^2 = E_q[w^2] / E_q[w]^2 (E_q[w^2] = VE with each finding's CPT entry
    squared); asymptotic samples for 90% of runs within 10% of p (validated against real sampling: 0.9-1.5x)."""
    from qddx.bn import BN, marginal, posterior_exact
    post, pe, _ = posterior_exact(bn, E, [t])
    p = post[t]
    sq = BN(bn.struct, [c.copy() for c in bn.cpts])
    for k, v in E.hard.items():
        j = bn.struct.index(k)
        sq.cpts[j][:, v] = bn.cpts[j][:, v] ** 2
    a0, a1 = marginal(sq, E, [bn.struct.index(t)])
    disp = (a0 + a1) / pe ** 2
    var = ((1 - p) ** 2 * a1 + p ** 2 * a0) / pe ** 2
    return {"p_evidence": float(pe), "p": float(p), "dispersion": float(disp), "samples": float((1.645 / 0.1) ** 2 * var / p ** 2)}


def stratified_patients(bn, test, dx, ev):
    """Held-out admissions with their full findings, up to 40 per two-decade bin of P(findings) (from 20,000 drawn
    with a fixed seed) -> [(Evidence, most probable diagnosis)]."""
    from qddx.bn import Evidence, posterior_exact
    rows = []
    for _, r in test.sample(n=20000, random_state=SEED).iterrows():
        E = Evidence({k: int(r[k]) for k in ev if r[k] >= 0}, {})
        post, pe, _ = posterior_exact(bn, E, dx)
        rows.append((E, max(post, key=post.get), math.log10(pe)))
    bins = np.arange(-16, 0.1, 2)
    pick = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        pick += [i for i, (_, _, l) in enumerate(rows) if lo < l <= hi][:40]
    pick += [i for i, (_, _, l) in enumerate(rows) if l <= bins[0]][:40]
    return [rows[i][:2] for i in pick]


def lwmech():
    from qddx.bn import BN, Evidence, Struct, posterior_exact, posterior_lw
    bn = _bn_load()
    test, dx, ev = _test_split()
    rec = []
    for n, (E, t) in enumerate(stratified_patients(bn, test, dx, ev)):
        x = lw_exact(bn, E, t)
        if n % 4 == 0:  # sampling check on a quarter of them: ESS fraction at 20k samples vs exact 1 / dispersion
            _, _, c = posterior_lw(bn, E, [t], 20000, SEED + n)
            x["ess_fraction_sampled"] = c["effective_sample_size"] / 20000
        x.update(target=t, findings=len(E.hard))
        rec.append(x)
    d = pd.DataFrame(rec)
    d["ess_fraction_exact"] = 1 / d.dispersion
    y = np.log10(d.samples)
    X = {"log10 1/P(E)": -np.log10(d.p_evidence), "log10 weight dispersion": np.log10(d.dispersion),
         "log10 (1-p)/p": np.log10((1 - d.p) / d.p)}

    def r2(cols):
        A = np.column_stack([np.ones(len(d))] + [X[c] for c in cols])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        return float(1 - res.var() / y.var()), dict(zip(["intercept"] + list(cols), map(float, beta)))
    reg = {"+".join(c) if c else "none": {"r2": r2(c)[0], "coef": r2(c)[1]} for c in
           ([("log10 1/P(E)",), ("log10 weight dispersion",), ("log10 (1-p)/p",),
             ("log10 1/P(E)", "log10 weight dispersion"), ("log10 weight dispersion", "log10 (1-p)/p"),
             ("log10 1/P(E)", "log10 weight dispersion", "log10 (1-p)/p")])}
    chk = d.dropna(subset=["ess_fraction_sampled"])
    ess_corr = float(np.corrcoef(np.log10(chk.ess_fraction_sampled), np.log10(chk.ess_fraction_exact))[0, 1])

    # where it breaks (1): textbook chain with evidence on the leaf, near-deterministic links (sampling order root -> leaf)
    chain = []
    for prior in (0.1, 1e-2, 1e-3, 1e-4, 1e-5):
        L = 6
        cpts = [np.array([[1 - prior, prior]])] + [np.array([[0.999, 0.001], [0.001, 0.999]]) for _ in range(L - 1)]
        b = BN(Struct([f"x{j}" for j in range(L)], [()] + [(j,) for j in range(L - 1)]), cpts)
        x = lw_exact(b, Evidence({f"x{L - 1}": 1}, {}), "x0")
        _, _, c = posterior_lw(b, Evidence({f"x{L - 1}": 1}, {}), ["x0"], 100000, SEED)
        chain.append({**x, "root_prior": prior, "ess_fraction_sampled": c["effective_sample_size"] / 100000,
                      "rejection_acceptance": x["p_evidence"]})
    # where it breaks (2): an atypical admission on the real network: every lab abnormal, no history, no drugs
    P = json.loads(VOC.read_text())["propositions"]
    atyp = Evidence({k: (1 if P[k]["kind"] == "lab" else 0) for k in ev if P[k]["kind"] != "demographic"} | {"age_ge_65": 0}, {})
    post, _, _ = posterior_exact(bn, atyp, dx)
    xa = lw_exact(bn, atyp, max(post, key=post.get))
    xa["target"] = max(post, key=post.get)
    _, _, c = posterior_lw(bn, atyp, [xa["target"]], 100000, SEED)
    xa["ess_fraction_sampled"] = c["effective_sample_size"] / 100000
    out = {"patients": d.to_dict("records"), "regression_log10_samples": reg, "ess_sampled_vs_exact_corr": ess_corr,
           "chain_failure": chain, "atypical_admission": xa,
           "summary": {"n": int(len(d)), "samples_median": float(d.samples.median()), "samples_max": float(d.samples.max()),
                       "dispersion_median": float(d.dispersion.median()), "dispersion_max": float(d.dispersion.max())}}
    (RES / "lw_mechanism.json").write_text(json.dumps(out, indent=1))
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axs[0]
    xdiff = d.dispersion * (1 - d.p) / d.p  # target odds x weight dispersion
    sc = ax.scatter(xdiff, d.samples, c=-np.log10(d.p_evidence), cmap="Blues", vmin=0, vmax=16, s=18, edgecolors=MUTED, linewidths=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("rarity, log10 1/P(findings)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("target odds × weight dispersion,  (1−p)/p × E[w²]/E[w]²")
    ax.set_ylabel("samples for 10% relative error")
    best = reg["log10 weight dispersion+log10 (1-p)/p"]["r2"]
    only_pe = reg["log10 1/P(E)"]["r2"]
    ax.set_title(f"Real patients: cost follows target odds × weight dispersion, not rarity\n(R² {best:.2f}; rarity alone "
                 f"{only_pe:.2f}, and rarer patients need fewer samples)", loc="left", fontsize=9.5, color=INK)
    ax.grid(alpha=0.25)
    ax = axs[1]
    ch = pd.DataFrame(chain)
    ax.plot(ch.root_prior, ch.samples, "o-", color=C_REJ, lw=2, ms=6, label="chain of 6, evidence on the leaf")
    for _, r in ch.iterrows():
        ax.annotate(f"ESS {100 * r.ess_fraction_sampled:.1f}%", (r.root_prior, r.samples), xytext=(6, -10), textcoords="offset points",
                    fontsize=7, color=MUTED)
    ax.axhspan(d.samples.min(), d.samples.max(), color=C_LW, alpha=0.12, lw=0, label=f"{len(d)} real patients (range)")
    ax.axhline(xa["samples"], color=C_AE, lw=1.5, ls="--", label=f"atypical real admission: {xa['samples']:,.0f} samples")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("prior of the chain's root cause  →  rarer explanation")
    ax.set_ylabel("samples for 10% relative error")
    ax.set_title("Where it breaks: evidence at the end of a near-\ndeterministic chain; samples explode as the\nexplanation becomes rare (ESS falls to ~0.6%)",
                 loc="left", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REP / "lw_mechanism.png", dpi=180)
    print(json.dumps({k: v for k, v in out.items() if k != "patients"}, indent=1))


# ------------------------------------------------------------------------------------- step 4
EPS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
RATIO_INTERIM = (3.66, 4.61)  # measured Grover-step CX / (2 x state-preparation CX), resources.json; refreshed by step 5


def node_max(bn, j, e, obs):
    """max over finding j's unobserved parent settings of P(x_j = e | parents), observed parents fixed."""
    pa = bn.struct.parents[j]
    best = 0.0
    for r in range(2 ** len(pa)):
        bits = [(r >> (len(pa) - 1 - i)) & 1 for i in range(len(pa))]
        if all(obs.get(p, b) == b for p, b in zip(pa, bits)):
            q = bn.cpts[j][r, 1]
            best = max(best, q if e == 1 else 1 - q)
    return best


def frontier():
    """Precision at which amplitude estimation would overtake likelihood weighting, per patient. Analytic cost laws:
    LW(eps) = N_lw(0.1) (0.1/eps)^2 (exact asymptotic, validated 0.9-1.5x against sampling); AE(eps) = kappa pi / eps
    (1/sqrt(a_E) + 1/sqrt(a_T)) model evaluations, kappa = 1 for an ideal (Heisenberg-limited) estimator and fitted to
    tuned IAE (16 shots, alpha 0.2; emulation checked kappa 5.8-7.9 for eps 0.1-0.02, i.e. 1/eps holds). Since LW goes
    as eps^-2 and AE as eps^-1, the crossover is eps* = 0.1 N_lw(0.1) / AE(0.1); AE wins for eps < eps*."""
    ratio = RATIO_INTERIM
    cm = RES / "cost_model.json"
    if cm.exists():  # step 5's measured law, if available
        c = json.loads(cm.read_text())
        ratio = tuple(c["ratio_range_v21_profiles"])
    rows = [json.loads(x) for x in Path("artifacts/lw_check/rows.jsonl").read_text().splitlines()]
    tuned = [r for r in rows if r.get("kb") == str(KB) and r.get("shots") == 16 and r.get("alpha") == 0.2 and r.get("qlw")]
    ks = []
    for r in tuned:
        aE = 10 ** (r["log10_pe"] - r["log10_W"])
        ks.append(r["qlw"] * 0.1 / (math.pi * (1 / math.sqrt(aE) + 1 / math.sqrt(aE * r["p"]))))
    kappa = float(np.median(ks))
    bn = _bn_load()
    test, dx, ev = _test_split()
    rec = []
    for E, t in stratified_patients(bn, test, dx, ev):
        x = lw_exact(bn, E, t)
        obs = {bn.struct.index(k): v for k, v in E.hard.items()}
        logW = sum(math.log(node_max(bn, j, v, obs)) for j, v in obs.items())
        aE_q, aE = x["p_evidence"] / math.exp(logW), x["p_evidence"]
        cost = lambda a, k: k * math.pi / 0.1 * (1 / math.sqrt(a) + 1 / math.sqrt(a * x["p"]))
        q = {"qlw_ideal": cost(aE_q, 1.0), "qlw_tuned": cost(aE_q, kappa), "ae_tuned": cost(aE, kappa)}
        r = {"p_evidence": aE, "p": x["p"], "lw_samples": x["samples"], **{f"cost_{k}": v for k, v in q.items()}}
        for k, v in q.items():
            r[f"eps_star_{k}"] = 0.1 * x["samples"] / v
        for lo_hi, rr in zip(("lo", "hi"), ratio):  # gate units: the Grover part costs `ratio` times more per call
            r[f"eps_star_qlw_tuned_cx_{lo_hi}"] = 0.1 * x["samples"] / (q["qlw_tuned"] * rr)
            r[f"eps_star_qlw_ideal_cx_{lo_hi}"] = 0.1 * x["samples"] / (q["qlw_ideal"] * rr)
        rec.append(r)
    d = pd.DataFrame(rec)
    d["decile"] = pd.qcut(-np.log10(d.p_evidence), 10, labels=False)  # 0 = commonest, 9 = rarest
    keys = [c for c in d.columns if c.startswith("eps_star_")]
    dec = d.groupby("decile").agg(p_evidence_median=("p_evidence", "median"), **{k: (k, "median") for k in keys}).reset_index()
    wins = {k: {f"{e:g}": float((d[k] >= e).mean()) for e in EPS} for k in keys}  # share of patients where AE wins at eps
    gap = {k: float(np.log10(0.1 / d[k].median())) for k in keys}  # orders of magnitude below the 10% operating point
    out = {"kappa_tuned_iae": kappa, "kappa_n": len(ks), "kappa_iqr": [float(np.quantile(ks, 0.25)), float(np.quantile(ks, 0.75))],
           "ratio_used": ratio, "patients": int(len(d)), "eps_grid": EPS, "deciles": dec.to_dict("records"),
           "share_patients_where_ae_wins": wins, "orders_of_magnitude_below_operating_point_median": gap,
           "rows": d.drop(columns="decile").to_dict("records")}
    (RES / "frontier.json").write_text(json.dumps(out, indent=1))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    xs = dec.p_evidence_median
    band_lo, band_hi = dec.eps_star_qlw_tuned_cx_hi, dec.eps_star_qlw_tuned_cx_lo
    ax.fill_between(xs, 1e-6, band_hi, color=C_AE, alpha=0.10, lw=0)
    ax.fill_between(xs, band_lo, band_hi, color=C_AE, alpha=0.25, lw=0, label=f"tuned IAE, gate count (ratio {ratio[0]:.2f}–{ratio[1]:.2f})")
    ax.plot(xs, dec.eps_star_qlw_tuned, color=C_AE, lw=2, label="tuned IAE, model evaluations")
    ax.plot(xs, dec.eps_star_qlw_ideal, color=C_AE, lw=1.5, ls="--", label="ideal estimator, model evaluations")
    ax.scatter(d.p_evidence, d.eps_star_qlw_tuned, s=8, color=C_AE, alpha=0.35, lw=0)
    ax.plot(xs, dec.eps_star_qlw_ideal_cx_lo, color=C_AE, lw=1, ls=":", label="ideal estimator, gate count")
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    ax.axhline(0.1, color=C_REJ, lw=1.5)
    ax.text(0.99, 0.118, "operating point: 10% relative error on the posterior", color=INK, fontsize=8, transform=tr, ha="right")
    ax.axhline(0.01, color=MUTED, lw=1, ls=":")
    ax.text(0.99, 0.0118, "1% relative error", color=MUTED, fontsize=8, transform=tr, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.invert_xaxis()
    ax.set_xlabel("P(findings)  →  rarer patients")
    ax.set_ylabel("relative error ε (smaller = more precise)")
    g = gap["eps_star_qlw_tuned_cx_lo"]
    ax.set_title("Quantum likelihood weighting beats likelihood weighting only below each curve", loc="left", fontsize=10, color=INK)
    gi = gap["eps_star_qlw_ideal_cx_lo"]
    fig.text(0.01, 0.01, f"Tuned IAE, gate count: the crossover is a median {g:.1f} orders of magnitude finer than the 10% operating point,\n"
             f"which is itself finer than any change in a diagnosis decision. Ideal (Heisenberg-limited) estimator, gate count: median {gi:.1f} orders\n"
             f"finer; it reaches 10% only for the commonest patients ({100 * wins['eps_star_qlw_ideal_cx_lo']['0.1']:.0f}%). No real estimator reaches that bound.",
             fontsize=7.5, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(REP / "frontier.png", dpi=180)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


# ------------------------------------------------------------------------------------- E-U1 Part A
FLOOR_CASES, TRAIN_ADM = 500, 437_808


def _train_base_rates():
    P = json.loads(VOC.read_text())["propositions"]
    dx = [k for k, v in P.items() if v["kind"] == "diagnosis"]
    df = pd.read_parquet(KB / "cohort.parquet", columns=["subject_id", *dx])
    subs = np.unique(df.subject_id.to_numpy())
    train = set(np.random.default_rng(SEED).permutation(subs)[: int(round(0.8 * len(subs)))].tolist())  # kb.split_mask
    tr = df.subject_id.isin(train).to_numpy()
    assert tr.sum() == TRAIN_ADM, tr.sum()
    return {d: float(df.loc[tr, d].mean()) for d in dx}


def prevalence():
    """Sweep the target posterior p from 0.5 to 1e-5 holding each real patient's LW dispersion D and evidence amplitude
    a_E fixed (backed out of frontier.json: LW(p) = D (1-p)/p at eps 0.1; qLW(p) = kappa pi/eps (1 + p^-1/2) / sqrt(a_E)).
    Relative metric: eps = 0.1. Decision metric: error 0.1 max(p, t), i.e. eps_eff = 0.1 max(p, t) / p.
    AE cost is multiplied by the gate ratio (1 = model evaluations; V0, V1 ranges from oracle_v2.json)."""
    fr = json.loads((RES / "frontier.json").read_text())
    kappa = fr["kappa_tuned_iae"]
    ov = json.loads((RES / "oracle_v2.json").read_text())["summary"]
    r = pd.DataFrame(fr["rows"])
    p0 = r.p.to_numpy()
    D = r.lw_samples.to_numpy() * p0 / (1 - p0)
    S = r.cost_qlw_ideal.to_numpy() * 0.1 / math.pi / (1 + p0 ** -0.5)  # 1 / sqrt(a_E)
    grid = np.logspace(math.log10(0.5), -5, 241)

    def costs(p, d, s, k, g, t):
        eps = 0.1 if t is None else 0.1 * np.maximum(p, t) / p
        lw = d * (1 - p) / p * (0.1 / eps) ** 2
        q = g * k * math.pi / eps * s * (1 + p ** -0.5)
        return lw, q

    # check: at each patient's own p the model reproduces step 4's crossover
    lw0, q0 = costs(p0, D, S, 1.0, 1.0, None)
    chk = float(np.max(np.abs(0.1 * lw0 / q0 / r.eps_star_qlw_ideal.to_numpy() - 1)))
    floor = FLOOR_CASES / TRAIN_ADM
    units = {"model_evaluations": (1.0, 1.0), "v0_gates": tuple(ov["v0"]["v21_ratio_range"]), "v1_gates": tuple(ov["v1"]["v21_ratio_range"])}
    out = {"check_reproduces_step4_max_rel_error": chk, "kappa_tuned": kappa, "floor_prevalence_500_cases": floor,
           "floor_sensitivity": {"200": 200 / TRAIN_ADM, "1000": 1000 / TRAIN_ADM}, "patients": int(len(r)),
           "grid": [float(grid[0]), float(grid[-1])], "settings": {}}
    for metric, t in (("relative", None), ("decision_t0.01", 0.01), ("decision_t0.001", 0.001)):
        for kname, k in (("ideal", 1.0), ("tuned", kappa)):
            for uname, (glo, ghi) in units.items():
                res = {}
                for end, g in (("lo", glo), ("hi", ghi)) if glo != ghi else (("", glo),):
                    pstar, share = [], {}
                    win = np.zeros((len(r), len(grid)), bool)
                    for i in range(len(r)):
                        lw, q = costs(grid, D[i], S[i], k, g, t)
                        win[i] = q <= lw
                        pstar.append(float(grid[win[i]].max()) if win[i].any() else 0.0)
                    pstar = np.array(pstar)
                    for pv in (0.1, 0.01, floor, 1e-4, 1e-5):
                        j = int(np.argmin(np.abs(np.log(grid) - math.log(pv))))
                        share[f"{pv:.3g}"] = float(win[:, j].mean())
                    res[end or "value"] = {"p_star_median": float(np.median(pstar)), "p_star_q10_q90": [float(np.quantile(pstar, .1)), float(np.quantile(pstar, .9))],
                                           "share_never_wins": float((pstar == 0).mean()), "share_ae_wins_at_p": share}
                out["settings"][f"{metric}|{kname}|{uname}"] = res
    # the plan's table: a_E at its 10th / 50th / 90th percentile, D at its median (worst-to-best for AE as a_E grows)
    tab = {}
    for qa in (0.1, 0.5, 0.9):
        s_q, d_m = float(np.quantile(S, 1 - qa)), float(np.median(D))
        for metric, t in (("relative", None), ("decision_t0.01", 0.01)):
            for kname, k in (("ideal", 1.0), ("tuned", kappa)):
                for uname, (glo, ghi) in units.items():
                    g = (glo + ghi) / 2
                    lw, q = costs(grid, d_m, s_q, k, g, t)
                    w = q <= lw
                    tab[f"aE_q{int(qa * 100)}|{metric}|{kname}|{uname}"] = float(grid[w].max()) if w.any() else 0.0
    out["p_star_by_aE_percentile"] = tab
    out["a_E_q10_q50_q90"] = [float(x) for x in np.quantile(1 / S ** 2, [.1, .5, .9])]
    out["D_median"] = float(np.median(D))
    rates = _train_base_rates()
    out["six_diagnoses_training_prevalence"] = rates
    (RES / "prevalence_crossover.json").write_text(json.dumps(out, indent=1))

    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    d_m, s_m = float(np.median(D)), float(np.median(S))
    g1 = units["v1_gates"]
    for ax, (metric, t, title) in zip(axs, (("relative", None, "10% relative error on the posterior"),
                                            ("decision_t0.01", 0.01, "Decision metric: error 0.1 × max(p, 1%)"))):
        lw, _ = costs(grid, d_m, s_m, 1, 1, t)
        ax.plot(grid, lw, color=C_LW, lw=2, label="Likelihood weighting")
        qt_lo, qt_hi = costs(grid, d_m, s_m, kappa, g1[0], t)[1], costs(grid, d_m, s_m, kappa, g1[1], t)[1]
        ax.fill_between(grid, qt_lo, qt_hi, color=C_AE, alpha=0.3, lw=0)
        ax.plot(grid, np.sqrt(qt_lo * qt_hi), color=C_AE, lw=2, label=f"Quantum LW, tuned IAE (κ {kappa:.1f}), V1 gates")
        qi = costs(grid, d_m, s_m, 1, sum(g1) / 2, t)[1]
        ax.plot(grid, qi, color=C_AE, lw=1.5, ls="--", label="Quantum LW, ideal estimator (κ 1), V1 gates")
        for q, ls in ((np.sqrt(qt_lo * qt_hi), "o"), (qi, "s")):
            w = np.flatnonzero(q <= lw)
            if len(w) and w[0] > 0:
                j = w[0]
                ax.plot(grid[j], lw[j], ls, color=INK, ms=5, zorder=5)
                ax.annotate(f"p* = {grid[j]:.1g}", (grid[j], lw[j]), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK)
        for name, v in rates.items():
            ax.axvline(v, color=MUTED, lw=0.8, alpha=0.6)
        ax.axvline(floor, color=INK, lw=1, ls=":")
        ax.text(floor, 0.97, " 500-case floor", transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK, va="top")
        ax.text(min(rates.values()), 0.97, "six diagnoses' prevalence ", transform=ax.get_xaxis_transform(), fontsize=7.5,
                color=MUTED, va="top", ha="right")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Target posterior P(diagnosis | findings)  →  rarer")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(alpha=0.25, which="major")
    axs[0].set_ylabel("Cost (model evaluations; quantum in V1 gate units)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.text(0.01, 0.005, "Median patient's weight dispersion and evidence amplitude held fixed while the posterior is swept; "
             "band = V1 ratio 0.91–1.11.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(REP / "prevalence_crossover.png", dpi=180)
    print(json.dumps({k: v for k, v in out.items() if k != "settings"}, indent=1))
    for k, v in out["settings"].items():
        print(k, {e: (round(x["p_star_median"], 6), x["share_never_wins"], x["share_ae_wins_at_p"]) for e, x in v.items()})


if __name__ == "__main__":
    {"baseline": baseline, "lwmech": lwmech, "frontier": frontier, "prevalence": prevalence}[sys.argv[1]]()
