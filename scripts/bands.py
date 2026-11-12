"""E-U1 Parts B and C (plan: tasks/plan_bands.md). Local data; aggregates printed only. Patient-level outputs stay in
artifacts/bands/ (never committed).

  python scripts/bands.py cats                 # ICD-10 category per admission (ICD-9 via CMS GEMs)
  python scripts/bands.py b1                   # posterior spread of the six -> posterior_spread.json / .png
  python scripts/bands.py b2                   # candidates + band targets -> candidate_targets.json, tasks/band_targets.json
  python scripts/bands.py b3 START END         # window coverage, chunk of the 3,000-admission sample
  python scripts/bands.py b3-summary           # -> window_coverage.json
  python scripts/bands.py c1 BAND              # learn the band network (uncommon | rare), kb.learn_bn unchanged
  python scripts/bands.py structure            # parents-only export of each band network (for V1 CX counts)
  python scripts/bands.py c2 BAND              # health checks for one band; c2-summary -> band_networks.json
  python scripts/bands.py c3 BAND START END    # exact per-instance quantities by evidence pattern (chunk)
  python scripts/bands.py c3-sample BAND       # sampled LW + emulated IAE on 24 window instances (amendment A6)
  python scripts/bands.py c5                   # C4 + C5 statistics (amendment A2-A5) -> band_costs.json, band_crossover.png
  python scripts/bands.py b3top START END      # top-5 differential in the window (A7); b3top-summary
  python scripts/bands.py c1-variant           # rare band, permutation-thresholded pipeline variant (A8)
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qddx import kb
from qddx.bn import BN, Evidence, marginal, posterior_exact
from qddx.cli import load_config
from qddx.engine import platt

ART, RES, REP = Path("artifacts/bands"), Path("artifacts/v2.1/results"), Path("reports/v2.1")
KB21, VOC21, YAML = Path("artifacts/v2.1/kb"), Path("configs/vocab_v2.1.json"), "configs/qddx.yaml"
SEED, Z90, REL, WIN, T_DEC, KAPPA = 2026, 1.645, 0.1, (0.001, 0.04), 0.01, None
BANDS = {"common": None, "uncommon": (0.01, 0.05), "rare": (0.001, 0.01)}
FLOOR = 500
EXCLUDE_CH = set("RVWXYZ")
EXCLUDE_CAT = {"I50", "I48", "N39", "N10", "N30", *[f"J{i}" for i in range(12, 19)], "N17", *[f"D{i}" for i in range(50, 65)]}
ART.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------------- shared
def cfg_for(vocab: Path) -> dict:
    return load_config(YAML, [], str(vocab))


def cohort():
    """v2.1 cohort with split masks (train, inner-train, validation, test) exactly as kb.split_mask / vocab.inner_split."""
    df = pd.read_parquet(KB21 / "cohort.parquet")
    tr = kb.split_mask(df.subject_id.to_numpy(), 0.8, SEED)
    inner = np.zeros(len(df), bool)
    inner[np.where(tr)[0]] = kb.split_mask(df.subject_id.to_numpy()[tr], 0.8, SEED + 1)
    assert tr.sum() == 437_808 and inner.sum() == 350_779, (tr.sum(), inner.sum())
    return df, tr, inner, tr & ~inner


def band_net(band: str):
    """-> (bn, cfg, frame with findings + this band's targets, target names)."""
    if band == "common":
        cfg = cfg_for(VOC21)
        df, *_ = cohort()
        return BN.from_dict(json.loads((KB21 / "bn.json").read_text())), cfg, df, kb.props(cfg, "diagnosis")
    src = "rare" if band == "rare_variant" else band
    cfg = cfg_for(Path(f"configs/vocab_band_{src}.json"))
    df = frame(src, cfg)
    return BN.from_dict(json.loads((ART / band / "bn.json").read_text())), cfg, df, kb.props(cfg, "diagnosis")


def frame(band: str, cfg: dict) -> pd.DataFrame:
    df, *_ = cohort()
    six = kb.props(cfg_for(VOC21), "diagnosis")
    names = kb.props(cfg, "diagnosis")
    t = pd.read_parquet(ART / band / "targets.parquet").set_index("hadm_id")[names].reindex(df.hadm_id).to_numpy(np.int8)
    df = df.drop(columns=six)
    for i, n in enumerate(names):
        df[n] = t[:, i]
    return df


def cluster_boot(stat, groups: np.ndarray, n: int = 1000, seed: int = SEED):
    """Patient-level bootstrap: resample patients with all their rows. stat(row_index_array) -> float."""
    u, inv = np.unique(groups, return_inverse=True)
    rows = [[] for _ in u]
    for i, g in enumerate(inv):
        rows[g].append(i)
    rows = [np.array(r) for r in rows]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u))
        out.append(stat(np.concatenate([rows[k] for k in pick])))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def auroc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.bincount(inv, weights=r) / cnt)[inv]
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#6b6a64", "axes.labelcolor": "#1f1f1e", "xtick.color": "#6b6a64",
                         "ytick.color": "#6b6a64", "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ------------------------------------------------------------------------------------- categories
def _diagnoses_path() -> Path:
    import yaml
    root = Path(yaml.safe_load(open(YAML))["paths"]["mimic_root"])
    base = root / "hosp" / "diagnoses_icd.csv"
    return base / "diagnoses_icd.csv" if base.is_dir() else base


def cats():
    gem = pd.read_csv(ART / "icd9toicd10cmgem.csv", dtype=str)
    gem = gem[gem.no_map != "1"]
    gem["cat"] = gem.icd10cm.str[:3]
    g9 = gem[["icd9cm", "cat"]].drop_duplicates()
    df, *_ = cohort()
    keep = set(df.hadm_id.tolist())
    parts = []
    for d in pd.read_csv(_diagnoses_path(), usecols=["hadm_id", "icd_code", "icd_version"], dtype=str, chunksize=1_000_000):
        d = d[d.hadm_id.astype(np.int64).isin(keep)]
        d10 = pd.DataFrame({"hadm_id": d.hadm_id[d.icd_version == "10"].astype(np.int64), "cat": d.icd_code[d.icd_version == "10"].str[:3]})
        d9 = d[d.icd_version == "9"].merge(g9, left_on="icd_code", right_on="icd9cm")[["hadm_id", "cat"]]
        d9["hadm_id"] = d9.hadm_id.astype(np.int64)
        parts.append(pd.concat([d10, d9]).drop_duplicates())
    out = pd.concat(parts).drop_duplicates().reset_index(drop=True)
    out["cat"] = out.cat.astype("category")
    out.to_parquet(ART / "dx_cat.parquet")
    (ART / "gem_icd9_by_cat.json").write_text(json.dumps({c: sorted(g.icd9cm.tolist()) for c, g in g9.groupby("cat")}))
    print(f"{len(out):,} admission-category pairs, {out.cat.nunique():,} categories")


def titles() -> dict:
    import yaml
    root = Path(yaml.safe_load(open(YAML))["paths"]["mimic_root"])
    p = root / "hosp" / "d_icd_diagnoses.csv"
    p = p / "d_icd_diagnoses.csv" if p.is_dir() else p
    d = pd.read_csv(p, dtype=str)
    d = d[d.icd_version == "10"]
    d["cat"] = d.icd_code.str[:3]
    d["len"] = d.icd_code.str.len()
    return d.sort_values(["cat", "len", "icd_code"]).groupby("cat").long_title.first().to_dict()  # 3-char title if listed


# ------------------------------------------------------------------------------------- B1
def b1():
    cfg = cfg_for(VOC21)
    df, tr, inner, val = cohort()
    bn = BN.from_dict(json.loads((KB21 / "bn.json").read_text()))
    dx = kb.props(cfg, "diagnosis")
    te = df[~tr].reset_index(drop=True)
    P, _ = kb.posteriors_by_pattern(cfg, bn, te)
    cal = cfg["engine"].get("calibration") or {}
    out, xs, ys = {"window": WIN, "test_admissions": int(len(te)), "diagnoses": {}}, [], []
    for i, d in enumerate(dx):
        p = P[:, i]
        pc = platt(p, *cal[d]) if d in cal else p
        pi = float(df.loc[tr, d].mean())
        out["diagnoses"][d] = {"prevalence_train": pi,
                               "posterior_quantiles": {f"{q:g}": float(np.quantile(p, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
                               "share_in_window": float(((p >= WIN[0]) & (p <= WIN[1])).mean()),
                               "share_in_window_calibrated": float(((pc >= WIN[0]) & (pc <= WIN[1])).mean()),
                               "share_in_0.01_0.04": float(((p >= 0.01) & (p <= WIN[1])).mean())}
        xs.append(math.log10(pi))
        ys.append(out["diagnoses"][d]["share_in_window"])
    b, a = np.polyfit(xs, ys, 1)
    pred = a + b * np.array(xs)
    r2 = 1 - ((np.array(ys) - pred) ** 2).sum() / ((np.array(ys) - np.mean(ys)) ** 2).sum()
    out["fit_share_in_window_vs_log10_prevalence"] = {"intercept": float(a), "slope": float(b), "r2": float(r2), "points": 6,
                                                       "note": "fitted on six points; an interpolation aid, not a law"}
    (RES / "posterior_spread.json").write_text(json.dumps(out, indent=1))
    plt = _plt()
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.2), sharex=True)
    bins = np.linspace(-6, 0, 61)
    for ax, (i, d) in zip(axs.flat, enumerate(dx)):
        lp = np.log10(np.clip(P[:, i], 1e-6, 1))
        ax.axvspan(math.log10(WIN[0]), math.log10(WIN[1]), color="#2a78d6", alpha=0.12, lw=0)
        ax.hist(lp, bins=bins, color="#1f1f1e", alpha=0.75, weights=np.full(len(lp), 1 / len(lp)))
        ax.axvline(math.log10(T_DEC), color="#6b6a64", lw=1, ls=":")
        x = out["diagnoses"][d]
        ax.set_title(f"{d.replace('_', ' ')}: prevalence {x['prevalence_train']:.1%}, {x['share_in_window']:.0%} in window", loc="left", fontsize=8.5)
        ax.set_yticks([])
    for ax in axs[1]:
        ax.set_xlabel("log10 posterior (network)")
    fig.text(0.01, 0.005, f"Shaded: decision window [{WIN[0]}, {WIN[1]}]. Dotted: t = 1%. {len(te):,} test admissions.", fontsize=7.5, color="#6b6a64")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(REP / "posterior_spread.png", dpi=180)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- B2
def b2():
    d = pd.read_parquet(ART / "dx_cat.parquet")
    df, tr, inner, val = cohort()
    h_tr, h_in = set(df.hadm_id[tr]), set(df.hadm_id[inner])
    n_tr = d[d.hadm_id.isin(h_tr)].cat.value_counts()
    n_in = d[d.hadm_id.isin(h_in)].cat.value_counts().reindex(n_tr.index, fill_value=0)
    c = pd.DataFrame({"cases_train": n_tr, "cases_inner": n_in})
    c["prevalence"] = c.cases_train / tr.sum()
    c = c[c.cases_train > 0]
    excl = c.index.str[0].isin(list(EXCLUDE_CH)) | c.index.isin(list(EXCLUDE_CAT))
    elig = c[~excl]
    floor_ok = elig.cases_inner >= FLOOR
    removed = elig[~floor_ok]
    sp = json.loads((RES / "posterior_spread.json").read_text())["fit_share_in_window_vs_log10_prevalence"]
    tt = titles()
    short = elig[floor_ok & (elig.prevalence >= 0.001) & (elig.prevalence < 0.05)].sort_values("prevalence", ascending=False)
    rows = [{"category": k, "title": tt.get(k, ""), "cases_train": int(r.cases_train), "cases_inner": int(r.cases_inner),
             "prevalence": float(r.prevalence),
             "predicted_share_in_window_B1": float(np.clip(sp["intercept"] + sp["slope"] * math.log10(r.prevalence), 0, 1))}
            for k, r in short.iterrows()]
    targets = {}
    pool = elig[floor_ok]
    for band, lim in BANDS.items():
        if lim is None:
            continue
        lo, hi = lim
        cand = pool[(pool.prevalence >= lo) & (pool.prevalence < hi)]
        chosen = []
        for i in range(6):
            pt = math.log10(lo) + (math.log10(hi) - math.log10(lo)) * (i + 0.5) / 6
            left = cand.drop(index=chosen)
            dist = (np.log10(left.prevalence) - pt).abs()
            best = left.assign(dist=dist).sort_values(["dist", "cases_train"], ascending=[True, False]).index[0]
            chosen.append(best)
        targets[band] = [{"category": k, "title": tt.get(k, ""), "prevalence": float(cand.loc[k, "prevalence"]),
                          "cases_train": int(cand.loc[k, "cases_train"]), "cases_inner": int(cand.loc[k, "cases_inner"])} for k in chosen]
    out = {"categories_with_cases": int(len(c)), "excluded_by_chapter_or_overlap": int(excl.sum()), "eligible": int(len(elig)),
           "removed_by_floor": int(len(removed)), "floor_cases_inner": FLOOR, "floor_prevalence_inner": FLOOR / int(inner.sum()),
           "largest_removed_prevalence": float(removed.prevalence.max()) if len(removed) else None,
           "smallest_kept_prevalence": float(elig[floor_ok].prevalence.min()),
           "shortlist_0.1_to_5pct": len(rows), "shortlist": rows,
           "prediction_note": "predicted_share_in_window_B1 extrapolates a six-point fit made at prevalence 5-25%"}
    (RES / "candidate_targets.json").write_text(json.dumps(out, indent=1))
    Path("tasks").mkdir(exist_ok=True)
    Path("tasks/band_targets.json").write_text(json.dumps({"rule": "tasks/plan_bands.md item 5", "bands": targets}, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "shortlist"}, indent=1))
    print(json.dumps(targets, indent=1))


# ------------------------------------------------------------------------------------- B3
B3_N = 3000


def _b3_sample():
    df, tr, *_ = cohort()
    te = df[~tr]
    return te.sample(n=B3_N, random_state=SEED)


def b3(start: int, end: int):
    s = _b3_sample().iloc[start:end]
    nets = {n: BN.from_dict(json.loads(Path(f"artifacts/hardness/bn_dx{n}.json").read_text())) for n in (20, 40, 80)}
    rows, t0 = [], time.perf_counter()
    for _, r in s.iterrows():
        rec = {"hadm_id": int(r.hadm_id), "subject_id": int(r.subject_id)}
        for n, bn in nets.items():
            names = bn.struct.names
            dx = [x for x in names if x.startswith("dx_")]
            E = Evidence({k: int(r[k]) for k in names if not k.startswith("dx_") and r[k] >= 0}, {})
            p = np.array([posterior_exact(bn, E, [t])[0][t] for t in dx])
            rec[f"in_window_{n}"] = int(((p >= WIN[0]) & (p <= WIN[1])).sum())
            rec[f"borderline_{n}"] = int(((p >= 0.01) & (p <= WIN[1])).sum())
        rows.append(rec)
    pd.DataFrame(rows).to_parquet(ART / f"b3_{start:05d}.parquet")
    print(f"{start}-{end}: {len(rows)} admissions ({time.perf_counter() - t0:.0f}s)")


def b3_summary():
    d = pd.concat([pd.read_parquet(p) for p in sorted(ART.glob("b3_*.parquet"))]).drop_duplicates("hadm_id")
    out = {"admissions": int(len(d)), "patients": int(d.subject_id.nunique()), "window": WIN, "networks": {}}
    g = d.subject_id.to_numpy()
    for n in (20, 40, 80):
        net = {}
        for key, col in (("any_in_window", f"in_window_{n}"), ("any_in_0.01_0.04", f"borderline_{n}")):
            x = (d[col].to_numpy() > 0).astype(float)
            net[key] = {"share": float(x.mean()), "ci95": cluster_boot(lambda idx: x[idx].mean(), g)}
        net["median_diagnoses_in_window"] = float(d[f"in_window_{n}"].median())
        out["networks"][f"{n}_diagnoses"] = net
    out["note"] = "[0.001, 0.04] is also where an absent diagnosis's posterior usually sits; the borderline row is stricter"
    (RES / "window_coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- C1
def _fit_platt(p, y):
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def c1(band: str):
    tg = json.loads(Path("tasks/band_targets.json").read_text())["bands"][band]
    gem = json.loads((ART / "gem_icd9_by_cat.json").read_text())
    voc = json.loads(VOC21.read_text())
    props = {}
    for k, v in voc["propositions"].items():  # targets take the six diagnoses' node positions
        if v["kind"] == "diagnosis":
            if not any(x["kind"] == "diagnosis" for x in props.values()):
                for t in tg:
                    props[f"dx_{t['category']}"] = {"kind": "diagnosis", "label": t["title"], "icd10": [t["category"]],
                                                    "icd9": gem.get(t["category"], [])}
        else:
            props[k] = v
    voc = {**voc, "name": f"band_{band}", "description": f"v2.1 findings, {band}-band targets (tasks/plan_bands.md)",
           "options": {"knowledge.max_parents": voc["options"]["knowledge.max_parents"]}, "propositions": props}
    vp = Path(f"configs/vocab_band_{band}.json")
    vp.write_text(json.dumps(voc, indent=1))
    cfg = cfg_for(vp)
    d = pd.read_parquet(ART / "dx_cat.parquet")
    df, tr, inner, val = cohort()
    names = kb.props(cfg, "diagnosis")
    tgt = pd.DataFrame({"hadm_id": df.hadm_id, "subject_id": df.subject_id})
    for t, n in zip(tg, names):
        tgt[n] = df.hadm_id.isin(set(d.hadm_id[d.cat == t["category"]])).astype(np.int8).to_numpy()
    (ART / band).mkdir(exist_ok=True)
    tgt.to_parquet(ART / band / "targets.parquet")
    f = frame(band, cfg)
    t0 = time.perf_counter()
    bn = kb.learn_bn(cfg, f[tr])
    (ART / band / "bn.json").write_text(json.dumps(bn.to_dict()))
    bn_in = kb.learn_bn(cfg, f[inner])  # calibration as vocab.py: inner-train network, validation patients
    Pm, _ = kb.posteriors_by_pattern(cfg, bn_in, f[val])
    Yv = f.loc[val, names].to_numpy(int)
    half = np.random.default_rng(SEED + 5).random(len(Yv)) < 0.5
    cross = np.zeros_like(Pm)
    for a_, b_ in ((half, ~half), (~half, half)):
        for i in range(len(names)):
            cross[b_, i] = platt(Pm[b_, i], *_fit_platt(Pm[a_, i], Yv[a_, i]))
    raw_b, cal_b = float(((Pm - Yv) ** 2).mean(0).mean()), float(((cross - Yv) ** 2).mean(0).mean())
    cal = {n: list(_fit_platt(Pm[:, i], Yv[:, i])) for i, n in enumerate(names)} if cal_b < raw_b - 1e-4 else None
    (ART / band / "calibration.json").write_text(json.dumps({"calibration": cal, "brier_raw": raw_b, "brier_crossfit_calibrated": cal_b}))
    print(f"{band}: {len(bn.struct.names)} nodes, {sum(len(p) for p in bn.struct.parents)} edges, learned in {time.perf_counter() - t0:.0f}s; "
          f"calibration {'kept' if cal else 'not kept'} (Brier {raw_b:.5f} -> {cal_b:.5f})")
    print({n: round(float(f.loc[tr, n].mean()), 5) for n in names})


def structure():
    for band in ("uncommon", "rare"):
        b = BN.from_dict(json.loads((ART / band / "bn.json").read_text()))
        (ART / band / "structure_only.json").write_text(json.dumps({"names": b.struct.names, "parents": [list(p) for p in b.struct.parents]}))
    print("written")


# ------------------------------------------------------------------------------------- C2
def c2(band: str):
    out = {}
    for band in [band]:
        bn, cfg, df, dx = band_net(band)
        _, tr, *_ = cohort()
        names = bn.struct.names
        y = df.loc[tr, names].to_numpy(np.int8)
        st = bn.struct
        cells = {}
        ti = [names.index(t) for t in dx]
        for t, j in zip(dx, ti):
            def rows_count(k):
                pa = st.parents[k]
                ok = (y[:, [k, *pa]] >= 0).all(1)
                idx = st.pa_index(y[ok].astype(np.int64), k)
                return np.bincount(idx, minlength=2 ** len(pa))
            own = rows_count(j)
            kids = [k for k in range(len(names)) if j in st.parents[k]]
            kid = np.concatenate([rows_count(k) for k in kids]) if kids else np.array([])
            cells[t] = {"own_rows_min": int(own.min()), "children": len(kids),
                        "children_rows_min": int(kid.min()) if len(kid) else None,
                        "children_rows_share_below_30": float((kid < 30).mean()) if len(kid) else 0.0}
        te = df[~tr].reset_index(drop=True)
        P, _ = kb.posteriors_by_pattern(cfg, bn, te)
        g = te.subject_id.to_numpy()
        au, cov = {}, {}
        for i, t in enumerate(dx):
            yt, pt = te[t].to_numpy(), np.round(P[:, i], 12)  # VE round-off (1e-17) would otherwise break ties at random
            n_par = len(st.parents[names.index(t)])
            au[t] = {"auroc": auroc(yt, pt), "ci95": cluster_boot(lambda idx: auroc(yt[idx], pt[idx]), g, n=200),
                     "prevalence_test": float(yt.mean()), "parents": n_par, "distinct_posteriors": int(len(np.unique(pt)))}
            inw = (pt >= WIN[0]) & (pt <= WIN[1])
            cov[t] = {"share_in_window": float(inw.mean())}
        allp = P.ravel()
        inw = (P >= WIN[0]) & (P <= WIN[1])
        out[band] = {"targets": dx, "edges": int(sum(len(p) for p in st.parents)), "cells": cells, "auroc": au, "coverage": cov,
                     "instances_in_window": int(inw.sum()), "patients_with_window_instance": int(len(np.unique(g[inw.any(1)]))),
                     "share_instances_in_window": float(inw.mean()), "median_posterior": float(np.median(allp))}
        (ART / band / "c2.json").write_text(json.dumps(out[band], indent=1))
        print(band, json.dumps(out[band], indent=1), flush=True)


def c2_summary():
    out = {b: json.loads((ART / b / "c2.json").read_text()) for b in BANDS}
    r = out["rare"]
    aus = [v["auroc"] for v in r["auroc"].values()]
    checks = {"auroc_median_ge_0.65": bool(np.median(aus) >= 0.65),
              "at_most_2_targets_ci_lo_below_0.6": bool(sum(v["ci95"][0] < 0.6 for v in r["auroc"].values()) <= 2),
              "own_rows_ge_30": bool(all(v["own_rows_min"] >= 30 for v in r["cells"].values())),
              "children_rows_below_30_lt_10pct": bool(all(v["children_rows_share_below_30"] < 0.10 for v in r["cells"].values())),
              "coverage_200_instances_100_patients": bool(r["instances_in_window"] >= 200 and r["patients_with_window_instance"] >= 100)}
    out["rare_band_checks"] = checks
    out["note"] = "AUROC CIs: patient-level bootstrap, 200 resamples"
    (RES / "band_networks.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(checks, indent=1))


# ------------------------------------------------------------------------------------- C3
def node_max(bn: BN, j: int, e: int, obs: dict) -> float:
    pa = bn.struct.parents[j]
    best = 0.0
    for r in range(2 ** len(pa)):
        bits = [(r >> (len(pa) - 1 - i)) & 1 for i in range(len(pa))]
        if all(obs.get(p, b) == b for p, b in zip(pa, bits)):
            q = bn.cpts[j][r, 1]
            best = max(best, q if e == 1 else 1 - q)
    return best


def lw_exact_all(bn: BN, E: Evidence, dx: list[str]):
    """quantum_study.lw_exact for all targets at once: one VE for the posteriors, one on the squared-CPT network."""
    post, pe, _ = posterior_exact(bn, E, dx)
    sq = BN(bn.struct, [c.copy() for c in bn.cpts])
    for k, v in E.hard.items():
        j = bn.struct.index(k)
        sq.cpts[j][:, v] = bn.cpts[j][:, v] ** 2
    tab = marginal(sq, E, [bn.struct.index(t) for t in dx])
    tot = tab.sum()
    out = []
    for i, t in enumerate(dx):
        a1 = float(np.moveaxis(tab, i, 0)[1].sum())
        a0 = tot - a1
        p = post[t]
        var = ((1 - p) ** 2 * a1 + p ** 2 * a0) / pe ** 2
        out.append((p, pe, (a0 + a1) / pe ** 2, (Z90 / REL) ** 2 * var / p ** 2))
    return out


def c3(band: str, start: int = 0, end: int = 10 ** 9):
    bn, cfg, df, dx = band_net(band)
    _, tr, *_ = cohort()
    ev = kb.evidence(cfg)
    te = df[~tr].reset_index(drop=True)
    E = te[ev].to_numpy(np.int8)
    pats, inv = np.unique(E, axis=0, return_inverse=True)
    part = ART / band / f"c3_patterns_{start:06d}.parquet"
    t0 = time.perf_counter()
    rows = []
    idx = range(start, min(end, len(pats)))
    sys.path.insert(0, "scripts")
    from quantum_study import lw_exact  # reference implementation, checked on the first patterns
    for i in idx:
        p = pats[i]
        Ev = Evidence({n: int(v) for n, v in zip(ev, p) if v >= 0}, {})
        res = lw_exact_all(bn, Ev, dx)
        if i < start + 3:
            for (pp, pe, disp, n), t in zip(res, dx):
                ref = lw_exact(bn, Ev, t)
                assert abs(ref["samples"] / n - 1) < 1e-9 and abs(ref["dispersion"] / disp - 1) < 1e-9, (ref, n, disp)
        obs = {bn.struct.index(k): v for k, v in Ev.hard.items()}
        logW = sum(math.log(node_max(bn, j, v, obs)) for j, v in obs.items())
        for (pp, pe, disp, n), t in zip(res, dx):
            rows.append((i, t, pp, pe, logW, disp, n, len(obs)))
    d = pd.DataFrame(rows, columns=["pattern", "target", "p", "p_evidence", "logW", "dispersion", "lw_samples", "findings"])
    d.to_parquet(part)
    pd.DataFrame({"hadm_id": te.hadm_id, "subject_id": te.subject_id, "pattern": inv.reshape(-1),
                  **{f"y_{t}": te[t].to_numpy() for t in dx}}).to_parquet(ART / band / "c3_admissions.parquet")
    print(f"{band}: patterns {start}-{idx.stop} of {len(pats):,} x {len(dx)} targets in {time.perf_counter() - t0:.0f}s")


def _c3_patterns(band: str) -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(q) for q in sorted((ART / band).glob("c3_patterns_*.parquet"))]).drop_duplicates(["pattern", "target"])
    n = pd.read_parquet(ART / band / "c3_admissions.parquet").pattern.nunique()
    assert d.pattern.nunique() == n, (band, d.pattern.nunique(), n)
    return d


def disconnected(band: str) -> list[str]:
    """Amendment A3: targets with a single distinct posterior over test admissions (no path to any finding)."""
    c = json.loads((ART / band / "c2.json").read_text())
    return [t for t, v in c["auroc"].items() if v["distinct_posteriors"] == 1]


def c3_sample(band: str):
    """Amendment A6: 24 window instances. LW at N0 = formula (cap 3e6), 20 repeats, N_emp = N0 (q90 / 0.1)^2;
    emulated IAE (16 shots, alpha 0.2) against the law with the tuned kappa."""
    os.environ["AE_SHOTS"], os.environ["AE_ALPHA"] = "16", "0.2"
    sys.path.insert(0, "scripts")
    import lw_check
    from qddx.bn import posterior_lw
    bn, cfg, df, dx = band_net(band)
    ev = kb.evidence(cfg)
    d = _c3_patterns(band)
    d = d[~d.target.isin(disconnected(band))]
    a = pd.read_parquet(ART / band / "c3_admissions.parquet")
    first = a.reset_index().groupby("pattern")["index"].first()
    w = d[(d.p >= WIN[0]) & (d.p <= WIN[1])].sample(n=24, random_state=SEED)
    _, tr, *_ = cohort()
    te = df[~tr].reset_index(drop=True)
    kappa = json.loads((RES / "frontier.json").read_text())["kappa_tuned_iae"]
    out_p = ART / band / "c3_sample.jsonl"
    done = {json.loads(x)["k"] for x in out_p.read_text().splitlines()} if out_p.exists() else set()
    t0 = time.perf_counter()
    for k, (_, r) in enumerate(w.iterrows()):
        if k in done:
            continue
        if time.perf_counter() - t0 > 110:
            print(f"paused at {k}")
            return
        row = te.iloc[int(first[r.pattern])]
        E = Evidence({n: int(row[n]) for n in ev if row[n] >= 0}, {})
        n0 = int(min(max(r.lw_samples, 1000), 3e6))
        errs = [abs(posterior_lw(bn, E, [r.target], n0, SEED + 1000 * q + k)[0][r.target] - r.p) / r.p for q in range(20)]
        n_emp = n0 * (float(np.quantile(errs, 0.9)) / REL) ** 2
        aE = r.p_evidence / math.exp(r.logW)
        import signal

        def _stop(*_):
            raise TimeoutError
        signal.signal(signal.SIGALRM, _stop)
        signal.alarm(30)  # qiskit's IAE loop can stall on rare seeds; recorded, not retried with another seed
        try:
            iae, hang = lw_check.ae_cost(aE, aE * r.p, r.p, SEED + k), False
        except TimeoutError:
            iae, hang = None, True
        finally:
            signal.alarm(0)
        law = kappa * math.pi / REL * (1 / math.sqrt(aE) + 1 / math.sqrt(aE * r.p))
        rec = {"k": k, "target": r.target, "p": float(r.p), "lw_formula": float(r.lw_samples), "lw_n0": n0, "lw_empirical": n_emp,
               "qlw_law": law, "qlw_emulated": iae, "emulator_stalled": hang}
        with open(out_p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("done")


# ------------------------------------------------------------------------------------- A8: rare-band variant
def c1_variant():
    """Rare targets join every later node's candidate list; an edge involving a target needs gain above that target's
    permutation threshold (q95 over 20 permutations of max MI with any partner); every other edge as kb.learn_bn.
    Resumable: the permutation null is cached per target."""
    from qddx.bn import Struct, cond_mi
    cfg = cfg_for(Path("configs/vocab_band_rare.json"))
    df = frame("rare", cfg)
    _, tr, *_ = cohort()
    names = kb.props(cfg)
    dx = kb.props(cfg, "diagnosis")
    ti = [names.index(t) for t in dx]
    k = cfg["knowledge"]
    y = df.loc[tr, names].to_numpy(np.int8)
    out = ART / "rare_variant"
    out.mkdir(exist_ok=True)
    nullp = out / "null.json"
    null = json.loads(nullp.read_text()) if nullp.exists() else {}
    t0 = time.perf_counter()
    for t, j in zip(dx, ti):
        if t in null:
            continue
        if time.perf_counter() - t0 > 120:
            print("paused (null)")
            return
        rng = np.random.default_rng(SEED + j)
        mx = []
        for _ in range(20):
            yp = y.copy()
            yp[:, j] = rng.permutation(yp[:, j])
            mx.append(max(cond_mi(yp, j, c, []) for c in range(len(names)) if c not in ti))
        null[t] = {"max_mi_permuted": mx, "threshold": float(np.quantile(mx, 0.95)),
                   "observed_max_mi": float(max(cond_mi(y, j, c, []) for c in range(len(names)) if c not in ti))}
        nullp.write_text(json.dumps(null, indent=1))
    thr = {j: null[t]["threshold"] for t, j in zip(dx, ti)}
    # kb.learn_bn's prefilter, exactly
    rows = y if len(y) <= int(k.get("structure_rows", 100000)) else \
        y[np.random.default_rng(cfg["seed"]).choice(len(y), int(k.get("structure_rows", 100000)), replace=False)]
    lim = int(k.get("parent_candidates", 8))
    allowed = {}
    for j in range(len(names)):
        mi = {c: cond_mi(rows, j, c, []) for c in range(j)}
        allowed[j] = sorted(mi, key=mi.get, reverse=True)[:lim]

    def greedy(j, cand, use_thr):
        chosen = []
        while len(chosen) < min(k["max_parents"], len(cand)):
            gains = {c: cond_mi(y, j, c, chosen) for c in cand if c not in chosen}
            if not gains:
                break
            ok = {c: g for c, g in gains.items() if g >= ((thr.get(c, thr.get(j)) if (c in thr or j in thr) else k["min_gain"]) if use_thr else k["min_gain"])}
            if not ok:
                break
            chosen.append(max(ok, key=ok.get))
        return tuple(sorted(chosen))
    # fidelity check: without the variant's two changes this reproduces the frozen rare-band network
    ref = BN.from_dict(json.loads((ART / "rare" / "bn.json").read_text()))
    for j in ti + [ti[-1] + 1, ti[-1] + 2]:
        assert greedy(j, allowed[j], False) == ref.struct.parents[j], (names[j], greedy(j, allowed[j], False), ref.struct.parents[j])
    parents = []
    for j in range(len(names)):
        cand = list(allowed[j]) + ([c for c in ti if c < j and c not in allowed[j]] if j > max(ti) else [])
        parents.append(greedy(j, cand, True))
    st = Struct(names, parents)
    cpts = []
    for j, pa in enumerate(parents):
        okr = (y[:, [j, *pa]] >= 0).all(1)
        idx = st.pa_index(y[okr].astype(np.int64), j)
        counts = np.full((2 ** len(pa), 2), 1.0)
        np.add.at(counts, (idx, y[okr, j].astype(np.int64)), 1)
        cpts.append(counts / counts.sum(1, keepdims=True))
    bn = BN(st, cpts)
    (out / "bn.json").write_text(json.dumps(bn.to_dict()))
    (out / "structure_only.json").write_text(json.dumps({"names": names, "parents": [list(p) for p in parents]}))
    edges_t = {t: {"parents": [names[c] for c in parents[j]], "children": [names[c] for c in range(len(names)) if j in parents[c]],
                   "threshold": thr[j], "observed_max_mi": null[t]["observed_max_mi"]} for t, j in zip(dx, ti)}
    (out / "edges.json").write_text(json.dumps(edges_t, indent=1))
    print(f"rare variant: {sum(len(p) for p in parents)} edges ({time.perf_counter() - t0:.0f}s)")
    print(json.dumps(edges_t, indent=1))


# ------------------------------------------------------------------------------------- A7: top-5 differential
def b3top(start: int, end: int):
    s = _b3_sample().iloc[start:end]
    nets = {n: BN.from_dict(json.loads(Path(f"artifacts/hardness/bn_dx{n}.json").read_text())) for n in (20, 40, 80)}
    rows, t0 = [], time.perf_counter()
    for _, r in s.iterrows():
        rec = {"hadm_id": int(r.hadm_id), "subject_id": int(r.subject_id)}
        for n, bn in nets.items():
            names = bn.struct.names
            dx = [x for x in names if x.startswith("dx_")]
            E = Evidence({k: int(r[k]) for k in names if not k.startswith("dx_") and r[k] >= 0}, {})
            p = np.sort([posterior_exact(bn, E, [t])[0][t] for t in dx])[::-1][:5]
            rec[f"top5_in_window_{n}"] = int(((p >= WIN[0]) & (p <= WIN[1])).sum())
            rec[f"top5_borderline_{n}"] = int(((p >= 0.01) & (p <= WIN[1])).sum())
            rec[f"top5_min_{n}"] = float(p[-1])
        rows.append(rec)
    pd.DataFrame(rows).to_parquet(ART / f"b3top_{start:05d}.parquet")
    print(f"{start}-{end}: {len(rows)} admissions ({time.perf_counter() - t0:.0f}s)")


def b3top_summary():
    d = pd.concat([pd.read_parquet(p) for p in sorted(ART.glob("b3top_*.parquet"))]).drop_duplicates("hadm_id")
    out = json.loads((RES / "window_coverage.json").read_text())
    g = d.subject_id.to_numpy()
    top = {"admissions": int(len(d))}
    for n in (20, 40, 80):
        x = {}
        for key, col in (("any_top5_in_window", f"top5_in_window_{n}"), ("any_top5_in_0.01_0.04", f"top5_borderline_{n}")):
            v = (d[col].to_numpy() > 0).astype(float)
            x[key] = {"share": float(v.mean()), "ci95": cluster_boot(lambda idx: v[idx].mean(), g)}
        m = d[f"top5_in_window_{n}"].to_numpy().astype(float)
        x["mean_of_top5_in_window"] = {"value": float(m.mean()), "ci95": cluster_boot(lambda idx: m[idx].mean(), g)}
        x["median_5th_posterior"] = float(d[f"top5_min_{n}"].median())
        top[f"{n}_diagnoses"] = x
    bn_ = json.loads((RES / "band_networks.json").read_text())
    top["per_instance_share_in_window"] = {b: bn_[b]["share_instances_in_window"] for b in ("common", "uncommon")}
    out["top5_differential"] = top
    (RES / "window_coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(top, indent=1))


# ------------------------------------------------------------------------------------- C4 + C5
def _jt(groups: list[np.ndarray]) -> float:
    """Jonckheere-Terpstra statistic for a decreasing order (pairs i < j with x_i > x_j; ties count 1/2)."""
    s = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = np.sort(groups[i]), np.sort(groups[j])
            lt = np.searchsorted(a, b, side="left")   # for each b: number of a below it
            le = np.searchsorted(a, b, side="right")
            s += float((len(a) - le).sum() + 0.5 * (le - lt).sum())  # a > b, plus ties
    return s


PRIMARY = ["common", "uncommon"]


def _instances(band: str, v1: dict) -> pd.DataFrame:
    d = _c3_patterns(band)
    d = d[~d.target.isin(disconnected(band))]
    a = pd.read_parquet(ART / band / "c3_admissions.parquet")[["subject_id", "pattern"]]
    x = a.merge(d, on="pattern")
    x["band"] = band
    x["ratio_v1"] = [v1[band][t][str(int(f))] for t, f in zip(x.target, x.findings)]
    x["aE"] = x.p_evidence / np.exp(x.logW)
    return x


def _log_ratio(x, t, k, unit):
    eps = REL if t is None else REL * np.maximum(x.p, t) / x.p
    lw = x.lw_samples * (REL / eps) ** 2
    q = k * math.pi / eps * (1 / np.sqrt(x.aE) + 1 / np.sqrt(x.aE * x.p))
    if unit == "v1":
        q = q * x.ratio_v1
    return np.log10(q / lw)


def _endpoint(R: pd.DataFrame, col: str = "lr") -> dict:
    from scipy.stats import wilcoxon
    lr, g = R[col].to_numpy(), R.subject_id.to_numpy()
    pm = R.groupby("subject_id")[col].mean().to_numpy()
    return {"instances": int(len(R)), "patients": int(len(pm)), "median_log10_ratio": float(np.median(lr)),
            "ratio": float(10 ** np.median(lr)), "ci95_log10": cluster_boot(lambda idx: float(np.median(lr[idx])), g),
            "share_qlw_cheaper": float((lr < 0).mean()), "wilcoxon_patient_means_p": float(wilcoxon(pm).pvalue),
            "median_patient_mean_log10": float(np.median(pm))}


def c5():
    fr = json.loads((RES / "frontier.json").read_text())
    kappa = fr["kappa_tuned_iae"]
    v1 = json.loads((ART / "v1_counts.json").read_text())
    bands = PRIMARY + (["rare_variant"] if (ART / "rare_variant" / "c3_admissions.parquet").exists() else [])
    X = pd.concat([_instances(b, v1) for b in bands], ignore_index=True)
    X["lr"] = _log_ratio(X, T_DEC, kappa, "v1")
    inwin = (X.p >= WIN[0]) & (X.p <= WIN[1])
    out = {"amendment": "tasks/plan_bands_amendment.md", "kappa_tuned": kappa, "window": WIN, "t": T_DEC,
           "excluded_disconnected_targets": {b: disconnected(b) for b in bands}}
    prim = X[inwin & X.band.isin(PRIMARY)].reset_index(drop=True)
    out["primary"] = {"definition": "median log10(qLW / LW): common + uncommon bands, network posterior in [0.001, 0.04], "
                                    "decision metric t = 1%, V1 gates, tuned kappa", **_endpoint(prim)}
    out["by_band"] = {b: _endpoint(X[inwin & (X.band == b)].reset_index(drop=True)) for b in bands}
    # post hoc (not pre-specified): targets whose posterior takes <= 2 values are a table lookup given one observed
    # parent; both samplers then estimate a near-independent Bernoulli, where AE's 1/sqrt(p) beats 1/p trivially
    few = [t for b in PRIMARY for t, v in json.loads((ART / b / "c2.json").read_text())["auroc"].items() if v["distinct_posteriors"] <= 2]
    out["post_hoc_excluding_lookup_targets"] = {"excluded": few,
                                                **_endpoint(X[inwin & X.band.isin(PRIMARY) & ~X.target.isin(few)].reset_index(drop=True))}
    out["by_target"] = {t: {"band": g.band.iloc[0], "instances": int(len(g)), "median_log10_ratio": float(g.lr.median()),
                            "share_qlw_cheaper": float((g.lr < 0).mean()), "median_weight_dispersion": float(g.dispersion.median())}
                        for t, g in X[inwin].groupby("target")}
    sec = {}
    for b in bands:
        B = X[inwin & (X.band == b)]
        for mname, t in (("decision_t0.01", 0.01), ("decision_t0.001", 0.001), ("relative", None)):
            for kname, k in (("tuned", kappa), ("ideal", 1.0)):
                for unit in ("v1", "model_evaluations"):
                    v = _log_ratio(B, t, k, unit).to_numpy()
                    sec[f"{b}|{mname}|{kname}|{unit}"] = {"n": int(len(v)), "median_log10_ratio": float(np.median(v)),
                                                          "share_qlw_cheaper": float((v < 0).mean())}
        sec[f"{b}|all_instances|decision_t0.01|tuned|v1"] = {"n": int((X.band == b).sum()),
                                                             "share_qlw_cheaper": float((X.lr[X.band == b] < 0).mean())}
    out["secondary_not_multiplicity_corrected"] = sec
    # A4: trend across the two ordered bands, within-patient permutation
    rng = np.random.default_rng(SEED)
    pmb = X[inwin & X.band.isin(PRIMARY)].groupby(["subject_id", "band"]).lr.mean().unstack()[PRIMARY]
    obs = _jt([pmb[b].dropna().to_numpy() for b in PRIMARY])
    M = pmb.to_numpy()
    ok = ~np.isnan(M)
    cv = np.take_along_axis(M, np.argsort(~ok, axis=1, kind="stable"), 1)
    null = []
    for _ in range(1000):
        keys = np.where(ok, rng.random(M.shape), np.inf)
        Mp = np.full_like(M, np.nan)
        np.put_along_axis(Mp, np.argsort(keys, axis=1), cv, 1)
        null.append(_jt([Mp[:, j][~np.isnan(Mp[:, j])] for j in range(M.shape[1])]))
    out["trend_common_to_uncommon"] = {"jt_statistic": obs, "p_value_within_patient_permutation": float((np.sum(np.array(null) >= obs) + 1) / 1001),
                                       "median_patient_mean_by_band": {b: float(np.nanmedian(pmb[b])) for b in PRIMARY}}
    # C4 on D_eff
    P = X[X.band.isin(PRIMARY)]
    D = P.lw_samples * P.p / (1 - P.p)
    xs, ys = np.log10(P.p.to_numpy()), np.log10(D.to_numpy())
    u, inv = np.unique(P.subject_id.to_numpy(), return_inverse=True)
    S = np.stack([np.bincount(inv, w) for w in (np.ones_like(xs), xs, ys, xs * xs, xs * ys)], 1)

    def slope(sm):
        n, sx, sy, sxx, sxy = sm
        return (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b_obs = float(slope(S.sum(0)))
    bs = [slope(S[rng.integers(0, len(u), len(u))].sum(0)) for _ in range(1000)]
    ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    rises = bool(b_obs < 0 and ci[1] < 0)
    c4 = {"slope_log10_Deff_on_log10_p": b_obs, "ci95": ci, "rises_as_posterior_falls": rises}
    r = pd.DataFrame(fr["rows"])
    D0 = float(np.median(r.lw_samples * r.p / (1 - r.p)))
    S0 = float(np.median(r.cost_qlw_ideal * 0.1 / math.pi / (1 + r.p ** -0.5)))
    p0 = float(np.median(r.p))
    g1 = float(np.mean(json.loads((RES / "oracle_v2.json").read_text())["summary"]["v1"]["v21_ratio_range"]))
    grid = np.logspace(math.log10(0.5), -5, 241)

    def partA(sl=0.0, t=T_DEC):
        Dg = D0 * (grid / p0) ** sl
        eps = REL * np.maximum(grid, t) / grid if t else REL
        return np.log10(g1 * kappa * math.pi / eps * S0 * (1 + grid ** -0.5) / (Dg * (1 - grid) / grid * (REL / eps) ** 2))
    for lab, sl in (("as_published", 0.0), ("with_measured_slope", b_obs)):
        for mname, t in (("decision", T_DEC), ("relative", None)):
            w = np.flatnonzero(partA(sl, t) <= 0)
            c4[f"partA_p_star_{mname}_{lab}"] = float(grid[w[0]]) if len(w) else 0.0
    out["C4_dispersion"] = c4
    curve = partA(b_obs if rises else 0.0)
    tol = {}
    for scope, Y in [("primary", X[inwin & X.band.isin(PRIMARY)])] + [(b, X[inwin & (X.band == b)]) for b in bands]:
        for lo, hi in ((0.001, 0.004), (0.004, 0.01), (0.01, 0.0400001)):
            B = Y[(Y.p >= lo) & (Y.p < hi)]
            if len(B) < 30:
                tol[f"{scope}|{lo}-{hi}"] = {"n": int(len(B)), "skipped": True}
                continue
            pm_ = float(np.median(B.p))
            pred = float(np.interp(math.log10(pm_), np.log10(grid[::-1]), curve[::-1]))
            meas = float(np.median(B.lr))
            tol[f"{scope}|{lo}-{hi}"] = {"n": int(len(B)), "median_p": pm_, "measured_log10": meas, "partA_log10": pred,
                                        "within_factor_2": bool(abs(meas - pred) <= math.log10(2))}
    out["tracks_partA"] = tol
    val = {}
    for b in bands:
        sp = ART / b / "c3_sample.jsonl"
        if not sp.exists():
            continue
        s = pd.DataFrame([json.loads(x) for x in sp.read_text().splitlines()])
        lw_r, q_r = (s.lw_empirical / s.lw_formula), (s.qlw_emulated / s.qlw_law).dropna()
        val[b] = {"n": int(len(s)), "lw_empirical_over_formula_median": float(lw_r.median()), "lw_iqr": [float(lw_r.quantile(.25)), float(lw_r.quantile(.75))],
                  "lw_capped_at_3e6": int((s.lw_n0 >= 3e6).sum()),
                  "qlw_emulated_over_law_median": float(q_r.median()), "qlw_iqr": [float(q_r.quantile(.25)), float(q_r.quantile(.75))],
                  "qlw_not_reached_or_stalled": int(s.qlw_emulated.isna().sum()),
                  "pass": bool(0.67 <= lw_r.median() <= 1.5 and 0.67 <= q_r.median() <= 1.5)}
    out["validation_in_window"] = val
    out["a_E_median_in_window_by_band"] = {b: float(np.median(X.aE[inwin & (X.band == b)])) for b in bands}
    pr = out["primary"]
    favourable = pr["ci95_log10"][1] < 0
    tracks = all(v.get("within_factor_2", True) for k, v in tol.items() if k.startswith("primary|"))
    moved = rises and c4["partA_p_star_decision_with_measured_slope"] < 0.001
    out["verdict_inputs"] = {"primary_ci_upper_below_0": favourable, "tracks_partA_primary": tracks, "Deff_rises": rises,
                             "validation_pass": {b: v["pass"] for b, v in val.items()}}
    out["verdict"] = ("claim holds" if favourable and tracks and not moved else
                      "claim fails" if (not tracks or moved) else "primary endpoint not met")
    (RES / "band_costs.json").write_text(json.dumps(out, indent=1))
    # figure
    plt = _plt()
    cols = {"common": "#6b6a64", "uncommon": "#eb6834", "rare_variant": "#2a78d6"}
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    edges = np.logspace(-5, math.log10(0.5), 29)
    for ax, (mname, t) in zip(axs, (("Decision metric (t = 1%), primary", T_DEC), ("10% relative error", None))):
        for b in bands:
            B = X[X.band == b]
            v = _log_ratio(B, t, kappa, "v1").to_numpy()
            idx = np.digitize(B.p.to_numpy(), edges)
            xc, med, lo_, hi_ = [], [], [], []
            for k in range(1, len(edges)):
                m = idx == k
                if m.sum() >= 30:
                    xc.append(math.sqrt(edges[k - 1] * edges[k]))
                    med.append(np.median(v[m]))
                    lo_.append(np.quantile(v[m], 0.25))
                    hi_.append(np.quantile(v[m], 0.75))
            ax.fill_between(xc, 10 ** np.array(lo_), 10 ** np.array(hi_), color=cols[b], alpha=0.15, lw=0)
            lab = {"rare_variant": "rare band (pipeline variant)"}.get(b, f"{b} band")
            ax.plot(xc, 10 ** np.array(med), color=cols[b], lw=2, label=f"{lab}, measured (median, IQR)")
        ax.plot(grid, 10 ** partA(b_obs if rises else 0.0, t), color="#1f1f1e", lw=1.5, ls="--", label="Part A calculation")
        ax.axhline(1, color="#1f1f1e", lw=0.8, ls=":")
        ax.axvspan(*WIN, color="#2a78d6", alpha=0.06, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("Posterior P(target | findings)  →  rarer")
        ax.set_title(mname, loc="left", fontsize=9.5)
        ax.grid(alpha=0.25, which="major")
    axs[0].set_ylabel("Quantum LW cost ÷ classical LW cost\n(tuned IAE, V1 gates; below 1 = quantum cheaper)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.01, 0.005, "All 108,220 test admissions × connected targets per band. Shaded: decision window [0.001, 0.04].", fontsize=7, color="#6b6a64")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(REP / "band_crossover.png", dpi=180)
    print(json.dumps({k: v for k, v in out.items() if k != "secondary_not_multiplicity_corrected"}, indent=1))


# ------------------------------------------------------------------------------------- W6: pruned fair baseline
def _v1_law(n: int, L: int, P: int, table: dict) -> tuple[float, float]:
    """(ratio_v1, A_v0) from the cost law (docs/quantum_cost_model.md): binary pruned network with n qubits, L literals
    (relevant evidence + target), state-preparation CX P. Saving term applied (the usual case)."""
    def m0(m):
        return 4 * m * m - 4 * m + 2 if 5 <= m <= 32 else table[str(m)] if str(m) in table else round(124 * m)
    f = n - L
    if L <= 1:
        o = L
    elif f == 0:
        o = m0(L)
    elif L == 2:
        o = 6
    elif L == 3:
        o = 14
    elif f >= -(-L // 2) - 1:
        o = 8 * L - 12
    else:
        o = 12 * L - 18
    A0 = P + o
    Q1 = 2 * (P + (6 * (L - 1) if L >= 2 else L)) + 6 * (n - 1)
    return Q1 / (2 * A0) if A0 else math.nan, A0


def w6(band: str, start: int = 0, end: int = 10 ** 9):
    """Per evidence pattern x target: drop evidence d-separated from the target given the rest (msbn.prune), then
    recompute LW and qLW quantities on what is left (barren nodes are pruned inside the elimination); the V1 ratio
    comes from the cost law on the pruned network."""
    sys.path.insert(0, "scripts")
    from msbn import from_binary, prune, ancestors
    bn, cfg, df, dx = band_net(band)
    ms = from_binary(bn)
    _, tr, *_ = cohort()
    ev = kb.evidence(cfg)
    te = df[~tr].reset_index(drop=True)
    E = te[ev].to_numpy(np.int8)
    pats = np.unique(E, axis=0)
    table = json.loads((RES / "cost_model.json").read_text())["mcx_noancilla_table"]
    st = bn.struct
    rows, t0 = [], time.perf_counter()
    for i in range(start, min(end, len(pats))):
        full = {st.index(n): int(v) for n, v in zip(ev, pats[i]) if v >= 0}
        for t in dx:
            ti = st.index(t)
            rel, keep = prune(ms, ti, full)
            Ev = Evidence({st.names[j]: v for j, v in rel.items()}, {})
            (pp, pe, disp, n_lw), = lw_exact_all(bn, Ev, [t])
            logW = sum(math.log(node_max(bn, j, v, rel)) for j, v in rel.items())
            P = int(sum(2 ** len(st.parents[j]) for j in keep if st.parents[j]))
            r1, A0 = _v1_law(len(keep), len(rel) + 1, P, table)
            rows.append((i, t, pp, pe, logW, disp, n_lw, len(full), len(rel), len(keep), r1))
    pd.DataFrame(rows, columns=["pattern", "target", "p", "p_evidence", "logW", "dispersion", "lw_samples", "evidence_given",
                                "evidence_used", "nodes_kept", "ratio_v1"]).to_parquet(ART / band / f"w6_{start:06d}.parquet")
    print(f"{band} W6: patterns {start}-{min(end, len(pats))} of {len(pats):,} ({time.perf_counter() - t0:.0f}s)")


def w6_summary():
    """Before/after pruning: the amended primary endpoint (A2) recomputed with pruned quantities for both engines."""
    kappa = json.loads((RES / "frontier.json").read_text())["kappa_tuned_iae"]
    v1 = json.loads((ART / "v1_counts.json").read_text())
    out = {"definition": "as band_costs.json primary (common + uncommon, window, decision t = 1%, V1 gates, tuned kappa), "
                         "both engines on the pruned network", "kappa": kappa}
    parts = []
    for b in PRIMARY:
        X = _instances(b, v1)
        w = pd.concat([pd.read_parquet(q) for q in sorted((ART / b).glob("w6_*.parquet"))]).drop_duplicates(["pattern", "target"])
        w = w.rename(columns={c: f"w_{c}" for c in w.columns if c not in ("pattern", "target")})
        n0 = len(X)
        X = X.merge(w, on=["pattern", "target"], how="left")
        assert len(X) == n0 and X.w_p.notna().all(), b
        assert np.allclose(X.p, X.w_p, rtol=1e-9, atol=1e-15)  # pruning is exact
        X["lr_before"] = _log_ratio(X, T_DEC, kappa, "v1")
        Y = pd.DataFrame({"p": X.p, "lw_samples": X.w_lw_samples, "aE": X.w_p_evidence / np.exp(X.w_logW), "ratio_v1": X.w_ratio_v1})
        X["lr"] = _log_ratio(Y, T_DEC, kappa, "v1")
        X = X.rename(columns={"w_evidence_given": "evidence_given", "w_evidence_used": "evidence_used", "w_nodes_kept": "nodes_kept"})
        parts.append(X)
    X = pd.concat(parts, ignore_index=True)
    inwin = (X.p >= WIN[0]) & (X.p <= WIN[1])
    W = X[inwin].reset_index(drop=True)
    out["before"] = _endpoint(W, "lr_before")
    out["after"] = _endpoint(W, "lr")
    out["by_band"] = {b: {"before": _endpoint(W[W.band == b].reset_index(drop=True), "lr_before"),
                          "after": _endpoint(W[W.band == b].reset_index(drop=True), "lr")} for b in PRIMARY}
    out["by_target"] = {t: {"before_median_log10": float(g.lr_before.median()), "after_median_log10": float(g.lr.median()),
                            "after_share_qlw_cheaper": float((g.lr < 0).mean()), "median_evidence_used": float(g.evidence_used.median()),
                            "median_evidence_given": float(g.evidence_given.median()), "median_nodes_kept": float(g.nodes_kept.median())}
                        for t, g in W.groupby("target")}
    frac = 1 - W.evidence_used / W.evidence_given
    out["pruning"] = {"median_share_of_evidence_removed": float(frac.median()),
                      "share_of_instances_with_most_evidence_removed": float((frac > 0.5).mean()),
                      "share_of_instances_with_no_evidence_left": float((W.evidence_used == 0).mean()),
                      "median_nodes_kept": float(W.nodes_kept.median())}
    out["by_evidence_left"] = {lab: _endpoint(W[m].reset_index(drop=True), "lr") for lab, m in
                               (("most_evidence_removed", frac > 0.5), ("half_or_more_kept", frac <= 0.5)) if m.sum() > 100}
    (RES / "w6_pruned_baseline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "by_target"}, indent=1))
    for t, v in out["by_target"].items():
        print(t, {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {"cats": cats, "b1": b1, "b2": b2, "b3": b3, "b3-summary": b3_summary, "b3top": b3top, "b3top-summary": b3top_summary,
          "c1": c1, "c1-variant": c1_variant, "structure": structure, "c2": c2, "c2-summary": c2_summary,
          "c3": c3, "c3-sample": c3_sample, "c5": c5, "w6": w6, "w6-summary": w6_summary}[cmd]
    fn(*[int(a) if a.isdigit() else a for a in args])
