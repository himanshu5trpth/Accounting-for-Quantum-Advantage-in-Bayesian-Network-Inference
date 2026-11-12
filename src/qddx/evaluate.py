"""Experiments: MIMIC-scale validation of the knowledge base, the 200-note study with ablations, engine
comparison, the rare-evidence quantum advantage study, noise, and privacy leakage."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import kb
from . import notes as N
from .agent import Agent, FakeClient, OllamaClient
from .bn import BN, Evidence, posterior_exact, posterior_lw, posterior_rejection
from .engine import dx_names, explain, posterior, report
from .quantum import check_circuit, circuit_qubits, iae_emulated, iae_relative, make_sampler, posterior_quantum

log = logging.getLogger("qddx")


# ------------------------------------------------------------------------------------- metrics
def _auc(y, p, w):
    keep = w > 0
    if len(np.unique(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], p[keep], sample_weight=w[keep]))


def metric_set(Y: np.ndarray, Pm: np.ndarray, dx: list[str], top_k: int, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(Y)) if w is None else w
    out = {}
    for i, d in enumerate(dx):
        out[f"auroc_{d}"] = _auc(Y[:, i], Pm[:, i], w)
        out[f"brier_{d}"] = float(np.average((Pm[:, i] - Y[:, i]) ** 2, weights=w))
    out["macro_auroc"] = float(np.nanmean([out[f"auroc_{d}"] for d in dx]))
    out["macro_brier"] = float(np.mean([out[f"brier_{d}"] for d in dx]))
    top = np.argsort(-Pm, axis=1)[:, :top_k]
    hit = Y[np.arange(len(Y))[:, None], top].max(1)
    pos = Y.max(1) > 0
    out[f"top{top_k}_hit"] = float(np.average(hit[pos], weights=w[pos])) if (w[pos] > 0).any() else float("nan")
    return out


def bootstrap(groups: np.ndarray, B: int, seed: int, fn) -> list[dict]:
    """Cluster bootstrap: resample groups (subjects) with replacement; fn(weights) -> metric dict."""
    uniq, gi = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    return [fn(np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[gi].astype(float)) for _ in range(B)]


def with_ci(point: dict, boots: list[dict]) -> dict:
    out = {}
    for k, v in point.items():
        b = np.array([x.get(k, np.nan) for x in boots], float)
        b = b[np.isfinite(b)]
        out[k] = {"value": v, "ci": [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))] if len(b) else [None, None]}
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


# ------------------------------------------------------------------------------------- logistic baseline
def lr_features(M: np.ndarray, kinds: list[str]) -> np.ndarray:
    """Memberships with NaN = unknown -> value (unknown as 0) plus a missing indicator for labs."""
    X = [np.nan_to_num(M, nan=0.0)]
    X += [np.isnan(M[:, [i]]).astype(float) for i, k in enumerate(kinds) if k == "lab"]
    return np.hstack(X)


def fit_lr(cfg: dict, kbdir: Path) -> dict:
    p = kbdir / "lr.json"
    if p.exists():
        return json.loads(p.read_text())
    df = pd.read_parquet(kbdir / "cohort.parquet")
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    ev = kb.evidence(cfg)
    M = df.loc[tr, ev].to_numpy(float, copy=True)
    M[M < 0] = np.nan
    X = lr_features(M, [cfg["propositions"][k]["kind"] for k in ev])
    out = {"evidence": ev, "models": {}}
    for d in dx_names(cfg):
        m = LogisticRegression(max_iter=2000).fit(X, df.loc[tr, d].to_numpy())
        out["models"][d] = {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
    p.write_text(json.dumps(out))
    return out


def lr_predict(lr: dict, cfg: dict, M: np.ndarray) -> np.ndarray:
    X = lr_features(M, [cfg["propositions"][k]["kind"] for k in lr["evidence"]])
    return np.stack([1 / (1 + np.exp(-(X @ np.array(m["coef"]) + m["intercept"]))) for m in lr["models"].values()], 1)


# ------------------------------------------------------------------------------------- MIMIC scale
def eval_kb(cfg: dict, kbdir: Path) -> dict:
    """Held-out MIMIC admissions: learned network vs naive network vs logistic regression, plus how rare the
    full admission profiles are (the regime where conditioning by sampling becomes expensive)."""
    dx, ev = dx_names(cfg), kb.evidence(cfg)
    bn, _ = kb.load(cfg, kbdir)
    naive, _ = kb.load(cfg, kbdir, naive=True)
    df = pd.read_parquet(kbdir / "cohort.parquet")
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    test = df[~tr].reset_index(drop=True)
    Y = test[dx].to_numpy(int)
    preds = {}
    t0 = time.perf_counter()
    preds["learned network"], pe = kb.posteriors_by_pattern(cfg, bn, test)
    secs = time.perf_counter() - t0
    preds["naive network (1 parent)"], _ = kb.posteriors_by_pattern(cfg, naive, test)
    M = test[ev].to_numpy(float, copy=True)
    M[M < 0] = np.nan
    preds["logistic regression"] = lr_predict(fit_lr(cfg, kbdir), cfg, M)
    prev = df.loc[tr, dx].to_numpy(float).mean(0)
    preds["prevalence"] = np.tile(prev, (len(test), 1))
    k, B = cfg["engine"]["top_k"], cfg["evaluation"]["bootstrap_kb"]
    groups = test.subject_id.to_numpy()
    res = {}
    for name, Pm in preds.items():
        pt = metric_set(Y, Pm, dx, k)
        pt.update({f"ece_{d}": ece(Y[:, i], Pm[:, i]) for i, d in enumerate(dx)})
        res[name] = with_ci(pt, bootstrap(groups, B, cfg["seed"], lambda w, Pm=Pm: metric_set(Y, Pm, dx, k, w)) if name != "prevalence" else [])
    E_tr = df.loc[tr, ev].to_numpy(np.int8)
    seen = {r.tobytes() for r in np.unique(E_tr, axis=0)}
    E_te = test[ev].to_numpy(np.int8)
    return {"models": res, "n_test": int(len(test)), "n_test_subjects": int(test.subject_id.nunique()),
            "prevalence_test": dict(zip(dx, Y.mean(0).tolist())),
            "p_evidence_quantiles": dict(zip(["1%", "10%", "50%", "90%", "99%"], np.quantile(pe, [0.01, 0.1, 0.5, 0.9, 0.99]).tolist())),
            "frac_p_evidence_below": {f"{t:g}": float((pe < t).mean()) for t in (1e-2, 1e-3, 1e-4)},
            "frac_profiles_unseen_in_train": float(np.mean([r.tobytes() not in seen for r in E_te])),
            "unique_profiles_test": int(len(np.unique(E_te, axis=0))), "exact_seconds": secs}


# ------------------------------------------------------------------------------------- notes
def make_agent(cfg: dict, cache: Path):
    a = cfg["agent"]
    if a["backend"] == "baseline":
        return None
    client = FakeClient(cfg) if a["backend"] == "fake" else OllamaClient(a["url"], a["model"], a["timeout_s"], a.get("think"), cfg["seed"], num_ctx=a.get("num_ctx"))
    return Agent(client, cfg, cache)


def prepare_notes(cfg: dict) -> list[dict]:
    out = []
    for r in N.load_notes(Path(cfg["paths"]["notes_csv"])):
        y = N.labels(r["text"], cfg)
        if y is None:
            continue
        deid, phi = N.deidentify(r["text"])
        out.append({"note_id": r["note_id"], "subject_id": r["subject_id"], "labels": y, "phi": phi,
                    "raw_evidence": N.evidence_text(r["text"], cfg), "evidence": N.evidence_text(deid, cfg), "deid": deid})
    return out


def run_notes(cfg: dict, kbdir: Path, art: Path) -> dict:
    """Per note: findings (baseline extractor; agent grounding; agent with engine-driven follow-ups),
    memberships, and posteriors for every ablation. Writes the f-string report of the main variant per note."""
    bn, limits = kb.load(cfg, kbdir)
    naive, _ = kb.load(cfg, kbdir, naive=True)
    lr = fit_lr(cfg, kbdir)
    ev_names = kb.evidence(cfg)
    agent = make_agent(cfg, art / "agent_cache" / cfg["agent"]["model"].replace(":", "_").replace("/", "_"))
    notes = prepare_notes(cfg)
    main = "agent" if agent else "baseline"
    (art / "note_reports").mkdir(parents=True, exist_ok=True)
    recs = []
    for i, nt in enumerate(notes):
        t0 = time.perf_counter()
        F = {"baseline": N.baseline_findings(nt["evidence"], cfg)}
        trace, rejected, errors = [], [], []
        if agent:
            r = agent.run(nt["evidence"], bn, limits, loop=True)
            F["agent_ground"], F["agent"], trace, rejected = r["ground_findings"], r["findings"], r["trace"], r["rejected"]
            errors = r["errors"]
        Mv = {v: N.memberships(f, cfg, limits) for v, f in F.items()}
        if agent:
            Mv["agent_crisp"] = N.memberships(F["agent"], cfg, limits, crisp=True)
        post = {v: posterior_exact(bn, Evidence.from_memberships(m), dx_names(cfg))[0] for v, m in Mv.items()}
        post[main + "_naive"] = posterior_exact(naive, Evidence.from_memberships(Mv[main]), dx_names(cfg))[0]
        vec = np.array([[np.nan if Mv[main][k] is None else Mv[main][k] for k in ev_names]])
        post[main + "_logistic"] = dict(zip(dx_names(cfg), lr_predict(lr, cfg, vec)[0].tolist()))
        res = explain(bn, Mv[main], cfg)
        (art / "note_reports" / f"{nt['note_id']}.txt").write_text(report(nt["note_id"], Mv[main], res, cfg, nt["labels"]))
        recs.append({"note_id": nt["note_id"], "subject_id": nt["subject_id"], "labels": nt["labels"],
                     "memberships": Mv, "posteriors": post, "p_evidence": res["p_evidence"], "trace": trace,
                     "rejected": rejected, "llm_errors": errors, "seconds": time.perf_counter() - t0})
        log.info("note %d/%d %s (%.1fs)", i + 1, len(notes), nt["note_id"], recs[-1]["seconds"])
    (art / "notes.json").write_text(json.dumps(recs))
    return {"main": main, "n_notes": len(recs)}


def eval_notes(cfg: dict, art: Path) -> dict:
    recs = json.loads((art / "notes.json").read_text())
    dx, k = dx_names(cfg), cfg["engine"]["top_k"]
    Y = np.array([[r["labels"][d] for d in dx] for r in recs])
    groups = np.array([r["subject_id"] for r in recs])
    variants = list(recs[0]["posteriors"])
    P = {v: np.array([[r["posteriors"][v][d] for d in dx] for r in recs]) for v in variants}
    B, seed = cfg["evaluation"]["bootstrap"], cfg["seed"]
    out = {"n_notes": len(recs), "n_subjects": int(len(set(groups.tolist()))), "label_counts": dict(zip(dx, Y.sum(0).tolist())),
           "variants": {}, "paired": {}}
    for v in variants:
        out["variants"][v] = with_ci(metric_set(Y, P[v], dx, k), bootstrap(groups, B, seed, lambda w, v=v: metric_set(Y, P[v], dx, k, w)))
    ref = "agent" if "agent" in P else "baseline"
    for v in variants:
        if v == ref:
            continue
        def diff(w, v=v):
            a, b = metric_set(Y, P[ref], dx, k, w), metric_set(Y, P[v], dx, k, w)
            return {"macro_auroc": a["macro_auroc"] - b["macro_auroc"], "macro_brier": a["macro_brier"] - b["macro_brier"],
                    f"top{k}_hit": a[f"top{k}_hit"] - b[f"top{k}_hit"]}
        out["paired"][f"{ref} - {v}"] = with_ci(diff(np.ones(len(Y))), bootstrap(groups, B, seed, diff))
    tr = [t for r in recs for t in r["trace"]]
    out["agent_loop"] = {"follow_ups": len(tr), "notes_with_follow_up": sum(bool(r["trace"]) for r in recs),
                         "resolved": sum(t["status"] != "unknown" for t in tr),
                         "changed_membership": sum(bool(t.get("changed")) for t in tr),
                         "searches": sum(len(t["queries"]) for t in tr),
                         "grounding_rejected": sum(len(r["rejected"]) for r in recs),
                         "followup_rejected": sum(bool(t["rejected"]) for t in tr),
                         "llm_errors": sum(len(r.get("llm_errors", [])) for r in recs),
                         "grounding_fallbacks": sum(any(e.startswith("ground") for e in r.get("llm_errors", [])) for r in recs)}
    main = "agent" if "agent" in recs[0]["memberships"] else "baseline"
    graded = [v for r in recs for v in r["memberships"][main].values() if v is not None and 0 < v < 1]
    out["graded_findings"] = {"count": len(graded), "notes": sum(any(v is not None and 0 < v < 1 for v in r["memberships"][main].values()) for r in recs)}
    out["seconds_per_note"] = float(np.mean([r["seconds"] for r in recs]))
    return out


# ------------------------------------------------------------------------------------- engines on notes
def compare_engines(cfg: dict, kbdir: Path, art: Path) -> dict:
    """Same evidence, every engine: error against exact enumeration and cost, for each note's graded evidence."""
    bn, _ = kb.load(cfg, kbdir)
    recs = json.loads((art / "notes.json").read_text())
    main = "agent" if "agent" in recs[0]["memberships"] else "baseline"
    dx = dx_names(cfg)
    rows, checks, skipped = [], [], 0
    for i, r in enumerate(recs):
        ev = Evidence.from_memberships(r["memberships"][main])
        ex, pe, _ = posterior(bn, ev, cfg, "exact")
        q = posterior(bn, ev, cfg, "quantum", seed=cfg["seed"] + i)
        budget = int(min(max(q[2]["oracle_calls"], 1000), cfg["engine"]["budget_samples"]))  # matched cost
        for b, (p, _, c) in (("quantum", q), ("rejection", posterior_rejection(bn, ev, dx, budget, cfg["seed"] + i)),
                             ("lw", posterior_lw(bn, ev, dx, budget, cfg["seed"] + i))):
            err = [abs(p[d] - ex[d]) for d in dx]
            rows.append({"engine": b, "p_evidence": pe, "max_abs_error": float(np.max(err)) if np.all(np.isfinite(err)) else float("nan"),
                         "failed": not np.all(np.isfinite(err)), "cost": c["oracle_calls"],
                         "accepted": c.get("accepted"), "ess": c.get("effective_sample_size"), "seconds": c["seconds"]})
        if i < cfg["evaluation"]["quantum_notes"]:
            if circuit_qubits(bn, ev, dx) <= cfg["evaluation"]["circuit_max_qubits"]:
                checks.append(check_circuit(bn, ev, dx))
            else:
                skipped += 1
    df = pd.DataFrame(rows)
    summ = {}
    for b, g in df.groupby("engine"):
        ok = g[~g.failed]
        summ[b] = {"notes": int(len(g)), "failed": int(g.failed.sum()), "median_max_abs_error": float(ok.max_abs_error.median()) if len(ok) else None,
                   "p90_max_abs_error": float(ok.max_abs_error.quantile(0.9)) if len(ok) else None,
                   "median_cost": float(g.cost.median()), "median_seconds": float(g.seconds.median())}
        if b == "rejection":
            summ[b]["median_accepted"] = float(g.accepted.median())
        if b == "lw":
            summ[b]["median_ess"] = float(g.ess.median())
    rare = df[df.p_evidence < np.quantile(df.p_evidence, 0.25)]
    summ["rarest_quartile"] = {b: float(g.max_abs_error.median()) for b, g in rare.groupby("engine")}
    return {"summary": summ, "p_evidence_median": float(df.p_evidence.median()),
            "circuit_checks": {"notes": len(checks), "skipped_too_large": skipped, "max_abs_error": max((c["max_abs_error"] for c in checks), default=None),
                               "qubits_range": [min((c["qubits"] for c in checks), default=None), max((c["qubits"] for c in checks), default=None)],
                               "prep_depth_max": max((c["prep_depth"] for c in checks), default=None),
                               "prep_two_qubit_gates_max": max((c["prep_two_qubit_gates"] for c in checks), default=None)}}


# ------------------------------------------------------------------------------------- quantum advantage
def _profiles(cfg: dict, bn: BN, test: pd.DataFrame, rng) -> list[dict]:
    """MIMIC test admissions with a growing number of revealed findings; target = most probable diagnosis."""
    a = cfg["evaluation"]["advantage"]
    ev, dx = kb.evidence(cfg), dx_names(cfg)
    rows = test[ev].to_numpy(np.int8)
    full = np.where((rows >= 0).all(1))[0]
    out = []
    for i in range(a["profiles"]):
        r = rows[rng.choice(full)]
        size = a["evidence_sizes"][i % len(a["evidence_sizes"])]
        keep = rng.choice(len(ev), size=min(size, len(ev)), replace=False)
        e = Evidence({ev[j]: int(r[j]) for j in keep}, {})
        post, pe, _ = posterior_exact(bn, e, dx)
        t = max(post, key=post.get)
        out.append({"evidence": e, "target": t, "p": post[t], "p_evidence": pe, "size": int(len(keep))})
    return out


def _pool(bn: BN, n_samples: int, seed: int, chunk: int = 200_000) -> np.ndarray:
    """Prior samples as uint8 [n_samples, n] (any network size; a bit-packed int would overflow past 31 nodes)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, bn.struct.n), np.uint8)
    for i in range(0, n_samples, chunk):
        m = min(chunk, n_samples - i)
        out[i:i + m] = bn.sample(rng.random((m, bn.struct.n)))[0]
    return out


def _first_ok(errs_by_cost: list[tuple[float, np.ndarray]], rel: float):
    for cost, errs in errs_by_cost:
        e = np.where(np.isfinite(errs), errs, np.inf)
        if np.quantile(e, 0.9, method="higher") <= rel:
            return cost
    return None


def advantage(cfg: dict, kbdir: Path) -> dict:
    """Oracle calls (model evaluations) each estimator needs so that 90% of repeats reach relative error
    `rel_error` on P(target | E), as P(E) shrinks. Rejection: ~1/P(E). Amplitude estimation: ~1/sqrt(P(E))."""
    a = cfg["evaluation"]["advantage"]
    rel, R, seed = a["rel_error"], a["repeats"], cfg["seed"]
    bn, _ = kb.load(cfg, kbdir)
    rng = np.random.default_rng(seed)
    profs = _profiles(cfg, bn, kb.test_split(cfg, kbdir), rng)
    nmax = max(a["sample_grid"])
    pools = [_pool(bn, nmax, seed + 100 + r) for r in range(R)]
    rows = []
    for pi, pr in enumerate(profs):
        ev, t, p = pr["evidence"], pr["target"], pr["p"]
        cols = [bn.struct.index(k) for k in ev.hard]
        vals = np.array([ev.hard[k] for k in ev.hard], np.uint8)
        ti = bn.struct.index(t)
        rej = []
        accs = [np.all(pool[:, cols] == vals, axis=1) for pool in pools]
        for N_ in a["sample_grid"]:
            errs = []
            for pool, acc_all in zip(pools, accs):
                acc = acc_all[:N_]
                errs.append(abs(pool[:N_][acc, ti].mean() - p) / p if acc.any() else np.inf)
            rej.append((N_, np.array(errs)))
        lw = []
        for N_ in [g for g in a["sample_grid"] if g <= a["lw_max"]]:
            errs = np.array([abs(posterior_lw(bn, ev, [t], N_, seed + 1000 * r + pi)[0][t] - p) / p for r in range(R)])
            lw.append((N_, errs))
            if np.quantile(errs, 0.9, method="higher") <= rel:
                break
        aE, aT = pr["p_evidence"], pr["p_evidence"] * p
        iae = []
        for f in a["rel_factors"]:
            errs, calls = [], []
            for r in range(R):
                s = seed + 7919 * r + pi
                e1 = iae_relative(lambda e: iae_emulated(aE, e, 0.05, cfg["engine"]["shots"], s), cfg["engine"]["eps"], f * rel)
                e2 = iae_relative(lambda e: iae_emulated(aT, e, 0.05, cfg["engine"]["shots"], s + 1), cfg["engine"]["eps"], f * rel)
                errs.append(abs(e2["estimate"] / e1["estimate"] - p) / p if e1["estimate"] > 0 else np.inf)
                calls.append(e1["oracle_queries"] + e2["oracle_queries"])
            iae.append((float(np.mean(calls)), np.array(errs)))
            if np.quantile(np.array(errs), 0.9, method="higher") <= rel:
                break
        rows.append({"profile": pi, "size": pr["size"], "p_evidence": pr["p_evidence"], "target": t, "p": p,
                     "rejection": _first_ok(rej, rel), "lw": _first_ok(lw, rel), "iae": _first_ok(iae, rel)})
    df = pd.DataFrame(rows)
    slopes = {}
    for m in ("rejection", "lw", "iae"):
        g = df[df[m].notna()]
        slopes[m] = {"achieved": int(len(g)), "censored": int(df[m].isna().sum()),
                     "slope": float(np.polyfit(np.log10(1 / g.p_evidence), np.log10(g[m].astype(float)), 1)[0]) if len(g) >= 3 else None}
    fits = {m: np.polyfit(np.log10(1 / df[df[m].notna()].p_evidence), np.log10(df[df[m].notna()][m].astype(float)), 1)
            for m in ("rejection", "iae") if df[m].notna().sum() >= 3}
    cross = None
    if len(fits) == 2 and fits["rejection"][0] > fits["iae"][0]:
        cross = float(10 ** -((fits["iae"][1] - fits["rejection"][1]) / (fits["rejection"][0] - fits["iae"][0])))
    both = df[df.rejection.notna() & df.iae.notna()]
    return {"profiles": rows, "slopes": slopes, "exact_model_evaluations": int(bn.struct.S), "crossover_p_evidence": cross,
            "rel_error": rel, "max_samples": int(nmax),
            "median_ratio_rejection_over_iae": float((both.rejection / both.iae).median()) if len(both) else None,
            "rarest_decile_ratio": float((both.rejection / both.iae)[both.p_evidence <= both.p_evidence.quantile(0.1)].median()) if len(both) >= 10 else None,
            "_profiles_obj": profs}


def circuit_validation(cfg: dict, kbdir: Path, profs: list[dict]) -> dict:
    """Real Grover circuits (Aer, ideal and depolarizing) on the most likely profiles vs the analytic emulation."""
    bn, _ = kb.load(cfg, kbdir)
    a, nz = cfg["evaluation"]["advantage"], cfg["evaluation"]["noise"]
    order = sorted(profs, key=lambda p: -p["p_evidence"])
    cap = cfg["evaluation"]["circuit_max_qubits"]  # gate-level simulation cost doubles per qubit
    fits = [p for p in order if circuit_qubits(bn, p["evidence"], [p["target"]]) <= cap]
    out = {"ideal": [], "noise": [], "max_qubits": cap, "profiles_too_large": len(order) - len(fits)}
    order = fits
    for i, pr in enumerate(order[: a["circuit_check_profiles"]]):
        t = pr["target"]
        smp, pm = make_sampler("aer", cfg["engine"]["shots"], cfg["seed"] + i)
        c, _, cc = posterior_quantum(bn, pr["evidence"], [t], nz["eps"], 0.05, cfg["engine"]["shots"], cfg["seed"] + i, "circuit", smp, pm)
        e, _, ce = posterior_quantum(bn, pr["evidence"], [t], nz["eps"], 0.05, cfg["engine"]["shots"], cfg["seed"] + i, "analytic")
        out["ideal"].append({"p_evidence": pr["p_evidence"], "exact": pr["p"], "circuit": c[t], "analytic": e[t],
                             "circuit_calls": cc["oracle_calls"], "analytic_calls": ce["oracle_calls"],
                             "qubits": cc["qubits"], "seconds": cc["seconds"]})
    for i, pr in enumerate(order[: nz["profiles"]]):
        t = pr["target"]
        for p2 in nz["depolarizing_2q"]:
            smp, pm = make_sampler("depolarizing", cfg["engine"]["shots"], cfg["seed"] + i, p2)
            c, _, cc = posterior_quantum(bn, pr["evidence"], [t], nz["eps"], 0.05, cfg["engine"]["shots"], cfg["seed"] + i, "circuit", smp, pm)
            out["noise"].append({"p2": p2, "p_evidence": pr["p_evidence"], "exact": pr["p"], "estimate": c[t],
                                 "abs_error": abs(c[t] - pr["p"]), "oracle_calls": cc["oracle_calls"]})
    return out


# ------------------------------------------------------------------------------------- privacy
def privacy(cfg: dict, art: Path) -> dict:
    """PHI exposure: header PHI values and name tokens found in (a) the text the agent would receive without the
    de-identification gate, (b) the text it does receive, (c) the generated reports."""
    notes = prepare_notes(cfg)
    rep = art / "note_reports"
    gate_off = [N.leaked(n["raw_evidence"], n["phi"]) for n in notes]
    gate_on = [N.leaked(n["evidence"], n["phi"]) for n in notes]
    whole = [N.leaked(n["deid"], n["phi"]) for n in notes]
    reps = [N.leaked((rep / f"{n['note_id']}.txt").read_text(), n["phi"]) for n in notes if (rep / f"{n['note_id']}.txt").exists()]
    items = sum(len(n["phi"]["values"]) + len(n["phi"]["names"]) for n in notes)

    def agg(x):
        return {"notes_with_leak": sum(bool(v) for v in x), "items_leaked": sum(len(v) for v in x)}
    return {"notes": len(notes), "phi_items": items, "agent_input_gate_off": agg(gate_off), "agent_input_gate_on": agg(gate_on),
            "deidentified_note": agg(whole), "reports": {"checked": len(reps), **agg(reps)}}
