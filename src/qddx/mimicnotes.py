"""Held-out evaluation on MIMIC-IV-Note discharge summaries (redacted, linked to MIMIC-IV hosp by hadm_id).

Eligible notes: admissions in the knowledge base's TEST split (subject-level), with a Discharge Diagnosis section,
excluding every patient of the 200-note development file (whose notes were used for the audit-informed fixes).
The protocol (prompts, vocabulary, thresholds, network) is frozen; the agent sample is drawn once with a fixed
seed and written to disk before any agent output exists.

Per admission there are two ground truths (the note's Discharge Diagnosis section; the admission's ICD codes, i.e.
the knowledge base's own definitions) and two evidence sources (findings read from the note; the admission's coded
history, first-24 h medications and first labs), so note reading can be compared with structured data directly.
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import kb
from . import notes as N
from .bn import Evidence, posterior_exact
from .engine import dx_names, platt
from .agent import ollama_info
from .evaluate import bootstrap, compare_engines, fit_lr, lr_predict, make_agent, metric_set, with_ci

log = logging.getLogger("qddx")


def _stream(path: Path):
    csv.field_size_limit(1 << 30)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        yield from csv.DictReader(f)


class Posteriors:
    """Exact P(diagnosis | memberships), cached by membership pattern; optional per-diagnosis calibration
    (vocabulary option engine.calibration, fitted on validation patients)."""

    def __init__(self, bn, dx, calibration: dict | None = None):
        self.bn, self.dx, self.cache, self.cal = bn, dx, {}, calibration

    def __call__(self, m: dict) -> list[float]:
        key = tuple(sorted((k, v) for k, v in m.items() if v is not None))
        if key not in self.cache:
            p, _, _ = posterior_exact(self.bn, Evidence.from_memberships(m), self.dx)
            out = [p[d] for d in self.dx]
            if self.cal:
                out = [float(platt(v, *self.cal[d])) if d in self.cal else v for v, d in zip(out, self.dx)]
            self.cache[key] = out
        return self.cache[key]


def structured_of(cfg: dict, row) -> dict:
    """Admission-record values for demographic propositions (from the kb_ columns of an eligible-notes row)."""
    return {k: int(row[f"kb_{k}"]) for k in kb.props(cfg, "demographic")}


def _crisp(m: dict) -> dict:
    return {k: None if v is None else float(v >= 0.5) for k, v in m.items()}


def _vec(ms: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[np.nan if m[k] is None else m[k] for k in names] for m in ms], float)


def scan(cfg: dict, kbdir: Path) -> tuple[pd.DataFrame, dict]:
    """Pass over discharge.csv.gz: eligibility, note labels, extractor memberships. -> (table, counts)."""
    mc = cfg["mimic_notes"]
    cohort = pd.read_parquet(kbdir / "cohort.parquet")
    test = kb.test_split(cfg, kbdir).set_index("hadm_id")
    dev = {r["subject_id"] for r in _stream(Path(cfg["paths"]["notes_csv"]))} if mc.get("exclude_dev_patients", True) else set()
    _, limits = kb.load(cfg, kbdir)
    ev_names = kb.evidence(cfg)
    counts = {"notes_in_file": 0, "not_in_kb": 0, "kb_train_split": 0, "dev_patient": 0, "no_discharge_diagnosis": 0, "eligible": 0}
    train_hadm = set(cohort.hadm_id.tolist()) - set(test.index.tolist())
    rows = []
    t0 = time.perf_counter()
    for r in _stream(Path(cfg["paths"]["mimic_notes"])):
        counts["notes_in_file"] += 1
        try:
            h = int(r["hadm_id"])
        except ValueError:
            counts["not_in_kb"] += 1
            continue
        if h not in test.index:
            counts["kb_train_split" if h in train_hadm else "not_in_kb"] += 1
            continue
        if r["subject_id"] in dev:
            counts["dev_patient"] += 1
            continue
        y = N.labels(r["text"], cfg)
        if y is None:
            counts["no_discharge_diagnosis"] += 1
            continue
        ev = N.evidence_text(N.deidentify(r["text"])[0], cfg)
        demo = {k: int(test.at[h, k]) for k in kb.props(cfg, "demographic")}
        m = N.memberships(N.baseline_findings(ev, cfg), cfg, limits, structured=demo)
        rows.append({"note_id": r["note_id"], "subject_id": int(r["subject_id"]), "hadm_id": h,
                     **{f"note_{d}": v for d, v in y.items()}, **{f"m_{k}": m[k] for k in ev_names}})
        counts["eligible"] += 1
        if counts["eligible"] % 5000 == 0:
            log.info("scan: %d eligible notes (%.0fs)", counts["eligible"], time.perf_counter() - t0)
    df = pd.DataFrame(rows)
    st = test.loc[df.hadm_id.to_numpy()]
    for c in kb.props(cfg):
        df[f"kb_{c}"] = st[c].to_numpy()
    return df, counts


def agent_sample(cfg: dict, df: pd.DataFrame, path: Path) -> list[str]:
    """Fixed random sample of eligible note_ids for the agent; frozen on first use."""
    if path.exists():
        return path.read_text().split()
    ids = sorted(df.note_id.tolist())
    n = min(int(cfg["mimic_notes"]["agent_sample"]), len(ids))
    pick = sorted(np.random.default_rng(cfg["seed"] + 17).choice(ids, size=n, replace=False).tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pick))
    return pick


def run_agent(cfg: dict, kbdir: Path, art: Path, ids: list[str]) -> dict:
    """Agent findings for the sampled notes (second pass over the file; outputs cached per note)."""
    agent = make_agent(cfg, art / "agent_cache" / cfg["agent"]["model"].replace(":", "_").replace("/", "_"))
    if agent is None:
        return {}
    bn, limits = kb.load(cfg, kbdir)
    want, out = set(ids), {}
    t0 = time.perf_counter()
    for r in _stream(Path(cfg["paths"]["mimic_notes"])):
        if r["note_id"] not in want:
            continue
        ev = N.evidence_text(N.deidentify(r["text"])[0], cfg)
        res = agent.run(ev, bn, limits, loop=True)
        out[r["note_id"]] = {"ground_findings": res["ground_findings"], "findings": res["findings"],
                             "trace": res["trace"], "rejected": res["rejected"], "errors": res["errors"]}
        done = len(out)
        if done % 25 == 0 or done == len(want):
            el = time.perf_counter() - t0
            log.info("agent: %d/%d notes, %.1fs/note, ~%.0f min left", done, len(want), el / done, el / done * (len(want) - done) / 60)
        if done == len(want):
            break
    return out


def _evaluate(cfg, P: dict[str, np.ndarray], Y: np.ndarray, groups: np.ndarray, pairs: list[tuple[str, str]], B: int) -> dict:
    dx, k = dx_names(cfg), cfg["engine"]["top_k"]
    res = {"n": int(len(Y)), "patients": int(len(set(groups.tolist()))), "positives": dict(zip(dx, Y.sum(0).astype(int).tolist())),
           "variants": {}, "paired": {}}
    for v, Pm in P.items():
        res["variants"][v] = with_ci(metric_set(Y, Pm, dx, k), bootstrap(groups, B, cfg["seed"], lambda w, Pm=Pm: metric_set(Y, Pm, dx, k, w)))
    for a, b in pairs:
        if a in P and b in P:
            def diff(w, a=a, b=b):
                x, z = metric_set(Y, P[a], dx, k, w), metric_set(Y, P[b], dx, k, w)
                return {m: x[m] - z[m] for m in ("macro_auroc", "macro_brier", f"top{k}_hit")}
            res["paired"][f"{a} - {b}"] = with_ci(diff(np.ones(len(Y))), bootstrap(groups, B, cfg["seed"], diff))
    return res


def label_agreement(df: pd.DataFrame, dx: list[str]) -> dict:
    out = {}
    for d in dx:
        a, b = df[f"note_{d}"].to_numpy(), df[f"kb_{d}"].to_numpy()
        po = float((a == b).mean())
        pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        out[d] = {"note_positive": int(a.sum()), "icd_positive": int(b.sum()), "agreement": po,
                  "kappa": (po - pe) / (1 - pe) if pe < 1 else 1.0}
    return out


def run(cfg: dict, kbdir: Path, art: Path) -> dict:
    mc, dx = cfg["mimic_notes"], dx_names(cfg)
    out = art / "mimic_notes"
    out.mkdir(parents=True, exist_ok=True)
    table = out / "eligible.parquet"
    if table.exists():
        df, counts = pd.read_parquet(table), json.loads((out / "counts.json").read_text())
    else:
        df, counts = scan(cfg, kbdir)
        df.to_parquet(table)
        (out / "counts.json").write_text(json.dumps(counts))
    if mc.get("max_extractor"):
        df = df.sample(n=min(int(mc["max_extractor"]), len(df)), random_state=cfg["seed"]).sort_values("note_id")
    df = df.reset_index(drop=True)
    bn, limits = kb.load(cfg, kbdir)
    naive, _ = kb.load(cfg, kbdir, naive=True)
    lr = fit_lr(cfg, kbdir)
    ev_names = kb.evidence(cfg)
    cal = cfg["engine"].get("calibration")
    post, post_naive = Posteriors(bn, dx, cal), Posteriors(naive, dx)
    M_ext = [{k: (None if pd.isna(r[f"m_{k}"]) else float(r[f"m_{k}"])) for k in ev_names} for _, r in df.iterrows()]
    t0 = time.perf_counter()
    P_all = {"extractor": np.array([post(m) for m in M_ext]),
             "extractor_crisp": np.array([post(_crisp(m)) for m in M_ext]),
             "extractor_naive": np.array([post_naive(m) for m in M_ext]),
             "extractor_logistic": lr_predict(lr, cfg, _vec(M_ext, ev_names))}
    S = df[[f"kb_{c}" for c in kb.props(cfg)]].copy()
    S.columns = kb.props(cfg)
    P_all["structured"], _ = kb.posteriors_by_pattern(cfg, bn, S)
    if cal:
        P_all["structured"] = np.stack([platt(P_all["structured"][:, i], *cal[d]) for i, d in enumerate(dx)], 1)
    Ms = S[ev_names].to_numpy(float, copy=True)
    Ms[Ms < 0] = np.nan
    P_all["structured_logistic"] = lr_predict(lr, cfg, Ms)
    log.info("posteriors for %d notes in %.0fs", len(df), time.perf_counter() - t0)
    groups = df.subject_id.to_numpy()
    Y = {"note": df[[f"note_{d}" for d in dx]].to_numpy(int), "icd": df[[f"kb_{d}" for d in dx]].to_numpy(int)}
    pairs_all = [("extractor", "extractor_crisp"), ("extractor", "extractor_naive"), ("extractor", "extractor_logistic"),
                 ("structured", "extractor"), ("structured", "structured_logistic")]
    B = int(mc["bootstrap"])
    result = {"counts": counts, "label_agreement": label_agreement(df, dx),
              "all": {t: _evaluate(cfg, P_all, Y[t], groups, pairs_all, B) for t in Y}}

    # agent on the frozen random sample: paired with the extractor and structured data on the same admissions
    ids = agent_sample(cfg, df, out / "agent_sample.txt")
    A = run_agent(cfg, kbdir, art, ids)
    if A:
        idx = df.index[df.note_id.isin(A)].to_numpy()
        sub = df.loc[idx]
        Ma = [N.memberships(A[n]["findings"], cfg, limits, structured=structured_of(cfg, r)) for n, (_, r) in zip(sub.note_id, sub.iterrows())]
        Mg = [N.memberships(A[n]["ground_findings"], cfg, limits, structured=structured_of(cfg, r)) for n, (_, r) in zip(sub.note_id, sub.iterrows())]
        P_s = {v: P_all[v][idx] for v in ("extractor", "structured")}
        P_s.update({"agent": np.array([post(m) for m in Ma]), "agent_ground": np.array([post(m) for m in Mg]),
                    "agent_crisp": np.array([post(_crisp(m)) for m in Ma]), "agent_naive": np.array([post_naive(m) for m in Ma]),
                    "agent_logistic": lr_predict(lr, cfg, _vec(Ma, ev_names))})
        pairs_s = [("agent", "extractor"), ("agent", "agent_ground"), ("agent", "agent_crisp"), ("agent", "agent_naive"),
                   ("agent", "agent_logistic"), ("structured", "agent")]
        gs = sub.subject_id.to_numpy()
        Ys = {t: Y[t][idx] for t in Y}
        result["agent_sample"] = {t: _evaluate(cfg, P_s, Ys[t], gs, pairs_s, B) for t in Ys}
        tr = [t for a in A.values() for t in a["trace"]]
        result["agent_loop"] = {"notes": len(A), "follow_ups": len(tr), "answered": sum(t["status"] != "unknown" for t in tr),
                                "changed_membership": sum(bool(t.get("changed")) for t in tr),
                                "searches": sum(len(t["queries"]) for t in tr),
                                "grounding_rejected": sum(len(a["rejected"]) for a in A.values()),
                                "followup_rejected": sum(bool(t["rejected"]) for t in tr),
                                "llm_errors": sum(len(a["errors"]) for a in A.values()),
                                "grounding_fallbacks": sum(any(e.startswith("ground") for e in a["errors"]) for a in A.values())}
        # matched-cost engine comparison on a fixed subset of the sample (reuses the 200-note machinery)
        eng = out / "engines"
        eng.mkdir(exist_ok=True)
        keep = list(sub.note_id)[: int(mc["engines_sample"])]
        (eng / "notes.json").write_text(json.dumps([{"note_id": n, "memberships": {"agent": m}} for n, m in zip(sub.note_id, Ma)][: len(keep)]))
        result["engines"] = compare_engines(cfg, kbdir, eng)
    if cfg["agent"]["backend"] == "llm":
        result["ollama"] = ollama_info(cfg["agent"]["url"], [cfg["agent"]["model"]])
    write_md(result, cfg, Path(cfg["paths"]["reports"]) / "mimic_notes.md")
    return result


def write_md(R: dict, cfg: dict, path: Path) -> Path:
    from .report import _ci, _f
    P = cfg["propositions"]
    dx, k = dx_names(cfg), cfg["engine"]["top_k"]
    c = R["counts"]
    names = {"extractor": "keyword extractor (note)", "extractor_crisp": "extractor, crisp findings",
             "extractor_naive": "extractor, naive network", "extractor_logistic": "extractor findings, logistic regression",
             "structured": "structured MIMIC data, learned network", "structured_logistic": "structured data, logistic regression",
             "agent": "agent with follow-ups (note)", "agent_ground": "agent, grounding only", "agent_crisp": "agent, crisp findings",
             "agent_naive": "agent, naive network", "agent_logistic": "agent findings, logistic regression"}
    L = ["# Held-out test: MIMIC-IV-Note discharge summaries", "",
         "Frozen protocol (prompts, vocabulary, thresholds and network unchanged). Notes come from patients in the knowledge "
         "base's held-out test split only; every patient of the 200-note development file is excluded. Notes are MIMIC's "
         "redacted versions. Aggregates only.", "",
         f"Notes in file {c['notes_in_file']:,}; knowledge-base training split {c['kb_train_split']:,}; not in the adult "
         f"cohort {c['not_in_kb']:,}; development-file patients {c['dev_patient']:,}; no Discharge Diagnosis section "
         f"{c['no_discharge_diagnosis']:,}; **eligible {c['eligible']:,}**.", "",
         "## Ground truth agreement: note section vs ICD codes (same admission)", "",
         "| diagnosis | note positives | ICD positives | agreement | kappa |", "|---|---|---|---|---|"]
    L += [f"| {P[d]['label']} | {a['note_positive']:,} | {a['icd_positive']:,} | {a['agreement']:.3f} | {a['kappa']:.2f} |"
          for d, a in R["label_agreement"].items()]

    def block(title, res):
        out = []
        for t, lab in (("note", "Discharge Diagnosis section"), ("icd", "ICD codes")):
            r = res[t]
            out += ["", f"### {title}: ground truth = {lab}", "",
                    f"{r['n']:,} notes, {r['patients']:,} patients; positives " + ", ".join(f"{P[d]['label']} {v:,}" for d, v in r["positives"].items())
                    + f". 95% CIs: patient bootstrap ({cfg['mimic_notes']['bootstrap']}).", "",
                    f"| variant | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
            out += [f"| {names.get(v, v)} | {_ci(m['macro_auroc'])} | {_ci(m['macro_brier'], '.4f')} | {_ci(m[f'top{k}_hit'])} |"
                    for v, m in r["variants"].items()]
            if r["paired"]:
                out += ["", f"| paired difference | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
                out += [f"| {p} | {_ci(m['macro_auroc'], '+.3f')} | {_ci(m['macro_brier'], '+.4f')} | {_ci(m[f'top{k}_hit'], '+.3f')} |"
                        for p, m in r["paired"].items()]
        return out

    L += ["", "## All eligible notes"] + block("All eligible notes", R["all"])
    if "agent_sample" in R:
        a = R["agent_loop"]
        L += ["", f"## Agent on a fixed random sample ({a['notes']:,} notes, seed {cfg['seed']} + 17, drawn before any agent run)"]
        L += block("Agent sample", R["agent_sample"])
        L += ["", f"Agent: {a['follow_ups']:,} follow-ups ({a['answered']:,} answered, {a['changed_membership']:,} changed a finding, "
              f"{a['searches']:,} searches); {a['grounding_rejected']:,} grounding findings and {a['followup_rejected']:,} follow-up "
              f"answers discarded by validation; LLM errors after retries {a['llm_errors']:,} (grounding fallbacks {a['grounding_fallbacks']:,})."]
        e = R.get("engines")
        if e:
            L += ["", f"## Engines at matched cost ({e['summary']['quantum']['notes']} sampled notes)", "",
                  "| engine | failed | median error | 90th pct error | median model evaluations |", "|---|---|---|---|---|"]
            for b in ("rejection", "lw", "quantum"):
                m = e["summary"][b]
                L.append(f"| {b} | {m['failed']}/{m['notes']} | {_f(m['median_max_abs_error'])} | {_f(m['p90_max_abs_error'])} | {m['median_cost']:,.0f} |")
            cc = e["circuit_checks"]
            L.append(f"\nGate-level check on {cc['notes']} notes: largest |circuit amplitude - exact| = {_f(cc['max_abs_error'], '.1e')}.")
    L += ["", "Limitations: discharge summaries are written at discharge; only admission-time sections are read, but those "
          "sections may still reflect hindsight. ICD codes are billing labels. Not for clinical use."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path
