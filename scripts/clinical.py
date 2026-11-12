"""Clinical paper: quote filter diagnosis, inter-model agreement, structured experiments (tasks/plan_clinical_experiments.md).
Local data only; prints aggregates only. Patient-level outputs stay in artifacts/clinical/ (never committed).

  python scripts/clinical.py texts                  # evidence text of the frozen 1,500-note agent sample (local parquet)
  python scripts/clinical.py map MODEL              # cache file <-> (note, proposition) for one panel model
  python scripts/clinical.py quotes                 # why follow-up quotes were rejected (raw answers cached for those)
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qddx import agent as AG
from qddx import kb
from qddx.cli import load_config

ART, RES = Path("artifacts/clinical"), Path("artifacts/v2.1/results")
CACHE = Path("artifacts/v2.1/agent_cache")
ART.mkdir(parents=True, exist_ok=True)


def cfg21() -> dict:
    return load_config("configs/qddx.yaml", [], "configs/vocab_v2.1.json")


def _slug(m: str) -> str:
    return m.replace(":", "_").replace("/", "_")


def models() -> dict:
    return json.loads((RES / "agent-models.json").read_text())["models"]


def texts():
    from qddx.experiments import sample_texts
    t0 = time.perf_counter()
    t = sample_texts(cfg21(), Path("artifacts/v2.1"))
    pd.DataFrame({"note_id": list(t), "text": list(t.values())}).to_parquet(ART / "sample_texts.parquet")
    print(f"{len(t)} notes ({time.perf_counter() - t0:.0f}s)")


def _key(model: str, note: str, *extra) -> str:
    return hashlib.sha256("|".join([AG.PROMPT_VERSION, model, note, *extra]).encode()).hexdigest()[:24]


def map_(model: str):
    cfg = cfg21()
    info = models()[model]
    text = info.get("protocol", "tools") == "text"
    props = [k for k, p in cfg["propositions"].items() if p["kind"] not in ("diagnosis", "demographic")]
    T = pd.read_parquet(ART / "sample_texts.parquet")
    d = CACHE / _slug(model)
    rows = []
    for nid, note in zip(T.note_id, T.text):
        g = _key(model, note, "ground")
        rows.append((nid, "ground", "", (d / f"{g}.json").exists(), g))
        for k in props:
            extra = ("investigate", AG.FOLLOWUP_VERSION, k) + (("text",) if text else ())
            f = _key(model, note, *extra)
            if (d / f"{f}.json").exists():
                rows.append((nid, "followup", k, True, f))
    m = pd.DataFrame(rows, columns=["note_id", "kind", "prop", "cached", "key"])
    m.to_parquet(ART / f"map_{_slug(model)}.parquet")
    files = len(list(d.glob("*.json")))
    print(f"{model}: ground cached {int(m[m.kind == 'ground'].cached.sum())}/1500, follow-ups mapped {int((m.kind == 'followup').sum())}, "
          f"cache files {files}, mapped {int(m.cached.sum())}")


_QUOTES = "\"'“”‘’`«»"


def _variants(q: str) -> dict:
    """The quote under progressively looser normalisations (each a candidate fix for the filter)."""
    n = AG._norm(q)
    v = {"as_is": n}
    v["strip_wrapping_quotes"] = AG._norm(q.strip().strip(_QUOTES))
    v["no_ellipsis"] = AG._norm(re.split(r"\.\.\.|…", v["strip_wrapping_quotes"])[0])
    v["alnum_only"] = " ".join(re.findall(r"[a-z0-9]+", n))
    return v


def quotes():
    cfg = cfg21()
    T = pd.read_parquet(ART / "sample_texts.parquet").set_index("note_id").text
    out = {}
    for model in models():
        mp = ART / f"map_{_slug(model)}.parquet"
        if not mp.exists():
            continue
        m = pd.read_parquet(mp)
        m = m[(m.kind == "followup") & m.cached]
        c = {"rejected_with_raw": 0, "quote_not_in_note": 0}
        cls = {}
        for nid, k, key in zip(m.note_id, m.prop, m.key):
            e = json.loads((CACHE / _slug(model) / f"{key}.json").read_text())
            if e.get("rejected") != "quote not in note" or "raw" not in e:
                continue
            c["quote_not_in_note"] += 1
            f = AG.parse_json(e["raw"])
            q = str(f.get("quote") or "") if isinstance(f, dict) else ""
            note = T[nid]
            nn, na = AG._norm(note), " ".join(re.findall(r"[a-z0-9]+", note.lower()))
            if not q.strip():
                lab = "empty quote"
            elif not isinstance(f, dict) or not f:
                lab = "answer not parseable"
            else:
                v = _variants(q)
                if v["strip_wrapping_quotes"] and v["strip_wrapping_quotes"] in nn:
                    lab = "wrapped in quotation marks"
                elif v["no_ellipsis"] and len(v["no_ellipsis"]) >= 8 and v["no_ellipsis"] in nn:
                    lab = "elided with ..."
                elif v["alnum_only"] and v["alnum_only"] in na:
                    lab = "punctuation or spacing differs"
                elif e["raw"].endswith(q[-20:]) is False and len(e["raw"]) >= 300:
                    lab = "answer truncated at 300 chars"
                else:
                    words = re.findall(r"[a-z0-9]+", q.lower())
                    hit = np.mean([w in na.split() for w in words]) if words else 0
                    lab = "paraphrase (most words in note)" if hit >= 0.8 else "not in note (fabricated or other text)"
            cls[lab] = cls.get(lab, 0) + 1
        out[model] = {**c, "classes": cls}
        print(model, out[model], flush=True)
    (ART / "quote_diagnosis.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- C1 diagnosis sample (re-run, raw saved)
PROBE_N, OLLAMA = 40, os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")  # local Ollama server


def probe_ids() -> list[str]:
    T = pd.read_parquet(ART / "sample_texts.parquet")
    return sorted(np.random.default_rng(2026).choice(T.note_id.to_numpy(), PROBE_N, replace=False).tolist())


class _Recorder:
    """Wraps the Ollama client and keeps every raw reply (the frozen pipeline discards rejected quotes).
    Each reply is also cached on disk by request hash, so a note cut off by the call limit resumes where it stopped
    (decoding is deterministic at temperature 0 with a fixed seed, so a cached reply is the reply a re-run gives)."""
    def __init__(self, client, cache=None):
        self.client, self.model, self.raw, self.cache = client, client.model, [], cache

    def chat(self, *a, **k):
        import hashlib
        f = None
        if self.cache is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
            f = self.cache / (hashlib.sha1(json.dumps([a, k], sort_keys=True, default=str).encode()).hexdigest() + ".json")
            if f.exists():
                m = json.loads(f.read_text())
                self.raw.append(m.get("content", ""))
                return m
        m = self.client.chat(*a, **k)
        if f is not None:
            f.write_text(json.dumps(m, default=str))
        self.raw.append(m.get("content", ""))
        return m


def probe(model: str, budget: str = "150"):
    """First pass re-run on the fixed 40-note sample at the frozen prompt and settings; raw replies saved locally."""
    cfg = cfg21()
    info = models()[model]
    out = ART / "probe" / _slug(model)
    out.mkdir(parents=True, exist_ok=True)
    T = pd.read_parquet(ART / "sample_texts.parquet").set_index("note_id").text
    a = cfg["agent"]
    client = AG.OllamaClient(OLLAMA, model, a["timeout_s"], info.get("think"), cfg["seed"], num_ctx=a.get("num_ctx"))
    t0, n = time.perf_counter(), 0
    for nid in probe_ids():
        f = out / f"{nid}.json"
        if f.exists():
            continue
        if time.perf_counter() - t0 > float(budget):
            break
        rec = _Recorder(client, out / "_partial" / nid)
        ag = AG.Agent(rec, AG_cfg(cfg, model), None)
        res = ag.ground(T[nid])
        f.write_text(json.dumps({"raw": rec.raw, "findings": res["findings"], "rejected": res["rejected"], "error": res.get("error")}))
        n += 1
    left = sum(not (out / f"{x}.json").exists() for x in probe_ids())
    print(f"{model}: {n} notes this call, {left} left ({time.perf_counter() - t0:.0f}s)")


def AG_cfg(cfg: dict, model: str) -> dict:
    c = json.loads(json.dumps(cfg))
    c["agent"]["model"] = model
    return c


# ------------------------------------------------------------------------------------- re-score with and without the filter
def _ground(model: str) -> dict:
    """note_id -> cached grounding entry (validated findings + rejection log)."""
    m = pd.read_parquet(ART / f"map_{_slug(model)}.parquet")
    m = m[(m.kind == "ground") & m.cached]
    return {nid: json.loads((CACHE / _slug(model) / f"{k}.json").read_text()) for nid, k in zip(m.note_id, m.key)}


def _unfiltered(entry: dict, P: dict) -> dict:
    """What the model claimed, with the quote filter switched off where the claim is recoverable from the
    rejection log: 'quote not in note' on history/medication was a present/possible claim (counted as present);
    'vocabulary mention conflicts with absent' was an absent claim. Rejected lab claims stay unknown (the value
    was not cached)."""
    f = dict(entry["findings"])
    for r in entry["rejected"]:
        k = r.get("id")
        if k not in P or (k in f and f[k]["status"] != "unknown"):
            continue
        if r["reason"] == "quote not in note" and P[k]["kind"] in ("history", "medication"):
            f[k] = {"status": "present", "value": None, "quote": ""}
        elif r["reason"] == "vocabulary mention conflicts with absent":
            f[k] = {"status": "absent", "value": None, "quote": ""}
    return f


def predictions(model: str):
    from qddx import notes as N
    from qddx.experiments import _eligible
    from qddx.mimicnotes import Posteriors, structured_of
    from qddx.engine import dx_names
    cfg = cfg21()
    bn, limits = kb.load(cfg, Path("artifacts/v2.1/kb"))
    post = Posteriors(bn, dx_names(cfg), cfg["engine"].get("calibration"))
    T = pd.read_parquet(ART / "sample_texts.parquet")
    sub = _eligible(Path("artifacts/v2.1")).set_index("note_id").loc[list(T.note_id)].reset_index()
    SD = {r["note_id"]: structured_of(cfg, r) for _, r in sub.iterrows()}
    G = _ground(model)
    empty = {"findings": {}, "rejected": []}
    on = np.array([post(N.memberships(G.get(n, empty)["findings"], cfg, limits, structured=SD[n])) for n in sub.note_id])
    off = np.array([post(N.memberships(_unfiltered(G.get(n, empty), cfg["propositions"]), cfg, limits, structured=SD[n])) for n in sub.note_id])
    np.savez(ART / f"pred_{_slug(model)}.npz", on=on, off=off, notes=np.array(sub.note_id))
    print(model, "predictions saved", flush=True)


def rescore():
    from qddx.experiments import _eligible, _truths
    from qddx.mimicnotes import _evaluate
    cfg = cfg21()
    T = pd.read_parquet(ART / "sample_texts.parquet")
    sub = _eligible(Path("artifacts/v2.1")).set_index("note_id").loc[list(T.note_id)].reset_index()
    Y = _truths(cfg, sub)
    ref = json.loads((RES / "agent-models.json").read_text())["metrics"]
    P, pairs, check = {}, [], {}
    for model in models():
        z = np.load(ART / f"pred_{_slug(model)}.npz", allow_pickle=True)
        P[f"{model}: filter on"], P[f"{model}: filter off"] = z["on"], z["off"]
        pairs.append((f"{model}: filter off", f"{model}: filter on"))
    B = int(cfg["experiments"]["bootstrap"])
    out = {}
    for t in Y:
        r = _evaluate(cfg, P, Y[t], sub.subject_id.to_numpy(), pairs, B)
        out[t] = r
        for model in models():
            want = ref[t]["variants"].get(f"{model}: grounding", {}).get("macro_auroc", {}).get("value")
            got = r["variants"][f"{model}: filter on"]["macro_auroc"]["value"]
            check[f"{t}|{model}"] = {"agent_models_json": want, "recomputed": got}
    (ART / "rescore.json").write_text(json.dumps({"reproduces_agent_models": check, "metrics": out}, indent=1))
    for t in Y:
        for model in models():
            a, b = out[t]["variants"][f"{model}: filter on"]["macro_auroc"], out[t]["variants"][f"{model}: filter off"]["macro_auroc"]
            d = out[t]["paired"][f"{model}: filter off - {model}: filter on"]["macro_auroc"]
            print(t, model, "on", round(a["value"], 3), "off", round(b["value"], 3), "diff", round(d["value"], 3), [round(x, 3) for x in d["ci"]],
                  "| json", check[f"{t}|{model}"]["agent_models_json"] and round(check[f"{t}|{model}"]["agent_models_json"], 3))


# ------------------------------------------------------------------------------------- inter-model agreement
def agreement():
    cfg = cfg21()
    P = cfg["propositions"]
    props = [k for k, p in P.items() if p["kind"] not in ("diagnosis", "demographic")]
    T = pd.read_parquet(ART / "sample_texts.parquet")
    ms = list(models())
    cat = {"present": 0, "possible": 0, "absent": 1, "unknown": 2}
    S = {}
    for model in ms:
        G = _ground(model)
        for variant, get in (("filter_on", lambda e: e["findings"]), ("filter_off", lambda e: _unfiltered(e, P))):
            A = np.full((len(T), len(props)), 2)
            for i, n in enumerate(T.note_id):
                if n not in G:
                    continue
                f = get(G[n])
                for j, k in enumerate(props):
                    st = f.get(k, {}).get("status", "unknown")
                    # undocumented history/medication is absent in the closed world the engine uses
                    A[i, j] = cat.get(st, 2) if not (st == "unknown" and P[k]["kind"] != "lab") else 1
            S[(model, variant)] = A

    def fleiss(X):  # X: [items, raters] categories 0..2
        n = X.shape[1]
        cnt = np.stack([(X == c).sum(1) for c in range(3)], 1)
        Pi = ((cnt * (cnt - 1)).sum(1)) / (n * (n - 1))
        pj = cnt.sum(0) / cnt.sum()
        Pe = (pj ** 2).sum()
        return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")

    def cohen(a, b):
        po = (a == b).mean()
        pe = sum((a == c).mean() * (b == c).mean() for c in range(3))
        return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
    out = {"categories": "present/possible, absent (incl. undocumented history/medication), unknown",
           "seed_variance": "not measurable from the caches: every model ran once at temperature 0 with seed 2026",
           "variants": {}}
    for variant in ("filter_on", "filter_off"):
        per = {}
        for j, k in enumerate(props):
            X = np.stack([S[(m, variant)][:, j] for m in ms], 1)
            per[k] = {"fleiss_kappa": fleiss(X), "kind": P[k]["kind"], "share_positive_any_model": float((X == 0).any(1).mean())}
        pair = {f"{a} | {b}": float(np.nanmean([cohen(S[(a, variant)][:, j], S[(b, variant)][:, j]) for j in range(len(props))]))
                for i, a in enumerate(ms) for b in ms[i + 1:]}
        kinds = {kd: float(np.nanmedian([v["fleiss_kappa"] for v in per.values() if v["kind"] == kd])) for kd in ("history", "medication", "lab")}
        out["variants"][variant] = {"per_item": per, "median_fleiss_by_kind": kinds,
                                    "median_fleiss_all": float(np.nanmedian([v["fleiss_kappa"] for v in per.values()])),
                                    "pairwise_mean_cohen": pair}
        print(variant, "median Fleiss", round(out["variants"][variant]["median_fleiss_all"], 3), {a: round(b, 3) for a, b in kinds.items()})
    (ART / "agreement.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- structured experiments
KB = Path("artifacts/v2.1/kb")


def _setup():
    from qddx.engine import dx_names
    cfg = cfg21()
    df = pd.read_parquet(KB / "cohort.parquet")
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    return cfg, df, tr, dx_names(cfg), kb.evidence(cfg)


def _score(cfg, Y, P, groups, B=200):
    from qddx.evaluate import bootstrap, metric_set, with_ci
    from qddx.engine import dx_names
    dx = dx_names(cfg)
    k = cfg["engine"]["top_k"]
    P = np.round(P, 12)  # elimination round-off (1e-17) would otherwise rank constant posteriors at random
    return with_ci(metric_set(Y, P, dx, k), bootstrap(groups, B, cfg["seed"], lambda w: metric_set(Y, P, dx, k, w)))


def _net_and_lr(cfg, train_df, test_df, dx, ev):
    """Network (kb.learn_bn, unchanged) and logistic regression (evaluate.fit_lr's recipe) on the given rows."""
    from sklearn.linear_model import LogisticRegression
    from qddx.evaluate import lr_features
    bn = kb.learn_bn(cfg, train_df)
    Pn, _ = kb.posteriors_by_pattern(cfg, bn, test_df)
    kinds = [cfg["propositions"][k]["kind"] for k in ev]
    M = train_df[ev].to_numpy(float, copy=True)
    M[M < 0] = np.nan
    X = lr_features(M, kinds)
    Mt = test_df[ev].to_numpy(float, copy=True)
    Mt[Mt < 0] = np.nan
    Xt = lr_features(Mt, kinds)
    Pl = np.stack([LogisticRegression(max_iter=2000).fit(X, train_df[d].to_numpy()).predict_proba(Xt)[:, 1] for d in dx], 1)
    return Pn, Pl


def missing(scope_key: str, m_s: str):
    """Round 3 C3.1: a fraction m of each admission's observed findings hidden at random (MCAR); separately, labs
    only. The network marginalises hidden findings; logistic regression sees unknown as 0 plus a lab-missing flag."""
    from qddx.evaluate import fit_lr, lr_predict
    cfg, df, tr, dx, ev = _setup()
    bn = BN_load()
    lr = fit_lr(cfg, KB)
    te = df[~tr].reset_index(drop=True)
    Y, g = te[dx].to_numpy(), te.subject_id.to_numpy()
    labs = np.array([cfg["propositions"][k]["kind"] == "lab" for k in ev])
    out = {}
    for scope in ({"all": "all findings", "labs": "labs only"}[scope_key],):
        for m in (float(m_s),):
            t = te.copy()
            rng = np.random.default_rng(SEED_X + int(m * 100) + (0 if scope == "all findings" else 1000))
            E = t[ev].to_numpy()
            hide = (rng.random(E.shape) < m) & (E >= 0) & (labs[None, :] if scope == "labs only" else True)
            E[hide] = -1
            t[ev] = E
            Pn, _ = kb.posteriors_by_pattern(cfg, bn, t)
            M = t[ev].to_numpy(float)
            M[M < 0] = np.nan
            Pl = lr_predict(lr, cfg, M)
            r = {"network": _score(cfg, Y, Pn, g), "logistic regression": _score(cfg, Y, Pl, g), "share_hidden": float(hide.mean())}
            out[f"{scope}|{m:g}"] = r
            print(scope, m, "net", round(r["network"]["macro_auroc"]["value"], 3), "lr", round(r["logistic regression"]["macro_auroc"]["value"], 3), flush=True)
    (ART / f"structured_missing_{scope_key}_{m_s}.json").write_text(json.dumps(out, indent=1))


SEED_X = 2026


def BN_load():
    from qddx.bn import BN
    return BN.from_dict(json.loads((KB / "bn.json").read_text()))


def _cal_stats(y, p):
    from qddx.evaluate import ece
    from sklearn.linear_model import LogisticRegression
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return {"ece_10": float(ece(y, p)), "brier": float(np.mean((p - y) ** 2)), "slope": float(m.coef_[0][0]),
            "intercept": float(m.intercept_[0]), "citl": float(p.mean() - y.mean())}


def calibration():
    """Round 3 C3.2: per diagnosis, raw and Platt posteriors (logistic regression alongside): 10-bin reliability,
    ECE, Brier, calibration slope and intercept, with patient-bootstrap CIs (100 resamples for slope/intercept)."""
    from qddx.engine import platt
    from qddx.evaluate import fit_lr, lr_predict
    cfg, df, tr, dx, ev = _setup()
    te = df[~tr].reset_index(drop=True)
    Pn, _ = kb.posteriors_by_pattern(cfg, BN_load(), te)
    cal = cfg["engine"].get("calibration") or {}
    Pc = np.stack([platt(Pn[:, i], *cal[d]) if d in cal else Pn[:, i] for i, d in enumerate(dx)], 1)
    M = te[ev].to_numpy(float)
    M[M < 0] = np.nan
    Pl = lr_predict(fit_lr(cfg, KB), cfg, M)
    Y = te[dx].to_numpy()
    g = te.subject_id.to_numpy()
    u, inv = np.unique(g, return_inverse=True)
    rows_of = pd.Series(np.arange(len(g))).groupby(inv).apply(np.array).tolist()
    rng = np.random.default_rng(SEED_X)
    boots = [np.concatenate([rows_of[k] for k in rng.integers(0, len(u), len(u))]) for _ in range(100)]
    out = {}
    for name, P in (("network raw", Pn), ("network + Platt", Pc), ("logistic regression", Pl)):
        per = {}
        for i, d in enumerate(dx):
            p, y = P[:, i], Y[:, i]
            st = _cal_stats(y, p)
            bs = [_cal_stats(y[b], p[b]) for b in boots]
            st["ci95"] = {k: [float(np.quantile([x[k] for x in bs], .025)), float(np.quantile([x[k] for x in bs], .975))] for k in st}
            q = np.linspace(0, 1, 11)
            b = np.clip((p * 10).astype(int), 0, 9)
            st["reliability_10_bins"] = [{"bin": f"{q[k]:.1f}-{q[k + 1]:.1f}", "mean_predicted": float(p[b == k].mean()) if (b == k).any() else None,
                                          "observed": float(y[b == k].mean()) if (b == k).any() else None, "n": int((b == k).sum())} for k in range(10)]
            per[d] = st
        out[name] = per
        print(name, {d: (round(v["ece_10"], 4), round(v["slope"], 2)) for d, v in per.items()}, flush=True)
    (ART / "structured_calibration.json").write_text(json.dumps(out, indent=1))


def labelnoise(mech: str, rate: str):
    """Round 3 C3.3: training labels flipped at rate eps (symmetric: every label; fn: positives only), the network
    relearned with kb.learn_bn, scored on clean test labels."""
    cfg, df, tr, dx, ev = _setup()
    r = float(rate)
    rng = np.random.default_rng(SEED_X + int(r * 1000) + (0 if mech == "fn" else 7))
    trd = df[tr].copy()
    flipped = {}
    for d in dx:
        y = trd[d].to_numpy().copy()
        f = (rng.random(len(y)) < r) & ((y == 1) if mech == "fn" else True)
        y[f] = 1 - y[f]
        trd[d] = y.astype(np.int8)
        flipped[d] = int(f.sum())
    te = df[~tr].reset_index(drop=True)
    t0 = time.perf_counter()
    bn = kb.learn_bn(cfg, trd)
    Pn, _ = kb.posteriors_by_pattern(cfg, bn, te)
    Y, g = te[dx].to_numpy(), te.subject_id.to_numpy()
    out = {"mechanism": mech, "rate": r, "flipped": flipped, "network": _score(cfg, Y, Pn, g)}
    (ART / f"structured_labelnoise_{mech}_{rate}.json").write_text(json.dumps(out, indent=1))
    print(mech, r, "net", round(out["network"]["macro_auroc"]["value"], 3), f"({time.perf_counter() - t0:.0f}s)")


def negctrl(which: str):
    """Round 3 C3.4: (a) training diagnosis rows permuted across patients, network relearned; (b) test findings
    shuffled across admissions. Pass: the macro AUROC 95% CI contains 0.5."""
    from qddx.evaluate import fit_lr, lr_predict
    cfg, df, tr, dx, ev = _setup()
    rng = np.random.default_rng(SEED_X + 99)
    te = df[~tr].reset_index(drop=True)
    Y, g = te[dx].to_numpy(), te.subject_id.to_numpy()
    out = {}
    if which == "a":
        trd = df[tr].copy()
        trd[dx] = trd[dx].to_numpy()[rng.permutation(len(trd))]
        Pn, _ = kb.posteriors_by_pattern(cfg, kb.learn_bn(cfg, trd), te)
        out["network"] = _score(cfg, Y, Pn, g)
    else:
        t = te.copy()
        t[ev] = t[ev].to_numpy()[rng.permutation(len(t))]
        Pn, _ = kb.posteriors_by_pattern(cfg, BN_load(), t)
        M = t[ev].to_numpy(float)
        M[M < 0] = np.nan
        out["network"] = _score(cfg, Y, Pn, g)
        out["logistic regression"] = _score(cfg, Y, lr_predict(fit_lr(cfg, KB), cfg, M), g)
    for k in out:
        ci = out[k]["macro_auroc"]["ci"]
        out[k]["pass_ci_contains_0.5"] = bool(ci[0] <= 0.5 <= ci[1])
        print(which, k, "macro AUROC", round(out[k]["macro_auroc"]["value"], 3), [round(x, 3) for x in ci], out[k]["pass_ci_contains_0.5"])
    (ART / f"structured_negctrl_{which}.json").write_text(json.dumps(out, indent=1))


def subgroups(column: str):
    """Round 3 C3.5: AUROC and calibration-in-the-large per diagnosis by sex, age (< 65 / >= 65) and race group
    (MIMIC categories; groups with fewer than 500 test admissions merged), patient-bootstrap CIs. Aggregates only."""
    import yaml
    from qddx.evaluate import bootstrap, with_ci
    cfg, df, tr, dx, ev = _setup()
    root = Path(yaml.safe_load(open("configs/qddx.yaml"))["paths"]["mimic_root"]) / "hosp"
    pth = lambda n: (root / n / n) if (root / n).is_dir() else root / n
    adm = pd.read_csv(pth("admissions.csv"), usecols=["hadm_id", "race"])
    pat = pd.read_csv(pth("patients.csv"), usecols=["subject_id", "gender"])
    te = df[~tr].reset_index(drop=True).merge(adm, on="hadm_id", how="left").merge(pat, on="subject_id", how="left")
    te["race"] = te.race.fillna("UNKNOWN")
    small = te.race.value_counts()
    small = set(small[small < 500].index)
    te["race_group"] = np.where(te.race.isin(small), "OTHER (merged, < 500 each)", te.race)
    te["age_group"] = np.where(te.age_ge_65 == 1, ">=65", "<65")
    Pn, _ = kb.posteriors_by_pattern(cfg, BN_load(), te)
    from qddx.engine import platt
    cal = cfg["engine"].get("calibration") or {}
    Pc = np.stack([platt(Pn[:, i], *cal[d]) if d in cal else Pn[:, i] for i, d in enumerate(dx)], 1)
    Y = te[dx].to_numpy()
    from qddx.evaluate import _auc

    def stats(w, idx):
        o = {}
        for i, d in enumerate(dx):
            o[f"auroc_{d}"] = _auc(Y[idx, i], Pc[idx, i], w)
            o[f"citl_{d}"] = float(np.average(Pc[idx, i] - Y[idx, i], weights=w))
        o["macro_auroc"] = float(np.mean([o[f"auroc_{d}"] for d in dx]))
        return o
    out = {"merged_race_categories": sorted(small), "bootstrap": 100}
    for col in (column,):
        out[col] = {}
        for val, idx in te.groupby(col).groups.items():
            idx = np.asarray(idx)
            g = te.subject_id.to_numpy()[idx]
            out[col][str(val)] = {"n": int(len(idx)), **with_ci(stats(np.ones(len(idx)), idx), bootstrap(g, 100, cfg["seed"], lambda w, idx=idx: stats(w, idx)))}
            print(col, val, len(idx), "macro AUROC", round(out[col][str(val)]["macro_auroc"]["value"], 3), flush=True)
    (ART / f"structured_subgroups_{column}.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------- C1 analysis of the diagnosis sample
_ELL = r"\.\.\.+|…|\[\.\.\.\]|\(\.\.\.\)"
_DEID = r"_{2,}|\[\*\*.*?\*\*\]|\[(?:name|date|redacted|hospital|location|md|doctor|phone|address)[^\]]*\]|\*{2,}"
MECHANICAL = ("ellipsis (each segment verbatim)", "punctuation / Unicode", "de-identification token", "value format",
              "mechanical (combined)")


def _uni(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s)).lower()
    return " ".join(re.findall(r"[a-z0-9.]+", s.replace("−", "-"))).replace(" .", " ").strip(". ")


def _deid(s: str) -> str:
    return re.sub(_DEID, " ", str(s), flags=re.I)


def _num(s: str) -> str:
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", str(s))
    s = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", s)
    s = re.sub(r"(\d+)\.0+(?!\d)", r"\1", s)
    return re.sub(r"(\d)([a-zA-Z%])", r"\1 \2", s)


def _segments(q: str, f) -> list[str]:
    return [x for x in (f(seg.strip().strip(_QUOTES)) for seg in re.split(_ELL, q)) if x]


def classify_quote(q: str, note: str) -> str:
    """One category per rejected quote, the first that applies, in the frozen plan's order."""
    from rapidfuzz import fuzz
    q = str(q or "")
    if not q.strip():
        return "empty quote"
    nn = AG._norm(note)
    if re.search(_ELL, q) and all(x in nn for x in _segments(q, AG._norm)):
        return "ellipsis (each segment verbatim)"
    qs = q.strip().strip(_QUOTES)
    if _uni(qs) and _uni(qs) in _uni(note):
        return "punctuation / Unicode"
    if re.search(_DEID, qs, flags=re.I) and _uni(_deid(qs)) and _uni(_deid(qs)) in _uni(_deid(note)):
        return "de-identification token"
    if re.search(r"\d", qs) and _uni(_num(qs)) and _uni(_num(qs)) in _uni(_num(note)):
        return "value format"
    full = lambda x: _uni(_num(_deid(x)))
    if all(x in full(note) for x in _segments(q, full)) and _segments(q, full):
        return "mechanical (combined)"
    if fuzz.partial_ratio(_uni(qs), _uni(note)) >= 90:
        return "paraphrase (fuzzy >= 0.9)"
    return "fabricated (no match)"


def corrected_validate(f: dict, note: str, cfg: dict) -> tuple[dict, str | None, str | None]:
    """The frozen validator, plus acceptance of quotes whose only fault is mechanical. -> (finding, reason, category)."""
    ok, why = AG.validate(f, note, cfg)
    if why not in ("quote not in note", "value not in quote"):
        return ok, why, None
    q = str(f.get("quote") or "")
    if why == "quote not in note":
        cat = classify_quote(q, note)
        if cat not in MECHANICAL:
            return ok, why, cat
        g = {**f, "quote": " ".join(_segments(q, lambda x: x))}
        st = AG.normalize_status(f.get("status"), cfg["propositions"][f["id"]]["kind"])
        v = f.get("value")
        try:
            v = float(v) if v is not None and cfg["propositions"][f["id"]]["kind"] == "lab" else None
        except (TypeError, ValueError):
            return ok, why, cat
        if v is not None and not any(abs(float(x) - v) < 1e-9 for x in re.findall(r"\d+(?:\.\d+)?", _num(q))):
            return ok, "value not in quote", cat
        return {"status": st, "value": v, "quote": q[:200]}, None, cat
    v = float(f.get("value"))
    if any(abs(float(x) - v) < 1e-9 for x in re.findall(r"\d+(?:\.\d+)?", _num(q))):
        return {"status": AG.normalize_status(f.get("status"), cfg["propositions"][f["id"]]["kind"]), "value": v,
                "quote": q[:200]}, None, "value format"
    return ok, why, "value not in quote (other)"


def _findings_of(raw: list[str]) -> list:
    for r in raw:  # the pipeline uses the schema reply, or the JSON-mode retry when the schema reply is empty
        got = AG.parse_json(r).get("findings") if isinstance(AG.parse_json(r), dict) else None
        if got:
            return got if isinstance(got, list) else []
    return []


def probe_analyse():
    """Classify every rejected quote in the diagnosis sample, apply the corrected validator, and compare the re-run
    with the original cache (reproducibility at temperature 0). Aggregates only."""
    cfg = cfg21()
    P = cfg["propositions"]
    props = [k for k, p in P.items() if p["kind"] not in ("diagnosis", "demographic")]
    T = pd.read_parquet(ART / "sample_texts.parquet").set_index("note_id").text
    from rapidfuzz import fuzz
    prompt_u = _uni(AG.SYSTEM + "\n" + "\n".join(AG.describe(k, p) for k, p in P.items()
                                                   if p["kind"] not in ("diagnosis", "demographic")))
    out = {"mechanical_categories": list(MECHANICAL), "models": {}}
    for model in models():
        d = ART / "probe" / _slug(model)
        files = sorted(d.glob("*.json")) if d.exists() else []
        if not files:
            continue
        G = _ground(model)
        cats, by_kind, reasons = {}, {}, {}
        same_note, n_cmp, agree, tot = 0, 0, 0, 0
        gained = {"present_or_possible": 0, "lab_value": 0, "absent": 0}
        for f in files:
            nid = f.stem
            e = json.loads(f.read_text())
            note = T[nid]
            got = _findings_of(e["raw"])
            # sanity: the frozen validator on the saved raw reproduces the saved validated findings
            for g in got:
                if not isinstance(g, dict):
                    continue
                _, why = AG.validate(g, note, cfg)
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                fixed, why2, cat = corrected_validate(g, note, cfg)
                if cat == "fabricated (no match)":  # where did the text come from? the prompt's proposition list?
                    u = _uni(str(g.get("quote") or ""))
                    cat = ("fabricated: verbatim from the prompt" if u and u in prompt_u else
                           "fabricated: near-verbatim from the prompt (fuzzy >= 0.9)" if u and fuzz.partial_ratio(u, prompt_u) >= 90 else
                           "fabricated: no match in note or prompt")
                if cat:
                    cats[cat] = cats.get(cat, 0) + 1
                    kd = P.get(g.get("id"), {}).get("kind", "other")
                    by_kind.setdefault(kd, {}).setdefault(cat, 0)
                    by_kind[kd][cat] += 1
                if why and not why2:
                    k = g["id"]
                    if fixed["status"] in ("present", "possible"):
                        gained["present_or_possible"] += 1
                    elif fixed["value"] is not None:
                        gained["lab_value"] += 1
                    else:
                        gained["absent"] += 1
            if nid in G:
                n_cmp += 1
                a = {k: G[nid]["findings"].get(k, {}).get("status", "unknown") for k in props}
                b = {k: e["findings"].get(k, {}).get("status", "unknown") for k in props}
                same_note += a == b and G[nid]["findings"] == e["findings"]
                agree += sum(a[k] == b[k] for k in props)
                tot += len(props)
        rej = sum(cats.values())
        out["models"][model] = {
            "notes": len(files), "rejections_by_reason": reasons, "quote_categories": cats, "quote_categories_by_kind": by_kind,
            "mechanical_share": (sum(v for k, v in cats.items() if k in MECHANICAL) / rej) if rej else None,
            "findings_recovered_by_corrected_validator": gained,
            "reproducibility": {"notes_compared": n_cmp, "notes_identical": same_note,
                                "status_agreement": agree / tot if tot else None}}
        print(model, json.dumps(out["models"][model]), flush=True)
    (ART / "probe_analysis.json").write_text(json.dumps(out, indent=1))


def agreement4(B: str = "200"):
    """C2 as frozen: Fleiss kappa per proposition across the 8 models on the validated status (present / possible /
    absent / unknown, no closed-world recoding), note-level bootstrap CI; pairwise Cohen; seed variance from the C1 probe."""
    cfg = cfg21()
    P = cfg["propositions"]
    props = [k for k, p in P.items() if p["kind"] not in ("diagnosis", "demographic")]
    T = pd.read_parquet(ART / "sample_texts.parquet")
    ms = list(models())
    cat = {"present": 0, "possible": 1, "absent": 2, "unknown": 3}
    S = {}
    for model in ms:
        G = _ground(model)
        A = np.full((len(T), len(props)), 3)
        for i, n in enumerate(T.note_id):
            f = G.get(n, {"findings": {}})["findings"]
            for j, k in enumerate(props):
                A[i, j] = cat.get(f.get(k, {}).get("status", "unknown"), 3)
        S[model] = A
    X = np.stack([S[m] for m in ms], 2)  # notes x props x raters
    C = 4

    def fleiss(x):  # x: items x raters
        n = x.shape[1]
        cnt = np.stack([(x == c).sum(1) for c in range(C)], 1)
        Pi = (cnt * (cnt - 1)).sum(1) / (n * (n - 1))
        pj = cnt.sum(0) / cnt.sum()
        Pe = (pj ** 2).sum()
        return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 - 1e-12 else float("nan")

    def cohen(a, b):
        po = (a == b).mean()
        pe = sum((a == c).mean() * (b == c).mean() for c in range(C))
        return float((po - pe) / (1 - pe)) if pe < 1 - 1e-12 else float("nan")
    rng = np.random.default_rng(cfg["seed"])
    idx = [rng.integers(0, len(T), len(T)) for _ in range(int(B))]
    per = {}
    for j, k in enumerate(props):
        x = X[:, j, :]
        bs = np.array([fleiss(x[b]) for b in idx])
        lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan))
        per[k] = {"kind": P[k]["kind"], "fleiss_kappa": fleiss(x), "ci95": [float(lo), float(hi)],
                  "status_share": {c: float((x == v).mean()) for c, v in cat.items()}}
    kinds = {kd: float(np.nanmedian([v["fleiss_kappa"] for v in per.values() if v["kind"] == kd])) for kd in ("history", "medication", "lab")}
    pair = {}
    for a_i, a in enumerate(ms):
        for b in ms[a_i + 1:]:
            pair[f"{a} | {b}"] = float(np.nanmedian([cohen(S[a][:, j], S[b][:, j]) for j in range(len(props))]))
    # seed variance = reproducibility of the C1 re-run (temperature 0), on the same 4 categories
    rep = {}
    for model in ms:
        d = ART / "probe" / _slug(model)
        files = sorted(d.glob("*.json")) if d.exists() else []
        G = _ground(model)
        a, b = [], []
        for f in files:
            if f.stem not in G:
                continue
            e = json.loads(f.read_text())
            a += [cat.get(G[f.stem]["findings"].get(k, {}).get("status", "unknown"), 3) for k in props]
            b += [cat.get(e["findings"].get(k, {}).get("status", "unknown"), 3) for k in props]
        if a:
            a, b = np.array(a), np.array(b)
            rep[model] = {"notes": len(files), "status_agreement": float((a == b).mean()), "cohen_kappa_rerun_vs_original": cohen(a, b)}
    out = {"categories": list(cat), "bootstrap": int(B), "per_item": per, "median_fleiss_by_kind": kinds,
           "median_fleiss_all": float(np.nanmedian([v["fleiss_kappa"] for v in per.values()])),
           "pairwise_median_cohen": pair, "seed_variance_reproducibility": rep}
    (ART / "agreement4.json").write_text(json.dumps(out, indent=1))
    print("median Fleiss", round(out["median_fleiss_all"], 3), {a: round(v, 3) for a, v in kinds.items()})
    for m, v in rep.items():
        print("rerun", m, v)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"texts": texts, "map": map_, "quotes": quotes, "probe": probe, "predictions": predictions, "rescore": rescore, "agreement": agreement,
     "missing": missing, "calibration": calibration, "labelnoise": labelnoise, "negctrl": negctrl, "subgroups": subgroups,
     "probe-analyse": probe_analyse, "agreement4": agreement4}[cmd](*args)
