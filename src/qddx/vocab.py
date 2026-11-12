"""Run 1 (`qddx discover`): choose the vocabulary from data, without touching test data.

1. Candidate pool of admission-time items for every training-split admission:
   previous-admission ICD categories (history), drugs started in the first 24 h, abnormal first labs (high / low),
   age >= 65 and sex. Same-admission diagnosis codes are never candidates.
2. Ranking on the inner-train patients: greedy max-relevance / min-redundancy with the 6 target diagnoses.
3. Answer-leak flags: same concept as a target, likely treatment of a target, strong single-target marker,
   low note extractability.
4. Note extractability on training-split MIMIC-IV-Note notes: mined lab abbreviations, drug names, history phrases.
5. Validation curve (inner-validation patients): networks from the top K items at several max-parents settings,
   logistic regression on the same items and on the whole pool (ceiling); recommended K by a written rule;
   per-diagnosis calibration for the recommended network.
Outputs: reports/vocab_discovery.md, artifacts/vocab/candidates.json, configs/vocab_v2.draft.json.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import kb
from . import notes as N
from .engine import dx_names, platt
from .evaluate import lr_features, metric_set

log = logging.getLogger("qddx")
_GENERIC = {"unspecified", "other", "without", "with", "of", "and", "or", "the", "due", "to", "in", "not", "elsewhere",
            "classified", "specified", "type", "nos", "site", "a", "an", "by", "for", "on", "as", "at"}


# ------------------------------------------------------------------------------------- splits
def inner_split(cfg: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> masks (inner_train, validation, test) over cohort rows; validation = held-back training patients."""
    tr = kb.split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    sub = df.subject_id.to_numpy()[tr]
    inner = np.zeros(len(df), bool)
    inner[np.where(tr)[0]] = kb.split_mask(sub, cfg["discover"]["inner_split"], cfg["seed"] + 1)
    return inner, tr & ~inner, ~tr


# ------------------------------------------------------------------------------------- 1. candidate pool
POOL_VERSION = 2   # v2: two-word salt names, ICD categories named after their most frequent code, per-part caches
_SALTS = {"sodium", "potassium", "magnesium", "calcium", "ferrous", "aluminum", "zinc"}
SUPPLY_WORDS = {"bag", "vial", "syringe", "flush", "kit", "tube", "needle", "lancet", "strip", "strips", "pad", "pads",
                "dressing", "swab", "catheter", "bottle", "container", "pen", "ampule", "diluent", "sterile", "water"}
ADMIN_PREFIXES = {"9": ("V", "E"), "10": ("Z", "V", "W", "X", "Y")}   # status / encounter / external-cause codes


def _drug_name_sql(col: str) -> str:
    """First word of the lower-cased drug name; two words when the first is a salt or ion (sodium chloride vs
    sodium bicarbonate)."""
    w = f"regexp_extract_all(lower({col}), '[a-z]+')"
    salts = ", ".join(f"'{x}'" for x in sorted(_SALTS))
    return f"(CASE WHEN {w}[1] IN ({salts}) AND len({w}) > 1 THEN {w}[1] || ' ' || {w}[2] ELSE {w}[1] END)"


def build_pool(cfg: dict, kbdir: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (long items [hadm_id, item_id, value], item metadata, first lab values [hadm_id, itemid, v]); training-split
    admissions only. Each source is cached separately under out/ (part_<source>_v<version>.parquet), so a fix to one
    source does not rescan the others; an older combined pool.parquet donates its lab and demographic parts."""
    final = {n: out / f"{n}.parquet" for n in ("pool", "meta", "labvals")}
    if all(p.exists() for p in final.values()):
        meta = pd.read_parquet(final["meta"])
        if "pool_version" in meta.columns and len(meta) and int(meta["pool_version"].iloc[0]) == POOL_VERSION:
            return pd.read_parquet(final["pool"]), meta, pd.read_parquet(final["labvals"])
    d, k = cfg["discover"], cfg["knowledge"]
    root = Path(cfg["paths"]["mimic_root"])
    part = {n: out / f"part_{n}_v{POOL_VERSION}.parquet" for n in ("hx", "rx", "lab", "demo")}
    limits_path, titles_path = out / f"lab_limits_v{POOL_VERSION}.parquet", out / f"icd_titles_v{POOL_VERSION}.parquet"
    # donate lab / demographic parts (and lab limits) from an older combined pool, to avoid rescanning labevents
    if final["pool"].exists() and final["meta"].exists():
        old, ometa = pd.read_parquet(final["pool"]), pd.read_parquet(final["meta"])
        names = ometa.set_index("item_id")["item"]
        for src, prefix in (("lab", "lab:"), ("demo", "demo:")):
            if not part[src].exists():
                ids = ometa.loc[ometa["item"].str.startswith(prefix), "item_id"]
                sub = old[old["item_id"].isin(ids)]
                pd.DataFrame({"hadm_id": sub["hadm_id"].to_numpy(), "item": names.loc[sub["item_id"]].to_numpy(),
                              "value": sub["value"].to_numpy()}).to_parquet(part[src])
        if not limits_path.exists() and {"lower", "upper"} <= set(ometa.columns):
            lm = ometa[ometa["source"] == "lab"].drop_duplicates("code")
            pd.DataFrame({"itemid": lm["code"].astype(int), "plo": lm["lower"], "phi": lm["upper"]}).to_parquet(limits_path)
    cohort = pd.read_parquet(kbdir / "cohort.parquet", columns=["subject_id", "hadm_id"])
    tr = kb.split_mask(cohort.subject_id.to_numpy(), k["split"][0], cfg["seed"])
    con = kb.connect(cfg)
    con.register("train_ids", cohort.loc[tr, ["hadm_id"]])
    lo_h, hi_h = k["lab_window_hours"]
    mlo, mhi = k["med_window_hours"]
    con.execute(f"""CREATE TEMP TABLE adm AS
        SELECT CAST(a.subject_id AS BIGINT) subject_id, CAST(a.hadm_id AS BIGINT) hadm_id,
               CAST(a.admittime AS TIMESTAMP) admittime,
               CAST(p.anchor_age AS INT) + (year(CAST(a.admittime AS TIMESTAMP)) - CAST(p.anchor_year AS INT)) age, p.gender sex
        FROM {kb._scan(root, 'admissions')} a JOIN {kb._scan(root, 'patients')} p USING (subject_id)
        WHERE CAST(a.hadm_id AS BIGINT) IN (SELECT hadm_id FROM train_ids)""")
    if not part["hx"].exists():  # previous admissions' ICD categories (discharged before this admission)
        con.execute(f"""COPY (
            WITH p AS (SELECT CAST(subject_id AS BIGINT) subject_id, CAST(hadm_id AS BIGINT) hadm_id,
                              CAST(dischtime AS TIMESTAMP) dischtime FROM {kb._scan(root, 'admissions')}),
                 dx AS (SELECT CAST(hadm_id AS BIGINT) hadm_id, icd_version v, substr(icd_code, 1, 3) cat
                        FROM {kb._scan(root, 'diagnoses_icd')})
            SELECT DISTINCT a.hadm_id, 'hx' || dx.v || ':' || dx.cat AS item, 1 AS value
            FROM adm a JOIN p ON p.subject_id = a.subject_id AND p.dischtime <= a.admittime JOIN dx ON dx.hadm_id = p.hadm_id
            ) TO '{part["hx"].as_posix()}' (FORMAT PARQUET)""")
    if not titles_path.exists():  # category name = title of its most frequent code; all titles kept for leak checks
        con.execute(f"""COPY (
            WITH n AS (SELECT icd_version v, icd_code c, count(*) cnt FROM {kb._scan(root, 'diagnoses_icd')} GROUP BY 1, 2),
                 t AS (SELECT icd_version v, icd_code c, long_title FROM {kb._scan(root, 'd_icd_diagnoses')})
            SELECT t.v, substr(t.c, 1, 3) cat, arg_max(t.long_title, coalesce(n.cnt, 0)) title,
                   string_agg(t.c || '=' || t.long_title, ' | ') all_titles
            FROM t LEFT JOIN n ON n.v = t.v AND n.c = t.c GROUP BY 1, 2
            ) TO '{titles_path.as_posix()}' (FORMAT PARQUET)""")
    if not part["rx"].exists():  # drugs started in the first 24 h
        name = _drug_name_sql("r.drug")
        con.execute(f"""COPY (
            SELECT DISTINCT a.hadm_id, 'rx:' || {name} AS item, 1 AS value
            FROM {kb._scan(root, 'prescriptions')} r JOIN adm a ON CAST(r.hadm_id AS BIGINT) = a.hadm_id
            WHERE TRY_CAST(r.starttime AS TIMESTAMP) >= a.admittime + INTERVAL ({mlo}) HOUR
              AND TRY_CAST(r.starttime AS TIMESTAMP) <= a.admittime + INTERVAL ({mhi}) HOUR
              AND length(regexp_extract(lower(r.drug), '[a-z]+', 0)) >= 3
            ) TO '{part["rx"].as_posix()}' (FORMAT PARQUET)""")
    if not (part["lab"].exists() and final["labvals"].exists() and limits_path.exists()):
        # first lab value per itemid in the window, abnormal high / low (row range, else population median range)
        con.execute(f"""CREATE TEMP TABLE lab AS
            SELECT CAST(l.subject_id AS BIGINT) subject_id, CAST(l.itemid AS INT) itemid, TRY_CAST(l.charttime AS TIMESTAMP) t,
                   TRY_CAST(l.valuenum AS DOUBLE) v, TRY_CAST(l.ref_range_lower AS DOUBLE) rlo, TRY_CAST(l.ref_range_upper AS DOUBLE) rhi
            FROM {kb._scan(root, 'labevents')} l WHERE TRY_CAST(l.valuenum AS DOUBLE) IS NOT NULL
              AND CAST(l.subject_id AS BIGINT) IN (SELECT DISTINCT subject_id FROM adm)""")
        con.execute(f"""CREATE TEMP TABLE first AS
            SELECT a.hadm_id, l.itemid, arg_min(l.v, l.t) v, arg_min(l.rlo, l.t) rlo, arg_min(l.rhi, l.t) rhi
            FROM lab l JOIN adm a ON l.subject_id = a.subject_id
             AND l.t >= a.admittime + INTERVAL ({lo_h}) HOUR AND l.t <= a.admittime + INTERVAL ({hi_h}) HOUR
            GROUP BY 1, 2""")
        con.execute("""CREATE TEMP TABLE lim AS SELECT itemid, median(rlo) FILTER (WHERE rlo > 0) plo,
                       median(rhi) FILTER (WHERE rhi > 0) phi FROM lab GROUP BY 1""")
        q = [f"""SELECT f.hadm_id, 'lab:' || f.itemid || ':{direction}' AS item, ({cond})::INT AS value
                 FROM first f JOIN lim USING (itemid) WHERE ({cond}) IS NOT NULL"""
             for direction, cond in (("high", "f.v > coalesce(f.rhi, lim.phi)"), ("low", "f.v < coalesce(f.rlo, lim.plo)"))]
        con.execute(f"COPY ({' UNION ALL '.join(q)}) TO '{part['lab'].as_posix()}' (FORMAT PARQUET)")
        con.execute(f"COPY (SELECT hadm_id, itemid, v FROM first) TO '{final['labvals'].as_posix()}' (FORMAT PARQUET)")
        con.execute(f"COPY (SELECT itemid, plo, phi FROM lim) TO '{limits_path.as_posix()}' (FORMAT PARQUET)")
    if not part["demo"].exists():
        a65 = int(d["age_ge"])
        con.execute(f"""COPY (SELECT hadm_id, 'demo:age_ge_{a65}' item, (age >= {a65})::INT value FROM adm
                             UNION ALL SELECT hadm_id, 'demo:sex_f' item, (sex = 'F')::INT value FROM adm)
                        TO '{part["demo"].as_posix()}' (FORMAT PARQUET)""")
    n_adm = con.execute("SELECT count(*) FROM adm").fetchall()[0][0]
    min_pos = max(1, int(float(d["min_prevalence"]) / 2 * n_adm))  # loose pre-filter; exact floor on inner-train later
    srcs = " UNION ALL ".join(f"SELECT hadm_id, item, value FROM read_parquet('{part[n].as_posix()}')" for n in part)
    con.execute(f"CREATE TEMP TABLE pool_long AS {srcs}")
    con.execute(f"""CREATE TEMP TABLE items AS SELECT item, (row_number() OVER (ORDER BY item) - 1)::INT AS item_id
                    FROM (SELECT item FROM pool_long GROUP BY item HAVING sum(value) >= {min_pos})""")
    pool = con.execute("SELECT p.hadm_id, i.item_id, p.value::TINYINT AS value FROM pool_long p JOIN items i USING (item)").df()
    names = con.execute("SELECT item FROM items ORDER BY item_id").df()["item"].tolist()
    labels = con.execute(f"SELECT CAST(itemid AS INT) itemid, label, fluid FROM {kb._scan(root, 'd_labitems')}").df()
    con.close()
    titles, limits = pd.read_parquet(titles_path), pd.read_parquet(limits_path)
    tmap = {f"hx{r.v}:{r.cat}": (r.title, r.all_titles) for r in titles.itertuples()}
    lmap = {int(r.itemid): (r.label, r.fluid) for r in labels.itertuples()}
    limmap = {int(r.itemid): (r.plo, r.phi) for r in limits.itertuples()}
    rows = []
    for iid, item in enumerate(names):
        src, _, rest = item.partition(":")
        m = {"item": item, "item_id": iid, "pool_version": POOL_VERSION,
             "source": {"hx9": "history", "hx10": "history", "rx": "medication", "lab": "lab", "demo": "demographic"}[src]}
        if src.startswith("hx"):
            t, allt = tmap.get(item, (rest, ""))
            m.update(icd_version=src[2:], code=rest, title=t, category_titles=allt)
        elif src == "rx":
            m.update(code=rest, title=rest)
        elif src == "lab":
            itemid, direction = rest.split(":")
            lab, fluid = lmap.get(int(itemid), (itemid, ""))
            lo, hi = limmap.get(int(itemid), (None, None))
            m.update(code=itemid, title=f"{lab} {direction}" + (f" ({fluid.lower()})" if fluid and fluid.lower() != "blood" else ""),
                     direction=direction, lab_label=lab, lower=lo, upper=hi)
        else:
            m.update(code=rest, title={"sex_f": "female"}.get(rest, f"age {int(d['age_ge'])} or older"))
        rows.append(m)
    meta = pd.DataFrame(rows)
    labvals = pd.read_parquet(final["labvals"])
    pool.to_parquet(final["pool"])
    meta.to_parquet(final["meta"])
    return pool, meta, labvals


def pool_matrix(pool: pd.DataFrame, meta: pd.DataFrame, hadm: np.ndarray, items: list[str]) -> np.ndarray:
    """int8 [len(hadm), len(items)]: 1 / 0; labs -1 when not measured, other items 0 when absent."""
    iid = meta.set_index("item").loc[items, "item_id"].to_numpy()
    col = pd.Series(np.arange(len(items)), index=iid)
    row = pd.Series(np.arange(len(hadm)), index=hadm)
    X = np.zeros((len(hadm), len(items)), np.int8)
    X[:, np.array([it.startswith("lab:") for it in items], bool)] = -1
    p = pool[pool["item_id"].isin(col.index) & pool["hadm_id"].isin(row.index)]
    X[row.loc[p["hadm_id"]].to_numpy(), col.loc[p["item_id"]].to_numpy()] = p["value"].to_numpy()
    return X


# ------------------------------------------------------------------------------------- 2-3. ranking and flags
def _mi_cols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mutual information (nats) between every binary column of X (-1 = missing) and binary y, on rows where the
    column is observed, scaled by the observed fraction."""
    obs = X >= 0
    n_obs = obs.sum(0).astype(float)
    out = np.zeros(X.shape[1])
    x1 = (X == 1)
    for a in (0, 1):
        ya = (y == a)[:, None]
        for b in (0, 1):
            nab = ((x1 if b else (obs & ~x1)) & ya).sum(0).astype(float)
            na = (ya & obs).sum(0).astype(float)
            nb = (x1 if b else (obs & ~x1)).sum(0).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                term = nab / n_obs * np.log(nab * n_obs / (na * nb))
            out += np.nan_to_num(term, nan=0.0, posinf=0.0, neginf=0.0)
    return out * n_obs / len(X)


def rank_items(cfg: dict, X: np.ndarray, Y: np.ndarray, items: list[str]) -> tuple[list[str], dict]:
    """Greedy mRMR: next item = argmax [sum_d MI(item; d) - mean_s MI(item; s)] over the items already selected.
    After `discover.rank_limit` greedy steps the rest follow by relevance alone."""
    rel = sum(_mi_cols(X, Y[:, i]) for i in range(Y.shape[1]))
    red_sum = np.zeros(len(items))
    chosen, left = [], set(range(len(items)))
    stats = {}
    limit = int(cfg["discover"].get("rank_limit", 200))
    while left and len(chosen) < limit:
        idx = np.array(sorted(left))
        score = rel[idx] - (red_sum[idx] / len(chosen) if chosen else 0)
        j = int(idx[int(np.argmax(score))])
        stats[items[j]] = {"relevance": float(rel[j]), "mrmr_score": float(score.max())}
        chosen.append(j)
        left.discard(j)
        if left:
            s = X[:, j]
            ys = np.where(s >= 0, s, 0)
            m = _mi_cols(X, ys) if (s >= 0).all() else _mi_cols(np.where((s >= 0)[:, None], X, -1), ys)
            red_sum += m
    for j in sorted(left, key=lambda j: -rel[j]):
        stats[items[j]] = {"relevance": float(rel[j]), "mrmr_score": None}
        chosen.append(j)
    return [items[j] for j in chosen], stats


def leak_flags(cfg: dict, meta_row: dict, X_col: np.ndarray, Y: np.ndarray) -> tuple[list[str], dict]:
    """Flags for the discussion (rules written before looking at any result):
    same_concept  history category that contains a target diagnosis: one of its codes starts with a target's ICD prefix,
                  or one of its code titles names the target (e.g. 'hypertensive heart disease with heart failure');
    treatment     drug started in the first 24 h with lift >= discover.treatment_lift for any target (given because the
                  clinician already suspected that diagnosis, so it partly reveals the answer);
    administrative  status / encounter / external-cause code (ICD-9 V/E, ICD-10 Z/V/W/X/Y): records care, not disease;
    supply        not a drug (packaging, sterile water, flushes);
    routine       order given to >= discover.routine_prevalence of admissions (hospital routine, not patient-specific);
    marker        lab with lift >= 3 for one target and <= 1.5 for the others (informational; labs are evidence)."""
    P = cfg["propositions"]
    d_ = cfg["discover"]
    dx = dx_names(cfg)
    flags, lift = [], {}
    pos = X_col == 1
    for i, d in enumerate(dx):
        base = Y[:, i].mean()
        lift[d] = float(Y[pos, i].mean() / base) if pos.any() and base > 0 else 0.0
    src = meta_row["source"]
    if src == "history":
        v, code = str(meta_row["icd_version"]), str(meta_row["code"])
        titles = str(meta_row.get("category_titles") or "")
        codes = [c.split("=", 1)[0].strip() for c in titles.split(" | ") if "=" in c] or [code]
        low = titles.lower()
        for d in dx:
            prefixes = [str(c) for c in P[d].get(f"icd{v}", [])]
            terms = [t.lower() for t in P[d].get("label_terms", []) if len(t) > 3]
            if any(c.startswith(pfx) or pfx.startswith(c) for c in codes for pfx in prefixes) or \
                    any(re.search(rf"(?<![a-z]){re.escape(t)}(?![a-z])", low) for t in terms):
                flags.append(f"same_concept:{d}")
        if code[:1].upper() in ADMIN_PREFIXES.get(v, ()):
            flags.append("administrative")
    elif src == "medication":
        name = str(meta_row["code"])
        if name in SUPPLY_WORDS or name.split()[0] in SUPPLY_WORDS:
            flags.append("supply")
        if pos.mean() >= float(d_.get("routine_prevalence", 0.3)):
            flags.append("routine")
        best = max(lift, key=lift.get)
        if lift[best] >= float(d_.get("treatment_lift", 3.0)):
            flags.append(f"treatment:{best}")
    elif src == "lab":
        top = sorted(lift.values(), reverse=True)
        if len(top) > 1 and top[0] >= 3 and top[1] <= 1.5:
            flags.append(f"marker:{max(lift, key=lift.get)}")
    return flags, lift


# ------------------------------------------------------------------------------------- 4. note extractability
def _note_sample(cfg: dict, train_hadm: set, val_hadm: set | None = None) -> list[dict]:
    """Deterministic sample of training-split MIMIC-IV-Note notes (never test-split ones). Notes of validation patients
    also keep their admission-time evidence text and Discharge Diagnosis labels (for the note-based curve)."""
    from .mimicnotes import _stream
    frac = float(cfg["discover"]["notes_fraction"])
    out = []
    for r in _stream(Path(cfg["paths"]["mimic_notes"])):
        try:
            h = int(r["hadm_id"])
        except ValueError:
            continue
        if h not in train_hadm:
            continue
        if int(hashlib.md5(r["note_id"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= frac:
            continue
        secs = N.sections(r["text"])
        stops = [s.lower() for s in cfg["notes"]["stop_markers"]]
        pr = secs.get("pertinent results", "")
        cut = min([pr.lower().find(s) for s in stops if pr.lower().find(s) >= 0], default=len(pr))
        rec = {"hadm_id": h, "pmh": secs.get("past medical history", "").lower(),
               "meds": (secs.get("medications on admission", "") + "\n" + secs.get("history of present illness", "")).lower(),
               "labs": pr[:cut]}
        if val_hadm is not None and h in val_hadm:
            y = N.labels(r["text"], cfg)
            if y is not None:
                rec.update(evidence=N.evidence_text(N.deidentify(r["text"])[0], cfg), labels=y)
        out.append(rec)
    return out


# TOKEN-value as written in MIMIC results ("Creat-1.2", "INR(PT)-1.4"); a value followed by '+' is a grade ("Anisocy-1+")
_PAIR = re.compile(r"(?<![A-Za-z0-9(])([A-Za-z][A-Za-z0-9()]{1,15})-(\d+(?:\.\d+)?)(?![\d.+])")


def mine_lab_aliases(notes: list[dict], labvals: pd.DataFrame, min_support: int = 20, min_precision: float = 0.5) -> dict:
    """{itemid: [(alias, precision, support)]}: note tokens (>= 2 characters) written as TOKEN-value whose value equals
    that admission's first value of the item (rounded to the note's decimals; an integer must match exactly). Counted
    per note, since results list the same test several times and only the first value is stored. Each token goes to
    the single item it matches in the most notes, and only if it matches in >= min_precision of the notes holding it."""
    vals = defaultdict(dict)
    for r in labvals.itertuples():
        vals[int(r.hadm_id)][int(r.itemid)] = float(r.v)
    hits, seen = Counter(), Counter()
    for n in notes:
        lv = vals.get(n["hadm_id"])
        if not lv:
            continue
        toks, matched = set(), set()
        for tok, val in _PAIR.findall(n["labs"]):
            dec = len(val.split(".")[1]) if "." in val else 0
            x = float(val)
            toks.add(tok)
            for itemid, v in lv.items():
                if (round(v, dec) == x) if dec else (v == x):
                    matched.add((tok, itemid))
        seen.update(toks)
        hits.update(matched)
    best = {}
    for (tok, itemid), c in hits.items():
        if c > best.get(tok, (None, 0))[1]:
            best[tok] = (itemid, c)
    out = defaultdict(list)
    for tok, (itemid, c) in best.items():
        prec = c / seen[tok]
        if c >= min_support and prec >= min_precision:
            out[itemid].append((tok, round(prec, 3), c))
    return {k: sorted(v, key=lambda t: (-t[2], -t[1]))[:3] for k, v in out.items()}


def _ngrams(text: str) -> set:
    w = [t for t in re.findall(r"[a-z][a-z0-9'-]+", text) if t not in _GENERIC]
    return set(w) | {" ".join(p) for p in zip(w, w[1:])}


def _title_terms(title: str) -> list[str]:
    t = re.sub(r"\(.*?\)|,.*$|\[.*?\]", "", str(title).lower())
    words = [w for w in re.findall(r"[a-z][a-z'-]+", t) if w not in _GENERIC]
    return [" ".join(words[:4])] if words else []


def extractability(cfg: dict, notes: list[dict], items: list[str], meta: pd.DataFrame, pos: dict[str, set],
                   labvals: pd.DataFrame) -> dict[str, dict]:
    """Per candidate: note terms (mined on training notes) and coverage = share of notes of positive admissions in
    which a term is found (lab: alias with a value; drug: name in Medications on Admission/HPI; history: term in PMH)."""
    mi = meta.set_index("item")
    aliases = mine_lab_aliases(notes, labvals)
    # mined history words must not name a target diagnosis ("hfpef") or a drug ("warfarin"): either leaks the answer
    words = {t.lower() for d in dx_names(cfg) for t in cfg["propositions"][d].get("label_terms", [])}
    words |= {str(c).lower() for c in meta.loc[meta.source == "medication", "code"]}
    banned = re.compile(r"(?<![a-z])(" + "|".join(sorted(map(re.escape, words), key=len, reverse=True)) + r")(?![a-z])")
    by_hadm = {n["hadm_id"]: n for n in notes}
    # history phrases: PMH n-grams enriched in notes of admissions with the previous-admission code
    df_count, pos_count = Counter(), defaultdict(Counter)
    hx_items = [i for i in items if i.startswith("hx")]
    note_grams = {}
    for n in notes:
        g = _ngrams(n["pmh"])
        note_grams[n["hadm_id"]] = g
        df_count.update(g)
        for it in hx_items:
            if n["hadm_id"] in pos[it]:
                pos_count[it].update(g)
    total = max(len(notes), 1)
    out = {}
    for it in items:
        m = mi.loc[it]
        notes_pos = [by_hadm[h] for h in pos[it] if h in by_hadm]
        rec = {"terms": [], "coverage": None, "positive_notes": len(notes_pos)}
        if m["source"] == "demographic":
            rec.update(mode="structured")
        elif m["source"] == "medication":
            rec["terms"] = [m["code"]]
            rx = re.compile(rf"(?<![a-z]){re.escape(m['code'])}(?![a-z])")
            rec["coverage"] = float(np.mean([bool(rx.search(n["meds"])) for n in notes_pos])) if notes_pos else None
        elif m["source"] == "lab":
            al = [a for a, _, _ in aliases.get(int(m["code"]), [])]
            rec["terms"] = al or [str(m["lab_label"])]
            rec["aliases"] = aliases.get(int(m["code"]), [])
            if al and notes_pos:
                rxs = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape, al)) + r")\s*[-:=]\s*\d")
                rec["coverage"] = float(np.mean([bool(rxs.search(n["labs"])) for n in notes_pos]))
            else:
                rec["coverage"] = 0.0 if notes_pos else None
        else:
            cand = []
            n_pos = max(len(notes_pos), 1)
            for g, c in pos_count[it].items():
                if c >= 20:
                    lift = (c / n_pos) / (df_count[g] / total)
                    if lift >= 5 and not banned.search(g):
                        cand.append((lift * math.log(c), g))
            mined = [g for _, g in sorted(cand, reverse=True)[:3]]
            rec["terms"] = list(dict.fromkeys(mined + _title_terms(m["title"])))
            rx = [re.compile(rf"(?<![a-z]){re.escape(t)}(?![a-z])") for t in rec["terms"]]
            rec["coverage"] = float(np.mean([any(r.search(n["pmh"]) for r in rx) for n in notes_pos])) if notes_pos and rx else None
        out[it] = rec
    return out


def _boot_se(Y: np.ndarray, P: np.ndarray, B: int, seed: int) -> float:
    """Bootstrap standard error (over cases) of the macro AUROC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(Y), len(Y))
        a = [roc_auc_score(Y[i, j], P[i, j]) for j in range(Y.shape[1]) if 0 < Y[i, j].sum() < len(i)]
        if a:
            vals.append(np.mean(a))
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# ------------------------------------------------------------------------------------- 5. validation curve
def _prop(it: str, m: dict, ext: dict) -> tuple[str, dict]:
    src = m["source"]
    if src == "history":
        pid = f"hx{m['icd_version']}_{m['code']}"
        return pid, {"kind": "history", "source": "prior", "label": f"history of {str(m['title']).lower()}",
                     f"icd{m['icd_version']}": [str(m["code"])], "note_terms": ext.get("terms") or _title_terms(m["title"]) or [str(m["code"]).lower()],
                     "note_section": ["past medical history"], "pool_item": it}
    if src == "medication":
        return f"rx_{m['code']}", {"kind": "medication", "label": str(m["code"]), "drug_terms": [str(m["code"])],
                                   "note_section": ["medications on admission", "history of present illness"], "pool_item": it}
    if src == "lab":
        return f"lab_{m['code']}_{m['direction']}", {"kind": "lab", "label": str(m["title"]), "itemids": [int(m["code"])],
                                                     "direction": m["direction"], "note_keys": ext.get("terms") or [str(m["lab_label"])],
                                                     "pool_item": it}
    code = str(m["code"])
    return code, ({"kind": "demographic", "label": "female", "sex": "F", "pool_item": it} if code == "sex_f" else
                  {"kind": "demographic", "label": str(m["title"]), "age_ge": int(code.rsplit("_", 1)[1]), "pool_item": it})


def vocab_props(cfg: dict, items: list[str], meta: pd.DataFrame, ext: dict) -> dict:
    """Ordered propositions: demographics and history, the (unchanged) target diagnoses, medications, labs."""
    mi = meta.set_index("item")
    built = [_prop(it, mi.loc[it].to_dict(), ext.get(it, {})) for it in items]
    order = {"demographic": 0, "history": 1, "medication": 3, "lab": 4}
    props = {}
    for pid, p in sorted(built, key=lambda t: order[t[1]["kind"]]):
        if order[p["kind"]] < 2:
            props[pid] = p
    props.update({d: cfg["propositions"][d] for d in dx_names(cfg)})
    for pid, p in sorted(built, key=lambda t: order[t[1]["kind"]]):
        if order[p["kind"]] >= 2:
            props[pid] = p
    return props


def _cfg_with(cfg: dict, props: dict, max_parents: int) -> dict:
    c = copy.deepcopy(cfg)
    c["propositions"] = props
    c["knowledge"]["max_parents"] = max_parents
    return c


def _frame(cfg: dict, props: dict, cohort: pd.DataFrame, X: np.ndarray, items: list[str], rows: np.ndarray) -> pd.DataFrame:
    col = {it: i for i, it in enumerate(items)}
    data = {"subject_id": cohort.subject_id.to_numpy()[rows], "hadm_id": cohort.hadm_id.to_numpy()[rows]}
    for pid, p in props.items():
        data[pid] = cohort[pid].to_numpy()[rows] if p["kind"] == "diagnosis" else X[rows, col[p["pool_item"]]]
    return pd.DataFrame(data)


def _fit_platt(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def discover(cfg: dict, kbdir: Path, art: Path) -> dict:
    d = cfg["discover"]
    out = art / "vocab"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cohort = pd.read_parquet(kbdir / "cohort.parquet")
    inner, val, test = inner_split(cfg, cohort)
    dx = dx_names(cfg)
    pool, meta, labvals = build_pool(cfg, kbdir, out)
    log.info("discover: pool %d rows, %d distinct items (%.0fs)", len(pool), meta.shape[0], time.perf_counter() - t0)
    # frequency floor on inner-train admissions
    inner_hadm = set(cohort.hadm_id[inner])
    pin = pool[pool.hadm_id.isin(inner_hadm)]
    id2item = meta.set_index("item_id")["item"]
    prev = (pin[pin.value == 1].groupby("item_id").size() / max(len(inner_hadm), 1)).rename(index=id2item)
    items = prev[prev >= float(d["min_prevalence"])].sort_values(ascending=False).index.tolist()[: int(d["max_pool"])]
    rng = np.random.default_rng(cfg["seed"])
    rank_rows = np.where(inner)[0]
    if len(rank_rows) > int(d["rank_rows"]):
        rank_rows = np.sort(rng.choice(rank_rows, int(d["rank_rows"]), replace=False))
    hadm_all = cohort.hadm_id.to_numpy()
    Xr = pool_matrix(pool, meta, hadm_all[rank_rows], items)
    Yr = cohort.loc[rank_rows, dx].to_numpy(int)
    order, rstats = rank_items(cfg, Xr, Yr, items)
    log.info("discover: ranked %d items (%.0fs)", len(order), time.perf_counter() - t0)
    mi = meta.set_index("item")
    flags, lifts = {}, {}
    for it in items:
        flags[it], lifts[it] = leak_flags(cfg, mi.loc[it].to_dict(), Xr[:, items.index(it)], Yr)
    # note extractability on training-split notes
    ext, notes = {}, []
    if Path(cfg["paths"]["mimic_notes"]).exists():
        train_hadm = set(cohort.hadm_id[inner | val])
        notes = _note_sample(cfg, train_hadm, set(cohort.hadm_id[val]))
        item2id = meta.set_index("item")["item_id"]
        want = set(item2id.loc[items])
        pp = pool[(pool.value == 1) & pool.item_id.isin(want) & pool.hadm_id.isin(train_hadm)]
        grouped = pp.groupby("item_id")["hadm_id"].apply(set)
        pos = {it: grouped.get(item2id[it], set()) for it in items}
        ext = extractability(cfg, notes, items, meta, pos, labvals)
        for it, e in ext.items():
            if e.get("coverage") is not None and e["coverage"] < float(d["min_note_coverage"]):
                flags[it].append("low_extractability")
        log.info("discover: extractability on %d training notes (%.0fs)", len(notes), time.perf_counter() - t0)
    excluded = {it for it, f in flags.items() if any(x.split(":")[0] in d["exclude_flags"] for x in f)}
    eligible = [it for it in order if it not in excluded]
    # validation curve
    learn_rows = np.where(inner)[0]
    if len(learn_rows) > int(d["learn_rows"]):
        learn_rows = np.sort(rng.choice(learn_rows, int(d["learn_rows"]), replace=False))
    val_rows = np.where(val)[0]
    if len(val_rows) > int(d["val_rows"]):
        val_rows = np.sort(rng.choice(val_rows, int(d["val_rows"]), replace=False))
    need = sorted(set(learn_rows) | set(val_rows))
    Xall = np.zeros((len(cohort), len(items)), np.int8)
    Xall[need] = pool_matrix(pool, meta, hadm_all[need], items)
    Yv = cohort.loc[val_rows, dx].to_numpy(int)
    from sklearn.linear_model import LogisticRegression
    from .quantum import resources, state_prep

    def lr_eval(sel):
        kinds = ["lab" if it.startswith("lab:") else "x" for it in sel]
        cols = [items.index(it) for it in sel]

        def feats(rows):
            M = Xall[np.ix_(rows, cols)].astype(float)
            M[M < 0] = np.nan
            return lr_features(M, kinds)
        Xtr, Xv = feats(learn_rows), feats(val_rows)
        Ytr = cohort.loc[learn_rows, dx].to_numpy(int)
        Pm = np.stack([LogisticRegression(max_iter=2000).fit(Xtr, Ytr[:, i]).predict_proba(Xv)[:, 1] for i in range(len(dx))], 1)
        return metric_set(Yv, Pm, dx, cfg["engine"]["top_k"])

    grid = sorted({int(k) for k in d["k_grid"] if int(k) <= len(eligible)}) or ([len(eligible)] if eligible else [])
    # note-based curve: the keyword extractor reads validation patients' training-split notes with the top-K items
    vnotes = [n for n in notes if "evidence" in n]
    note_M, Yn = None, None
    if vnotes and grid:
        top_props = vocab_props(cfg, eligible[: max(grid)], meta, ext)
        c_all = _cfg_with(cfg, top_props, int(d["parents_grid"][0]))
        lim = {}
        for pid, pp in top_props.items():
            if pp["kind"] == "lab":
                m_ = mi.loc[pp["pool_item"]]
                if pd.notna(m_.get("upper" if pp["direction"] == "high" else "lower")):
                    lim[pid] = {**{b: (float(m_[b]) if pd.notna(m_[b]) else None) for b in ("lower", "upper")},
                                "direction": pp["direction"]}
        hn = np.array([n["hadm_id"] for n in vnotes])
        demo = [(pid, items.index(pp["pool_item"])) for pid, pp in top_props.items() if pp["kind"] == "demographic"]
        Xn = pool_matrix(pool, meta, hn, items) if demo else None
        note_M = []
        for i, n in enumerate(vnotes):
            F = N.baseline_findings(n["evidence"], c_all)
            for pid, pp in top_props.items():
                if pp["kind"] == "lab" and pid not in lim:
                    F[pid] = {"status": "unknown", "value": None, "quote": ""}
            note_M.append(N.memberships(F, c_all, lim, structured={pid: int(Xn[i, j]) for pid, j in demo} if demo else None))
        coh = cohort.set_index("hadm_id")
        Yn = {"note": np.array([[n["labels"][x] for x in dx] for n in vnotes]), "icd": coh.loc[hn, dx].to_numpy(int)}
        log.info("discover: note memberships for %d validation-patient notes (%.0fs)", len(vnotes), time.perf_counter() - t0)
    curve, best, preds, note_preds = [], None, {}, {}
    for K in grid:
        sel = eligible[:K]
        props = vocab_props(cfg, sel, meta, ext)
        row = {"K": K, "qubits": len(props) + 1, "networks": {}}
        nets_k = {}
        for p in d["parents_grid"]:
            c = _cfg_with(cfg, props, int(p))
            bn = nets_k[str(p)] = kb.learn_bn(c, _frame(c, props, cohort, Xall, items, learn_rows))
            fv = _frame(c, props, cohort, Xall, items, val_rows)
            Pm, _ = kb.posteriors_by_pattern(c, bn, fv)
            mset = metric_set(Yv, Pm, dx, cfg["engine"]["top_k"])
            row["networks"][str(p)] = {"macro_auroc": mset["macro_auroc"], "macro_brier": mset["macro_brier"],
                                       f"top{cfg['engine']['top_k']}_hit": mset[f"top{cfg['engine']['top_k']}_hit"],
                                       "edges": int(sum(len(x) for x in bn.struct.parents))}
            preds.setdefault(K, {})[str(p)] = Pm
            if best is None or mset["macro_auroc"] > best["auroc"]:
                best = {"auroc": mset["macro_auroc"], "K": K, "p": int(p)}
        lr = lr_eval(sel)
        row["logistic"] = {"macro_auroc": lr["macro_auroc"], "macro_brier": lr["macro_brier"]}
        if note_M is not None:
            pb = max(row["networks"], key=lambda q: row["networks"][q]["macro_auroc"])
            from .mimicnotes import Posteriors
            post = Posteriors(nets_k[pb], dx)
            keep = [pid for pid, pp in props.items() if pp["kind"] != "diagnosis"]
            Pn = note_preds[K] = np.array([post({pid: m[pid] for pid in keep}) for m in note_M])
            row["notes"] = {"n": len(note_M), "max_parents": int(pb),
                            **{f"macro_auroc_{t}": metric_set(Yn[t], Pn, dx, cfg["engine"]["top_k"])["macro_auroc"] for t in Yn},
                            **{f"macro_brier_{t}": metric_set(Yn[t], Pn, dx, cfg["engine"]["top_k"])["macro_brier"] for t in Yn}}
        curve.append(row)
        log.info("discover: K=%d done (%.0fs)", K, time.perf_counter() - t0)
    ceiling = lr_eval(items) if items else None
    # recommendation: the one-standard-error rule on the note-based curve (the real task) when available, else the
    # structured one: the smallest K whose AUROC is within one bootstrap SE of the best K's (SE over validation cases)
    def one_se(points, Y):
        kb_, ab, Pb = max(points, key=lambda t: t[1])
        se = _boot_se(Y, Pb, int(d.get("se_boot", 200)), cfg["seed"])
        return {"K": min(k for k, a, _ in points if a >= ab - se), "best_K": kb_, "best_auroc": ab, "se": se}
    rec, knees = None, {}
    if curve:
        pts = []
        for r in curve:
            q = max(r["networks"], key=lambda q: r["networks"][q]["macro_auroc"])
            pts.append((r["K"], r["networks"][q]["macro_auroc"], preds[r["K"]][q]))
        knees["structured"] = one_se(pts, Yv)
        if all("notes" in r for r in curve):
            knees["notes"] = one_se([(r["K"], r["notes"]["macro_auroc_note"], note_preds[r["K"]]) for r in curve], Yn["note"])
        kk = (knees.get("notes") or knees["structured"])["K"]
        r = next(r for r in curve if r["K"] == kk)
        pbest, nbest = max(r["networks"].items(), key=lambda kv: kv[1]["macro_auroc"])
        rec = {"K": kk, "max_parents": int(pbest), "macro_auroc": nbest["macro_auroc"], "rule": "notes" if "notes" in knees else "structured",
               "note_auroc": r.get("notes", {}).get("macro_auroc_note")}
    calibration, cal_report, resources_k = None, None, {}
    if rec:
        sel = eligible[: rec["K"]]
        props = vocab_props(cfg, sel, meta, ext)
        c = _cfg_with(cfg, props, rec["max_parents"])
        bn = kb.learn_bn(c, _frame(c, props, cohort, Xall, items, learn_rows))
        Pm, _ = kb.posteriors_by_pattern(c, bn, _frame(c, props, cohort, Xall, items, val_rows))
        half = np.random.default_rng(cfg["seed"] + 5).random(len(val_rows)) < 0.5
        cross = np.zeros_like(Pm)
        for a_, b_ in ((half, ~half), (~half, half)):
            for i in range(len(dx)):
                pa, pb = _fit_platt(Pm[a_, i], Yv[a_, i])
                cross[b_, i] = platt(Pm[b_, i], pa, pb)
        raw_b = metric_set(Yv, Pm, dx, cfg["engine"]["top_k"])["macro_brier"]
        cal_b = metric_set(Yv, cross, dx, cfg["engine"]["top_k"])["macro_brier"]
        cal_report = {"brier_raw": raw_b, "brier_calibrated_crossfit": cal_b, "use": bool(cal_b < raw_b - 1e-4)}
        if cal_report["use"]:
            calibration = {d_: list(_fit_platt(Pm[:, i], Yv[:, i])) for i, d_ in enumerate(dx)}
        resources_k = {"qubits": len(props) + 1, **resources(state_prep(bn))}
    # v1 items for reference
    v1_equiv = {"furosemide": "rx:furosemide", "metoprolol": "rx:metoprolol", "warfarin": "rx:warfarin",
                "creatinine high": "lab:50912:high", "sodium low": "lab:50983:low", "wbc high": "lab:51301:high",
                "hemoglobin low": "lab:51222:low", "history ckd (N18)": "hx10:N18", "history diabetes (E11)": "hx10:E11",
                "history copd (J44)": "hx10:J44"}
    v1_rank = {k: (order.index(v) + 1 if v in order else None) for k, v in v1_equiv.items()}
    cands = []
    for r_, it in enumerate(order, 1):
        m = mi.loc[it]
        cands.append({"rank": r_, "item": it, "source": m["source"], "title": str(m["title"]),
                      "prevalence_inner_train": float(prev[it]), **rstats[it], "lift": lifts[it], "flags": flags[it],
                      "excluded_by_default": it in excluded, "note": ext.get(it, {})})
    (out / "candidates.json").write_text(json.dumps(cands, indent=1, default=float))
    draft = None
    if rec:
        sel = eligible[: rec["K"]]
        options = {"knowledge.max_parents": rec["max_parents"]}
        if calibration:
            options["engine.calibration"] = calibration
        draft = {"name": "v2-draft", "description": "Data-selected vocabulary (draft from qddx discover; edit, then save as "
                 "configs/vocab_v2.json and freeze)", "selection": {"rule": f"top-{rec['K']} mRMR items excluding flags "
                 f"{d['exclude_flags']}; K = smallest K within one bootstrap SE of the best on the {rec['rule']} validation curve",
                 "k": rec["K"], "ranked_items": sel}, "options": options, "propositions": vocab_props(cfg, sel, meta, ext)}
        dpath = Path(d["draft_path"])
        dpath.parent.mkdir(parents=True, exist_ok=True)
        dpath.write_text(json.dumps(draft, indent=1))
    res = {"pool_rows": int(len(pool)), "distinct_items": int(meta.shape[0]), "candidates": len(items),
           "excluded_by_default": len(excluded), "inner_train_admissions": int(inner.sum()), "validation_admissions": int(val.sum()),
           "notes_used": len(notes), "curve": curve, "ceiling_logistic_all_candidates": ceiling and
           {"macro_auroc": ceiling["macro_auroc"], "macro_brier": ceiling["macro_brier"], "features": len(items)},
           "recommended": rec, "knees": knees, "calibration": cal_report, "recommended_resources": resources_k, "v1_item_ranks": v1_rank,
           "top": cands[: int(d["report_top"])], "draft_path": d["draft_path"] if draft else None,
           "seconds": time.perf_counter() - t0}
    write_md(res, cfg, Path(cfg["paths"]["reports"]) / "vocab_discovery.md")
    return res


def write_md(R: dict, cfg: dict, path: Path) -> Path:
    d = cfg["discover"]
    L = ["# Vocabulary discovery (Run 1)", "",
         f"Training patients only: {R['inner_train_admissions']:,} inner-train admissions (ranking, learning) and "
         f"{R['validation_admissions']:,} validation admissions (curve). No test-split admission or held-out note was read.", "",
         f"Candidate pool: {R['distinct_items']:,} distinct admission-time items ({R['pool_rows']:,} admission-item rows); "
         f"{R['candidates']:,} above {d['min_prevalence']:.1%} prevalence (cap {d['max_pool']}); {R['excluded_by_default']} flagged "
         f"and excluded from the draft by default ({', '.join(d['exclude_flags'])}). Training notes used for extractability: {R['notes_used']:,}.", ""]
    has_notes = all("notes" in r for r in R["curve"]) and R["curve"]
    L += ["## Validation curve", "", "Structured columns: networks learned on inner-train admissions, scored on validation "
          "admissions using their coded items. Note columns: the keyword extractor reads validation patients' training-split "
          "notes with the same items (the real task), scored against the note's Discharge Diagnosis section and the ICD codes.", "",
          "| K items | qubits | " + " | ".join(f"network p={p} AUROC / Brier" for p in d["parents_grid"]) + " | logistic AUROC |"
          + (" notes AUROC (discharge dx / ICD) |" if has_notes else ""),
          "|---|---|" + "---|" * (len(d["parents_grid"]) + 1) + ("---|" if has_notes else "")]
    for r in R["curve"]:
        L.append(f"| {r['K']} | {r['qubits']} | " + " | ".join(
            f"{r['networks'][str(p)]['macro_auroc']:.3f} / {r['networks'][str(p)]['macro_brier']:.4f}" for p in d["parents_grid"])
                 + f" | {r['logistic']['macro_auroc']:.3f} |"
                 + (f" {r['notes']['macro_auroc_note']:.3f} / {r['notes']['macro_auroc_icd']:.3f} |" if has_notes else ""))
    if has_notes:
        L.append(f"\nNote-based curve on {R['curve'][0]['notes']['n']:,} validation-patient notes.")
    c = R.get("ceiling_logistic_all_candidates")
    if c:
        L += ["", f"Ceiling: logistic regression on all {c['features']} candidates: AUROC {c['macro_auroc']:.3f}, Brier {c['macro_brier']:.4f}."]
    rec = R.get("recommended")
    if rec:
        kn = R.get("knees", {})
        L += ["", f"**Recommended (one-SE rule: the smallest K whose AUROC is within one bootstrap standard error of the "
              f"best K's, on the {rec.get('rule', 'structured')} curve):** K = {rec['K']}, max parents "
              f"{rec['max_parents']}, structured validation AUROC {rec['macro_auroc']:.3f}"
              + (f", notes AUROC {rec['note_auroc']:.3f}" if rec.get("note_auroc") is not None else "") + ". "
              + " ".join(f"{a.capitalize()}: best K={b['best_K']} ({b['best_auroc']:.3f}, SE {b['se']:.4f}) -> K={b['K']}."
                         for a, b in kn.items())]
        cal = R.get("calibration")
        if cal:
            L.append(f"Calibration (2-fold cross-fit on validation): Brier {cal['brier_raw']:.4f} -> {cal['brier_calibrated_crossfit']:.4f}; "
                     f"{'used' if cal['use'] else 'not used'}.")
        rr = R.get("recommended_resources")
        if rr:
            L.append(f"Circuit for the recommended network: {rr['qubits']} qubits (+1 per graded finding), state preparation depth "
                     f"{rr['depth']:,}, {rr['two_qubit_gates']:,} CX.")
        L.append(f"Draft vocabulary: `{R['draft_path']}`.")
    L += ["", "Where the v1 items rank: " + ", ".join(f"{a} {b if b else 'not in pool'}" for a, b in R["v1_item_ranks"].items()) + ".", "",
          f"## Top {len(R['top'])} candidates", "",
          "| rank | item | source | prevalence | relevance | highest lift | note coverage / terms | flags |", "|---|---|---|---|---|---|---|---|"]
    for cnd in R["top"]:
        lift = max(cnd["lift"].items(), key=lambda kv: kv[1]) if cnd["lift"] else ("", 0)
        note = cnd.get("note") or {}
        cov = "structured" if note.get("mode") == "structured" else (f"{note['coverage']:.0%}" if note.get("coverage") is not None else "n/a")
        L.append(f"| {cnd['rank']} | {cnd['title']} | {cnd['source']} | {cnd['prevalence_inner_train']:.1%} | {cnd['relevance']:.4f} | "
                 f"{lift[0]} x{lift[1]:.1f} | {cov} {', '.join(map(str, note.get('terms', [])[:3]))} | {', '.join(cnd['flags'])}"
                 + (" **(excluded)**" if cnd["excluded_by_default"] else "") + " |")
    L += ["", "Flags (answer-containing or not patient information; rules fixed before the run): same_concept = a "
          "previous-admission ICD category containing a target (a code with the target's prefix, or a code title naming it); "
          f"treatment = drug started in the first 24 h with lift >= {d.get('treatment_lift', 3)} for a target; administrative = "
          "status/encounter/external-cause code; supply = not a drug; routine = order given to >= "
          f"{float(d.get('routine_prevalence', 0.3)):.0%} of admissions; marker = lab specific to one target (kept: labs are "
          "evidence); low_extractability = rarely findable in training notes. Full list: artifacts/vocab/candidates.json."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path


# ------------------------------------------------------------------------------------- after the discussion
def resolve_items(spec: str, drop: str, art: Path) -> list[str]:
    """Items from a spec: comma-separated pool items, 'top:N' (first N unflagged candidates in rank order),
    'recommended' (top:K with discover's recommended K), '@file'
    (one item per line); minus the comma-separated `drop` list. Order is kept, duplicates removed."""
    cands = json.loads((art / "vocab" / "candidates.json").read_text())
    tokens = Path(spec[1:]).read_text().split() if spec.startswith("@") else [t.strip() for t in spec.split(",") if t.strip()]
    out = []
    for t in tokens:
        if t == "recommended":  # the K that discover recommended
            t = f"top:{json.loads((art / 'results' / 'discover.json').read_text())['recommended']['K']}"
        if t.startswith("top:"):
            out += [c["item"] for c in cands if not c["excluded_by_default"]][: int(t[4:])]
        else:
            out.append(t)
    gone = {t.strip() for t in (drop or "").split(",") if t.strip()}
    seen, items = set(), []
    for it in out:
        if it not in gone and it not in seen:
            seen.add(it)
            items.append(it)
    return items


def freeze(cfg: dict, kbdir: Path, art: Path, items: list[str], out_path: Path, name: str = "v2", note: str = "") -> dict:
    """Write the final vocabulary for Run 2 from the items chosen in the discussion. Max parents and calibration are
    chosen on validation patients (training split only), exactly as Run 1 does for its draft."""
    d = cfg["discover"]
    cohort = pd.read_parquet(kbdir / "cohort.parquet")
    inner, val, _ = inner_split(cfg, cohort)
    pool, meta, _ = build_pool(cfg, kbdir, art / "vocab")
    unknown = [it for it in items if it not in set(meta["item"])]
    if unknown:
        raise SystemExit(f"not in the candidate pool: {unknown}")
    cands = json.loads((art / "vocab" / "candidates.json").read_text())
    ext = {c["item"]: c.get("note") or {} for c in cands}
    flags = {c["item"]: c["flags"] for c in cands}
    rng = np.random.default_rng(cfg["seed"] + 11)
    learn_rows, val_rows = np.where(inner)[0], np.where(val)[0]
    if len(learn_rows) > int(d["learn_rows"]):
        learn_rows = np.sort(rng.choice(learn_rows, int(d["learn_rows"]), replace=False))
    if len(val_rows) > int(d["val_rows"]):
        val_rows = np.sort(rng.choice(val_rows, int(d["val_rows"]), replace=False))
    need = sorted(set(learn_rows) | set(val_rows))
    Xall = np.zeros((len(cohort), len(items)), np.int8)
    Xall[need] = pool_matrix(pool, meta, cohort.hadm_id.to_numpy()[need], items)
    dx, k = dx_names(cfg), cfg["engine"]["top_k"]
    Yv = cohort.loc[val_rows, dx].to_numpy(int)
    props = vocab_props(cfg, items, meta, ext)
    scores, nets = {}, {}
    for p in d["parents_grid"]:
        c = _cfg_with(cfg, props, int(p))
        nets[int(p)] = kb.learn_bn(c, _frame(c, props, cohort, Xall, items, learn_rows))
        Pm, _ = kb.posteriors_by_pattern(c, nets[int(p)], _frame(c, props, cohort, Xall, items, val_rows))
        scores[int(p)] = (metric_set(Yv, Pm, dx, k), Pm)
    best = max(scores, key=lambda p: scores[p][0]["macro_auroc"])
    Pm = scores[best][1]
    half = np.random.default_rng(cfg["seed"] + 5).random(len(val_rows)) < 0.5
    cross = np.zeros_like(Pm)
    for a_, b_ in ((half, ~half), (~half, half)):
        for i in range(len(dx)):
            pa, pb = _fit_platt(Pm[a_, i], Yv[a_, i])
            cross[b_, i] = platt(Pm[b_, i], pa, pb)
    raw_b, cal_b = scores[best][0]["macro_brier"], metric_set(Yv, cross, dx, k)["macro_brier"]
    options = {"knowledge.max_parents": best}
    if cal_b < raw_b - 1e-4:
        options["engine.calibration"] = {d_: list(_fit_platt(Pm[:, i], Yv[:, i])) for i, d_ in enumerate(dx)}
    voc = {"name": name, "description": note or "Data-selected vocabulary, frozen after the Run 1 discussion.",
           "selection": {"items": items, "flags": {it: flags.get(it, []) for it in items if flags.get(it)},
                         "chosen_on": "training patients only (inner-train ranking, validation choice of options)"},
           "validation": {"by_max_parents": {str(p): {"macro_auroc": s[0]["macro_auroc"], "macro_brier": s[0]["macro_brier"]}
                                             for p, s in scores.items()},
                          "max_parents": best, "brier_raw": raw_b, "brier_calibrated_crossfit": cal_b,
                          "calibration": "engine.calibration" in options},
           "options": options, "propositions": props}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(voc, indent=1))
    return {"path": str(out_path), "items": len(items), "propositions": len(props), "qubits": len(props) + 1,
            "max_parents": best, "calibration": "engine.calibration" in options, "validation": voc["validation"]}


def compare_vocabs(cfg1: dict, art1: Path, cfg2: dict, art2: Path) -> dict:
    """v2 vs v1 on the same held-out notes: extractor and structured data, both ground truths, paired bootstrap."""
    from .experiments import _stored_memberships, _truths
    from .mimicnotes import Posteriors, _evaluate
    e1 = pd.read_parquet(art1 / "mimic_notes" / "eligible.parquet").set_index("note_id")
    e2 = pd.read_parquet(art2 / "mimic_notes" / "eligible.parquet").set_index("note_id")
    same = set(e1.index) == set(e2.index)
    ids = sorted(set(e1.index) & set(e2.index))
    e1, e2 = e1.loc[ids].reset_index(), e2.loc[ids].reset_index()
    dx = dx_names(cfg2)
    P = {}
    for tag, cfg, art, e in (("v1", cfg1, art1, e1), ("v2", cfg2, art2, e2)):
        bn, _ = kb.load(cfg, art / "kb")
        post = Posteriors(bn, dx, cfg["engine"].get("calibration"))
        P[f"{tag} extractor"] = np.array([post(m) for m in _stored_memberships(cfg, e)])
        S = e[[f"kb_{c}" for c in kb.props(cfg)]].copy()
        S.columns = kb.props(cfg)
        Ps, _ = kb.posteriors_by_pattern(cfg, bn, S)
        cal = cfg["engine"].get("calibration")
        P[f"{tag} structured"] = np.stack([platt(Ps[:, i], *cal[d]) for i, d in enumerate(dx)], 1) if cal else Ps
    pairs = [("v2 extractor", "v1 extractor"), ("v2 structured", "v1 structured")]
    Y = _truths(cfg2, e2)
    B = int(cfg2["mimic_notes"]["bootstrap"])
    agent = {}
    for tag, art in (("v1", art1), ("v2", art2)):
        r = art / "results" / "mimic-notes.json"
        if r.exists():
            s = json.loads(r.read_text()).get("agent_sample", {})
            agent[tag] = {t: {v: m["macro_auroc"]["value"] for v, m in s.get(t, {}).get("variants", {}).items()} for t in ("note", "icd")}
    n1, n2 = len(cfg1["propositions"]), len(cfg2["propositions"])
    return {"same_eligible_notes": same, "notes": len(ids), "propositions": {"v1": n1, "v2": n2},
            "qubits": {"v1": n1 + 1, "v2": n2 + 1},
            "metrics": {t: _evaluate(cfg2, P, Y[t], e2.subject_id.to_numpy(), pairs, B) for t in Y}, "agent_auroc": agent}


def write_compare_md(R: dict, cfg: dict, path: Path) -> Path:
    from .report import _ci, _f
    k = cfg["engine"]["top_k"]
    L = ["# v1 (16 hand-picked) vs v2 (data-selected) vocabulary", "",
         f"Same held-out MIMIC-IV-Note notes ({R['notes']:,}; identical eligible set: {'yes' if R['same_eligible_notes'] else 'NO'}). "
         f"Propositions: v1 {R['propositions']['v1']}, v2 {R['propositions']['v2']} (qubits {R['qubits']['v1']} vs {R['qubits']['v2']}). "
         "v1 is the pre-registered primary result; v2 was selected on training patients only and is a secondary analysis.", ""]
    for t, lab in (("note", "Discharge Diagnosis section"), ("icd", "ICD codes")):
        r = R["metrics"][t]
        L += [f"## Ground truth: {lab}", "", f"| variant | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
        L += [f"| {v} | {_ci(m['macro_auroc'])} | {_ci(m['macro_brier'], '.4f')} | {_ci(m[f'top{k}_hit'])} |" for v, m in r["variants"].items()]
        L += ["", f"| paired difference | macro AUROC | macro Brier | top-{k} hit |", "|---|---|---|---|"]
        L += [f"| {p} | {_ci(m['macro_auroc'], '+.4f')} | {_ci(m['macro_brier'], '+.4f')} | {_ci(m[f'top{k}_hit'], '+.3f')} |"
              for p, m in r["paired"].items()]
        L.append("")
    if R.get("agent_auroc"):
        L += ["## Agent (frozen 1,500-note sample), macro AUROC", "", "| variant | v1 note | v2 note | v1 ICD | v2 ICD |", "|---|---|---|---|---|"]
        names = sorted(set(R["agent_auroc"].get("v1", {}).get("note", {})) | set(R["agent_auroc"].get("v2", {}).get("note", {})))
        A = R["agent_auroc"]
        for v in names:
            cells = [_f(A.get(tag, {}).get(t, {}).get(v)) for tag, t in (("v1", "note"), ("v2", "note"), ("v1", "icd"), ("v2", "icd"))]
            L.append(f"| {v} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    return path
