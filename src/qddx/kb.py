"""Knowledge base from MIMIC-IV hosp: one binary row per adult admission, data-derived lab limits,
a learned Bayesian network, and the MIMIC-scale evaluation of that network.

Row semantics (per admission):
  history / diagnosis : 1 if an ICD code of the admission starts with a configured prefix, else 0
  medication          : 1 if a matching prescription starts in the first 24 h, else 0
  lab                 : first value in [admit - 12 h, admit + 24 h] outside the row's reference range
                        (population median range when the row has none); -1 if not measured
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .bn import BN, Evidence, cond_mi, learn, posterior_exact

log = logging.getLogger("qddx")


def props(cfg: dict, *kinds: str) -> list[str]:
    return [k for k, v in cfg["propositions"].items() if not kinds or v["kind"] in kinds]


def evidence(cfg: dict) -> list[str]:
    """All non-diagnosis propositions (the findings), in vocabulary order."""
    return [k for k, v in cfg["propositions"].items() if v["kind"] != "diagnosis"]


def find_table(root: Path, name: str) -> Path:
    """hosp/<name>.csv, .csv.gz, or the nested hosp/<name>.csv/<name>.csv layout."""
    base = Path(root) / "hosp"
    for c in (base / f"{name}.csv", base / f"{name}.csv.gz"):
        if c.is_file():
            return c
        if c.is_dir():
            for inner in (c / f"{name}.csv", c / f"{name}.csv.gz"):
                if inner.is_file():
                    return inner
    raise FileNotFoundError(f"MIMIC table '{name}' not found under {base}")


def _scan(root: Path, name: str) -> str:
    p = find_table(root, name).as_posix().replace("'", "''")
    return f"read_csv('{p}', header=true, all_varchar=true)"


def _like_any(col: str, pats: list, contains: bool = False) -> str:
    if not pats:
        return "FALSE"
    w = "%{}%" if contains else "{}%"
    return "(" + " OR ".join(f"{col} LIKE '{w.format(str(p).lower() if contains else p)}'" for p in pats) + ")"


def connect(cfg: dict) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    mem = cfg["compute"].get("duckdb_memory_limit", "auto")
    if mem and mem != "auto":
        con.execute(f"SET memory_limit='{mem}'")
    if cfg["compute"].get("duckdb_threads"):
        con.execute(f"SET threads={int(cfg['compute']['duckdb_threads'])}")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path(cfg["paths"]["artifacts"]) / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def _q(n: str) -> str:
    """Quoted SQL identifier: proposition ids may hold spaces ("rx_potassium chloride")."""
    return '"' + n.replace('"', '""') + '"'


def extract_cohort(cfg: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    """-> (rows [subject_id, hadm_id, <propositions>], lab limits {prop: {lower, upper}})."""
    k = cfg["knowledge"]
    P = cfg["propositions"]
    con = connect(cfg)
    lo_h, hi_h = k["lab_window_hours"]
    mlo, mhi = k["med_window_hours"]
    con.execute(f"""
        CREATE TEMP TABLE adm AS
        SELECT CAST(a.subject_id AS BIGINT) subject_id, CAST(a.hadm_id AS BIGINT) hadm_id,
               CAST(a.admittime AS TIMESTAMP) admittime,
               CAST(p.anchor_age AS INT) + (year(CAST(a.admittime AS TIMESTAMP)) - CAST(p.anchor_year AS INT)) age,
               p.gender sex
        FROM {_scan(root, 'admissions')} a JOIN {_scan(root, 'patients')} p USING (subject_id)
        WHERE CAST(p.anchor_age AS INT) + (year(CAST(a.admittime AS TIMESTAMP)) - CAST(p.anchor_year AS INT)) >= {int(k['adult_age'])}""")
    log.info("adult admissions: %d", con.execute("SELECT count(*) FROM adm").fetchall()[0][0])

    def icd_cond(n):
        return (f"(icd_version = '10' AND {_like_any('icd_code', P[n].get('icd10', []))}) OR "
                f"(icd_version = '9' AND {_like_any('icd_code', P[n].get('icd9', []))})")

    same = [n for n in props(cfg, "history", "diagnosis") if P[n].get("source", "same") == "same"]
    prior = [n for n in props(cfg, "history") if P[n].get("source") == "prior"]
    dx_cols = ",\n".join(f"bool_or({icd_cond(n)})::INT AS {_q(n)}" for n in same)
    con.execute(f"""CREATE TEMP TABLE dx AS SELECT CAST(hadm_id AS BIGINT) hadm_id, {dx_cols}
                    FROM {_scan(root, 'diagnoses_icd')} GROUP BY 1""")
    if prior:  # history known at admission: codes of the patient's earlier admissions (discharged before this one)
        pr_cols = ",\n".join(f"bool_or({icd_cond(n)})::INT AS {_q(n)}" for n in prior)
        con.execute(f"""CREATE TEMP TABLE prior_dx AS
            WITH p AS (SELECT CAST(subject_id AS BIGINT) subject_id, CAST(hadm_id AS BIGINT) hadm_id,
                              CAST(dischtime AS TIMESTAMP) dischtime FROM {_scan(root, 'admissions')}),
                 d AS (SELECT CAST(hadm_id AS BIGINT) hadm_id, icd_code, icd_version FROM {_scan(root, 'diagnoses_icd')})
            SELECT a.hadm_id, {pr_cols}
            FROM adm a JOIN p ON p.subject_id = a.subject_id AND p.dischtime <= a.admittime JOIN d ON d.hadm_id = p.hadm_id
            GROUP BY 1""")
    else:
        con.execute("CREATE TEMP TABLE prior_dx AS SELECT hadm_id FROM adm WHERE false")

    meds = props(cfg, "medication")
    if meds:
        med_cols = ",\n".join(f"bool_or({_like_any('lower(r.drug)', P[n]['drug_terms'], contains=True)})::INT AS {_q(n)}" for n in meds)
        con.execute(f"""CREATE TEMP TABLE med AS
            SELECT a.hadm_id, {med_cols}
            FROM {_scan(root, 'prescriptions')} r JOIN adm a ON CAST(r.hadm_id AS BIGINT) = a.hadm_id
            WHERE TRY_CAST(r.starttime AS TIMESTAMP) >= a.admittime + INTERVAL ({mlo}) HOUR
              AND TRY_CAST(r.starttime AS TIMESTAMP) <= a.admittime + INTERVAL ({mhi}) HOUR
            GROUP BY 1""")
    else:
        con.execute("CREATE TEMP TABLE med AS SELECT hadm_id FROM adm WHERE false")

    labs = props(cfg, "lab")
    if not labs:
        con.execute("CREATE TEMP TABLE labw AS SELECT hadm_id FROM adm WHERE false")
        limits = {}
    else:
        limits = _lab_table(con, root, cfg, labs, lo_h, hi_h)
    demo = props(cfg, "demographic")
    sel = ", ".join([f"coalesce(dx.{_q(n)}, 0) {_q(n)}" for n in same] + [f"coalesce(prior_dx.{_q(n)}, 0) {_q(n)}" for n in prior]
                    + [f"coalesce(med.{_q(n)}, 0) {_q(n)}" for n in meds] + [f"coalesce(labw.{_q(n)}, -1) {_q(n)}" for n in labs]
                    + [(f"(adm.age >= {int(P[n]['age_ge'])})::INT {_q(n)}" if "age_ge" in P[n] else f"(adm.sex = '{P[n]['sex']}')::INT {_q(n)}")
                       for n in demo])
    df = con.execute(f"""SELECT adm.subject_id, adm.hadm_id, {sel} FROM adm
        LEFT JOIN dx USING (hadm_id) LEFT JOIN prior_dx USING (hadm_id) LEFT JOIN med USING (hadm_id)
        LEFT JOIN labw USING (hadm_id) ORDER BY adm.hadm_id""").df()
    con.close()
    for n in props(cfg):
        df[n] = df[n].astype("int8")
    return df[["subject_id", "hadm_id", *props(cfg)]], limits


def _lab_table(con, root: Path, cfg: dict, labs: list[str], lo_h, hi_h) -> dict:
    """First value of each lab proposition in the window, abnormal vs the row's range (population median if none).
    One itemid may serve several propositions (e.g. high and low)."""
    P = cfg["propositions"]
    item_rows = ", ".join(f"({int(i)}, '{n}')" for n in labs for i in P[n]["itemids"])
    con.execute(f"CREATE TEMP TABLE item_map AS SELECT * FROM (VALUES {item_rows}) t(itemid, prop)")
    ids = ", ".join(f"'{int(i)}'" for n in labs for i in P[n]["itemids"])
    con.execute(f"""CREATE TEMP TABLE lab AS
        SELECT CAST(l.subject_id AS BIGINT) subject_id, m.prop, TRY_CAST(l.charttime AS TIMESTAMP) charttime,
               TRY_CAST(l.valuenum AS DOUBLE) v, TRY_CAST(l.ref_range_lower AS DOUBLE) rlo,
               TRY_CAST(l.ref_range_upper AS DOUBLE) rhi
        FROM {_scan(root, 'labevents')} l JOIN item_map m ON CAST(l.itemid AS INT) = m.itemid
        WHERE l.itemid IN ({ids}) AND TRY_CAST(l.valuenum AS DOUBLE) IS NOT NULL""")
    limits = {}
    for n in labs:
        lo, hi = con.execute(f"""SELECT median(rlo) FILTER (WHERE rlo > 0), median(rhi) FILTER (WHERE rhi > 0)
                                 FROM lab WHERE prop = '{n}'""").fetchall()[0]
        limits[n] = {"lower": lo, "upper": hi, "direction": P[n]["direction"]}
    lim_rows = ", ".join(f"('{n}', {limits[n]['lower'] or 'NULL'}, {limits[n]['upper'] or 'NULL'})" for n in labs)
    con.execute(f"CREATE TEMP TABLE lim AS SELECT * FROM (VALUES {lim_rows}) t(prop, plo, phi)")
    abn = " ".join(f"WHEN '{n}' THEN " + ("f.v > coalesce(f.rhi, lim.phi)" if P[n]["direction"] == "high" else "f.v < coalesce(f.rlo, lim.plo)")
                   for n in labs)
    con.execute(f"""CREATE TEMP TABLE firstlab AS
        WITH w AS (
          SELECT a.hadm_id, l.prop, arg_min(l.v, l.charttime) v, arg_min(l.rlo, l.charttime) rlo, arg_min(l.rhi, l.charttime) rhi
          FROM lab l JOIN adm a ON l.subject_id = a.subject_id
           AND l.charttime >= a.admittime + INTERVAL ({lo_h}) HOUR AND l.charttime <= a.admittime + INTERVAL ({hi_h}) HOUR
          GROUP BY 1, 2)
        SELECT f.hadm_id, f.prop, (CASE f.prop {abn} END)::INT abnormal FROM w f JOIN lim USING (prop)""")
    lab_cols = ", ".join(f"max(abnormal) FILTER (WHERE prop = '{n}') AS {_q(n)}" for n in labs)
    con.execute(f"CREATE TEMP TABLE labw AS SELECT hadm_id, {lab_cols} FROM firstlab GROUP BY 1")
    return limits


def split_mask(subject_ids: np.ndarray, frac_train: float, seed: int) -> np.ndarray:
    """Subject-level split: True = train. Deterministic in (subject set, seed)."""
    subs = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    train = set(rng.permutation(subs)[: int(round(frac_train * len(subs)))].tolist())
    return np.array([s in train for s in subject_ids])


def learn_bn(cfg: dict, df: pd.DataFrame, max_parents: int | None = None) -> BN:
    """Structure and CPTs from the given rows. For large vocabularies each node's parent search is restricted to the
    `knowledge.parent_candidates` earlier nodes with the highest mutual information (on up to
    `knowledge.structure_rows` rows), which keeps learning near-linear in the number of propositions."""
    names = props(cfg)
    k = cfg["knowledge"]
    y = df[names].to_numpy(np.int8)
    allowed = None
    if len(names) > int(k.get("prefilter_above", 24)):
        rows = y if len(y) <= int(k.get("structure_rows", 100000)) else \
            y[np.random.default_rng(cfg["seed"]).choice(len(y), int(k.get("structure_rows", 100000)), replace=False)]
        lim = int(k.get("parent_candidates", 8))
        allowed = {}
        for j in range(len(names)):
            mi = {c: cond_mi(rows, j, c, []) for c in range(j)}
            allowed[j] = sorted(mi, key=mi.get, reverse=True)[:lim]
    return learn(names, y, max_parents if max_parents is not None else k["max_parents"], allowed=allowed,
                 min_gain=k["min_gain"])


def edges(bn: BN) -> list[tuple[str, str]]:
    st = bn.struct
    return [(st.names[p], st.names[j]) for j, pa in enumerate(st.parents) for p in pa]


def build(cfg: dict, out: Path) -> dict:
    """Extract (cached), split, learn. Writes cohort.parquet (patient-level), bn.json, bn_naive.json, limits.json."""
    out.mkdir(parents=True, exist_ok=True)
    cohort = out / "cohort.parquet"
    if cohort.exists() and (out / "limits.json").exists():
        df = pd.read_parquet(cohort)
        limits = json.loads((out / "limits.json").read_text())
    else:
        df, limits = extract_cohort(cfg, Path(cfg["paths"]["mimic_root"]))
        df.to_parquet(cohort.with_suffix(".tmp"))
        os.replace(cohort.with_suffix(".tmp"), cohort)
        (out / "limits.json").write_text(json.dumps(limits, indent=1))
    tr = split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])
    bn = learn_bn(cfg, df[tr])
    naive = learn_bn(cfg, df[tr], max_parents=1)
    (out / "bn.json").write_text(json.dumps(bn.to_dict()))
    (out / "bn_naive.json").write_text(json.dumps(naive.to_dict()))
    names = props(cfg)
    summary = {"admissions": int(len(df)), "subjects": int(df.subject_id.nunique()),
               "train_admissions": int(tr.sum()), "test_admissions": int((~tr).sum()),
               "prevalence_train": {n: float((df.loc[tr, n] == 1).sum() / max((df.loc[tr, n] >= 0).sum(), 1)) for n in names},
               "missing_train": {n: float((df.loc[tr, n] < 0).mean()) for n in names},
               "edges": edges(bn), "edges_naive": edges(naive), "limits": limits}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def load(cfg: dict, out: Path, naive: bool = False) -> tuple[BN, dict]:
    bn = BN.from_dict(json.loads((out / ("bn_naive.json" if naive else "bn.json")).read_text()))
    return bn, json.loads((out / "limits.json").read_text())


def test_split(cfg: dict, out: Path) -> pd.DataFrame:
    df = pd.read_parquet(out / "cohort.parquet")
    return df[~split_mask(df.subject_id.to_numpy(), cfg["knowledge"]["split"][0], cfg["seed"])].reset_index(drop=True)


def posteriors_by_pattern(cfg: dict, bn: BN, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Exact P(diagnosis | history, meds, labs) for every row, computed once per unique evidence pattern.
    -> (posteriors [rows, n_dx], P(E) [rows])."""
    ev_names = evidence(cfg)
    dx = props(cfg, "diagnosis")
    E = df[ev_names].to_numpy(np.int8)
    pats, inv = np.unique(E, axis=0, return_inverse=True)
    post = np.zeros((len(pats), len(dx)))
    pe = np.zeros(len(pats))
    for i, p in enumerate(pats):
        o, pe[i], _ = posterior_exact(bn, Evidence({n: int(v) for n, v in zip(ev_names, p) if v >= 0}, {}), dx)
        post[i] = [o[d] for d in dx]
    inv = inv.reshape(-1)
    return post[inv], pe[inv]
