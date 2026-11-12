"""Quick checks: which parts of the diagnosis problem are hard for classical computers? (v2.1 network, local data,
aggregates printed only). Results go to artifacts/hardness/.

  python scripts/hardness.py lwtail START END      # exact LW sample count per structured test patient (chunk)
  python scripts/hardness.py lwtail-validate       # formula vs empirical LW on 12 patients
  python scripts/hardness.py lwtail-summary
  python scripts/hardness.py lookahead N_NOTES DEPTH [open]   # exact k-step test-ordering lookahead
  python scripts/hardness.py mmap-build N_DX       # network with N_DX ICD diagnosis categories
  python scripts/hardness.py mmap N_DX N_PATIENTS  # exact marginal MAP over the diagnoses (constrained VE)
  python scripts/hardness.py structure N_ITEMS     # greedy structure vs exact optimum (BIC)
"""
import gzip
import json
import os
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from qddx.bn import BN, Evidence, Struct, cond_mi, learn, marginal, posterior_exact, posterior_lw

# any learned network: QDDX_KB=artifacts/<name>/kb QDDX_VOCAB=configs/vocab_<name>.json
KB, VOC = Path(os.environ.get("QDDX_KB", "artifacts/v2.1/kb")), Path(os.environ.get("QDDX_VOCAB", "configs/vocab_v2.1.json"))
OUT = Path("artifacts/hardness")
SEED, Z90, REL = 2026, 1.645, 0.1
OUT.mkdir(parents=True, exist_ok=True)


def load():
    bn = BN.from_dict(json.loads((KB / "bn.json").read_text()))
    P = json.loads(VOC.read_text())["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    ev = [k for k, p in P.items() if p["kind"] != "diagnosis"]
    df = pd.read_parquet(KB / "cohort.parquet")
    subs = np.unique(df.subject_id.to_numpy())
    train = set(np.random.default_rng(SEED).permutation(subs)[: int(round(0.8 * len(subs)))].tolist())  # = kb.split_mask
    is_train = df.subject_id.isin(train).to_numpy()
    return bn, P, dx, ev, df, is_train


# ------------------------------------------------------------------------------------- 3. LW hard tail
def lw_samples(bn: BN, E: Evidence, t: str) -> tuple[float, float, float]:
    """Exact asymptotic LW samples for 90% of runs within REL of p = P(t|E): self-normalised importance sampling has
    relative variance E_q[w^2 (1_t - p)^2] / (N P(E)^2 p^2); E_q[w^2 ...] is a VE on the network with each finding's
    CPT entry squared. -> (samples, p, P(E))."""
    post, pe, _ = posterior_exact(bn, E, [t])
    p = post[t]
    sq = BN(bn.struct, [c.copy() for c in bn.cpts])
    for k, v in E.hard.items():
        j = bn.struct.index(k)
        sq.cpts[j][:, v] = bn.cpts[j][:, v] ** 2
    a0, a1 = marginal(sq, E, [bn.struct.index(t)])
    var = ((1 - p) ** 2 * a1 + p ** 2 * a0) / pe ** 2
    return float((Z90 / REL) ** 2 * var / p ** 2), p, pe


def patient(bn, dx, ev, row):
    E = Evidence({k: int(row[k]) for k in ev if row[k] >= 0}, {})
    post, pe, _ = posterior_exact(bn, E, dx)
    return E, max(post, key=post.get)


def lwtail(start: int, end: int):
    bn, P, dx, ev, df, is_train = load()
    test = df[~is_train].reset_index(drop=True).iloc[start:end]
    rows = []
    for _, r in test.iterrows():
        E, t = patient(bn, dx, ev, r)
        n, p, pe = lw_samples(bn, E, t)
        rows.append({"hadm_id": int(r.hadm_id), "target": t, "p": p, "log10_pe": math.log10(pe), "lw_samples": n,
                     "findings": len(E.hard), "positives": int(sum(E.hard.values()))})
    pd.DataFrame(rows).to_parquet(OUT / f"lwtail_{start:06d}.parquet")
    print(f"{start}-{end}: {len(rows)} patients")


def lwtail_validate():
    bn, P, dx, ev, df, is_train = load()
    test = df[~is_train].sample(n=12, random_state=1)
    grid = [100, 300, 1000, 3000, 10000, 30000, 100000]
    for i, (_, r) in enumerate(test.iterrows()):
        E, t = patient(bn, dx, ev, r)
        n, p, pe = lw_samples(bn, E, t)
        emp = None
        for g in grid:
            errs = [abs(posterior_lw(bn, E, [t], g, SEED + 1000 * k + i)[0][t] - p) / p for k in range(20)]
            if np.quantile(errs, 0.9, method="higher") <= REL:
                emp = g
                break
        print(f"P(E)=1e{math.log10(pe):.1f}  formula {n:,.0f}  empirical first grid point {emp}")


def lwtail_summary():
    d = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("lwtail_*.parquet"))])
    q = d.lw_samples.quantile([0.5, 0.9, 0.99, 0.999]).to_dict()
    print(f"{len(d):,} test patients. LW samples: median {q[0.5]:,.0f}, p90 {q[0.9]:,.0f}, p99 {q[0.99]:,.0f}, "
          f"p99.9 {q[0.999]:,.0f}, max {d.lw_samples.max():,.0f}; >100k: {(d.lw_samples > 1e5).sum()}, >1M: {(d.lw_samples > 1e6).sum()}")
    d["bin"] = pd.cut(d.log10_pe, [-99, -14, -10, -8, -6, -4, -2, 0])
    print(d.groupby("bin", observed=True).lw_samples.describe(percentiles=[0.5, 0.99])[["count", "50%", "99%", "max"]].round(0).to_string())
    top = d.nlargest(int(max(1, len(d) * 0.001)), "lw_samples")
    print("hardest 0.1%: target", top.target.value_counts().to_dict(), "| median P(target|E)", round(top.p.median(), 4),
          "| median log10 P(E)", round(top.log10_pe.median(), 1), "| median findings", top.findings.median(),
          "| median positive findings", top.positives.median(), "(all patients:", d.positives.median(), ")")


# ------------------------------------------------------------------------------------- 2. lookahead
def _test_notes(n: int, df, is_train):
    import csv
    test_h = set(df.hadm_id[~is_train].tolist())
    age = df.set_index("hadm_id")["age_ge_65"].to_dict()
    csv.field_size_limit(10 ** 9)
    import yaml
    cfg = yaml.safe_load(open("configs/qddx.yaml"))
    with gzip.open(cfg["paths"]["mimic_notes"], "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                h = int(r["hadm_id"])
            except ValueError:
                continue
            if h in test_h:
                yield r["text"], {"age_ge_65": int(age[h])}
                n -= 1
                if n == 0:
                    return


def lookahead(n_notes: int, depth: int, open_world: bool):
    import yaml
    from qddx import notes as N
    bn, P, dx, ev, df, is_train = load()
    cfg = yaml.safe_load(open("configs/qddx.yaml"))
    cfg["propositions"] = P
    lim = json.loads((KB / "limits.json").read_text())

    def H(E):
        post, _, _ = posterior_exact(bn, E, dx)
        return sum(-(q * math.log(q) + (1 - q) * math.log(1 - q)) for q in (min(max(v, 1e-12), 1 - 1e-12) for v in post.values()))

    secs, unknowns, calls = [], [], [0]
    for text, demo in _test_notes(n_notes, df, is_train):
        m = N.memberships(N.baseline_findings(N.evidence_text(N.deidentify(text)[0], cfg), cfg), cfg, lim,
                          closed_world=not open_world, structured=demo)
        E0 = Evidence.from_memberships(m)
        unk = [k for k, v in m.items() if v is None]
        memo = {}

        def V(added: tuple, k: int) -> float:
            key = (added, k)
            if key in memo:
                return memo[key]
            E = Evidence({**E0.hard, **dict(added)}, E0.soft)
            calls[0] += 1
            rest = [f for f in unk if f not in dict(added)]
            if k == 0 or not rest:
                v = H(E)
            else:
                v = math.inf
                for f in rest:
                    p1 = posterior_exact(bn, E, [f])[0][f]
                    calls[0] += 1
                    v = min(v, p1 * V(tuple(sorted(added + ((f, 1),))), k - 1) + (1 - p1) * V(tuple(sorted(added + ((f, 0),))), k - 1))
            memo[key] = v
            return v

        t0 = time.perf_counter()
        V((), depth)
        secs.append(time.perf_counter() - t0)
        unknowns.append(len(unk))
    s = np.array(secs)
    print(f"depth {depth} ({'open' if open_world else 'closed'} world), {len(s)} notes: unknown findings median {np.median(unknowns):.0f} "
          f"[{min(unknowns)}-{max(unknowns)}]; seconds/note median {np.median(s):.2f}, p90 {np.quantile(s, 0.9):.2f}, max {s.max():.2f}; "
          f"exact inferences {calls[0]:,}")


# ------------------------------------------------------------------------------------- 1. marginal MAP
def mmap_build(n_dx: int):
    import yaml
    bn0, P, dx0, ev, df, is_train = load()
    cfg = yaml.safe_load(open("configs/qddx.yaml"))
    root = Path(cfg["paths"]["mimic_root"])
    base = root / "hosp" / "diagnoses_icd.csv"
    path = base / "diagnoses_icd.csv" if base.is_dir() else base
    cache = OUT / "dx_categories.parquet"
    if not cache.exists():  # chunked: the table has ~6M rows
        parts = []
        for d in pd.read_csv(path, usecols=["hadm_id", "icd_code", "icd_version"], dtype=str, chunksize=1_000_000):
            parts.append(pd.DataFrame({"hadm_id": d.hadm_id.astype(np.int64),
                                       "cat": (d.icd_version + ":" + d.icd_code.str[:3]).astype("category")}).drop_duplicates())
        d = pd.concat(parts).drop_duplicates()
        d["cat"] = d.cat.astype(str)
        d.to_parquet(cache)
    d = pd.read_parquet(cache)
    d = d[d.hadm_id.isin(df.hadm_id)]
    train_h = set(df.hadm_id[is_train])
    cats = d[d.hadm_id.isin(train_h)].cat.value_counts().index[:n_dx].tolist()
    X = pd.crosstab(d[d.cat.isin(cats)].hadm_id, d[d.cat.isin(cats)].cat).clip(upper=1).reindex(df.hadm_id, fill_value=0)
    hx = [k for k in ev if P[k]["kind"] in ("demographic", "history")]
    rest = [k for k in ev if k not in hx]
    names = hx + [f"dx_{c}" for c in cats] + rest
    Y = np.column_stack([df[hx].to_numpy(np.int8), X[cats].to_numpy(np.int8), df[rest].to_numpy(np.int8)])
    tr = np.where(is_train)[0]
    tr = np.random.default_rng(SEED).choice(tr, min(100000, len(tr)), replace=False)
    y = Y[tr]
    allowed = {j: sorted(range(j), key=lambda c: cond_mi(y, j, c, []), reverse=True)[:8] for j in range(len(names))}
    t0 = time.perf_counter()
    bn = learn(names, y, 4, allowed=allowed, min_gain=1e-3)
    (OUT / f"bn_dx{n_dx}.json").write_text(json.dumps(bn.to_dict()))
    np.save(OUT / f"test_dx{n_dx}.npy", Y[~is_train][np.random.default_rng(SEED).choice((~is_train).sum(), 500, replace=False)])
    print(f"{n_dx} diagnoses: {len(names)} nodes, {sum(len(p) for p in bn.struct.parents)} edges, learned in {time.perf_counter() - t0:.0f}s")


def _factors(bn: BN, hard: dict[int, int], needed: set[int]):
    st, rel, stack = bn.struct, set(), list(needed)
    while stack:
        j = stack.pop()
        if j not in rel:
            rel.add(j)
            stack.extend(st.parents[j])
    fs = []
    for j in rel:
        vs = tuple(st.parents[j]) + (j,)
        t = bn.cpts[j].reshape((2,) * len(vs))[tuple(hard[v] if v in hard else slice(None) for v in vs)]
        fs.append((tuple(v for v in vs if v not in hard), t))
    return fs


def _eliminate(fs, var, op, cap):
    inv = [f for f in fs if var in f[0]]
    out = [f for f in fs if var not in f[0]]
    scope = sorted({v for vs, _ in inv for v in vs})
    if len(scope) > cap:
        raise MemoryError(len(scope))
    L = {v: chr(65 + i) if i < 26 else chr(71 + i) for i, v in enumerate(scope)}
    keep = [v for v in scope if v != var]
    t = np.einsum(",".join("".join(L[v] for v in vs) for vs, _ in inv) + "->" + "".join(L[v] for v in scope), *[t for _, t in inv])
    t = t.max(axis=scope.index(var)) if op == "max" else t.sum(axis=scope.index(var))
    return out + [(tuple(keep), t)], len(scope)


def mmap_value(bn: BN, hard: dict[int, int], mapv: list[int], cap: int = 27):
    """max over the MAP variables of sum over the other unobserved variables of P(all, E): constrained variable
    elimination (sum variables first, then max), greedy min-scope order. -> (value, largest scope)."""
    fs = _factors(bn, hard, set(hard) | set(mapv))
    sumv = {v for vs, _ in fs for v in vs} - set(mapv)
    width = 0
    for group, op in ((sumv, "sum"), ({v for vs, _ in fs for v in vs} & set(mapv), "max")):
        group = set(group)
        while group:
            v = min(group, key=lambda u: len({w for vs, _ in fs if u in vs for w in vs}))
            fs, w = _eliminate(fs, v, op, cap)
            width = max(width, w)
            group.discard(v)
    return float(np.prod([float(t) for _, t in fs])), width


def _upper(bn: BN, hard: dict[int, int], mapv: list[int], cap: int = 27) -> float:
    """Upper bound on marginal MAP: eliminate in an unconstrained greedy order (max for MAP variables, sum for the
    rest); moving a max inside a sum can only increase the value."""
    fs = _factors(bn, hard, set(hard) | set(mapv))
    free = {v for vs, _ in fs for v in vs}
    while free:
        v = min(free, key=lambda u: len({w for vs, _ in fs if u in vs for w in vs}))
        fs, _ = _eliminate(fs, v, "max" if v in mapv else "sum", cap)
        free.discard(v)
    return float(np.prod([float(t) for _, t in fs]))


def mmap_bnb(bn: BN, hard: dict[int, int], mapv: list[int], budget_s: float = 60.0):
    """Depth-first branch and bound over the MAP variables with the unconstrained-order upper bound.
    -> (best value, proved optimal?, nodes, seconds)."""
    t0, nodes = time.perf_counter(), 0
    best = mmap_value(bn, {**hard, **{v: 0 for v in mapv}}, [])[0]  # incumbent: no extra diagnosis
    order = list(mapv)
    stack = [dict()]
    proved = True
    while stack:
        if time.perf_counter() - t0 > budget_s:
            proved = False
            break
        A = stack.pop()
        nodes += 1
        rest = [v for v in order if v not in A]
        if not rest:
            best = max(best, mmap_value(bn, {**hard, **A}, [])[0])
            continue
        kids = []
        for val in (0, 1):
            B = {**A, rest[0]: val}
            ub = _upper(bn, {**hard, **B}, rest[1:])
            if ub > best * (1 + 1e-9):
                kids.append((ub, B))
        for _, B in sorted(kids, key=lambda k: k[0]):  # best bound explored first (pushed last)
            stack.append(B)
    return best, proved, nodes, time.perf_counter() - t0


def mmap(n_dx: int, n_pat: int):
    bn = BN.from_dict(json.loads((OUT / f"bn_dx{n_dx}.json").read_text()))
    Y = np.load(OUT / f"test_dx{n_dx}.npy")[:n_pat]
    st = bn.struct
    mapv = [j for j, n in enumerate(st.names) if n.startswith("dx_")]
    secs, widths, fails = [], [], 0
    for r in Y:
        hard = {j: int(r[j]) for j in range(st.n) if j not in mapv and r[j] >= 0}
        t0 = time.perf_counter()
        try:
            _, w = mmap_value(bn, hard, mapv)
            widths.append(w)
            secs.append(time.perf_counter() - t0)
        except MemoryError as e:
            fails += 1
            widths.append(int(str(e)))
    s = np.array(secs) if secs else np.array([np.nan])
    print(f"{n_dx} diagnoses, {len(Y)} patients: solved {len(secs)}, too large {fails}; seconds median {np.median(s):.3f}, "
          f"max {np.max(s):.3f}; largest factor scope median {np.median(widths):.0f}, max {max(widths)} variables")


def mmap_check():
    """VE marginal MAP vs brute force (all diagnosis settings) on the 6-diagnosis network."""
    bn, P, dx, ev, df, is_train = load()
    st = bn.struct
    mapv = [st.index(d) for d in dx]
    for _, r in df[~is_train].sample(n=5, random_state=3).iterrows():
        hard = {st.index(k): int(r[k]) for k in ev if r[k] >= 0}
        brute = marginal(bn, Evidence({st.names[j]: v for j, v in hard.items()}, {}), mapv).max()
        val, w = mmap_value(bn, hard, mapv)
        print(f"brute {brute:.6e}  VE {val:.6e}  rel diff {abs(val - brute) / brute:.1e}")


# ------------------------------------------------------------------------------------- 5. structure
def _local(y: np.ndarray, j: int, pa: tuple) -> float:
    """BIC local score of node j with parent set pa (complete rows)."""
    idx = np.zeros(len(y), np.int64)
    for p in pa:
        idx = idx * 2 + y[:, p]
    c = np.bincount(idx * 2 + y[:, j], minlength=2 ** (len(pa) + 1)).reshape(-1, 2).astype(float)
    n = c.sum(1, keepdims=True)
    ll = float(np.where(c > 0, c * np.log(np.where(c > 0, c, 1) / np.where(n > 0, n, 1)), 0).sum())
    return ll - 0.5 * math.log(len(y)) * 2 ** len(pa)


def structure(n_items: int, max_pa: int = 4):
    bn, P, dx, ev, df, is_train = load()
    names = list(bn.struct.names)
    sel = names[:1] + [n for n in names if P[n]["kind"] == "history"][: (n_items - 7) // 2 + 1] + dx
    sel += [n for n in names if P[n]["kind"] in ("medication", "lab") and n not in sel][: n_items - len(sel)]
    sel = [n for n in names if n in sel]  # keep the network's order
    y = df.loc[is_train, sel].to_numpy(np.int8)
    y = y[(y >= 0).all(1)]
    y = y[np.random.default_rng(SEED).choice(len(y), min(100000, len(y)), replace=False)].astype(np.int64)
    n = len(sel)
    t0 = time.perf_counter()
    g = learn(sel, y, max_pa, min_gain=1e-3)
    greedy = sum(_local(y, j, g.struct.parents[j]) for j in range(n))
    t_greedy = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc = {j: {pa: _local(y, j, pa) for k in range(max_pa + 1) for pa in combinations([c for c in range(n) if c != j], k)} for j in range(n)}
    t_scores = time.perf_counter() - t0
    ordered = sum(max(v for pa, v in sc[j].items() if all(p < j for p in pa)) for j in range(n))
    # global optimum over all DAGs (Silander-Myllymaki): best parent set within every candidate set, then best order
    t0 = time.perf_counter()
    full = 1 << n
    best = {}
    for j in range(n):
        b = np.full(full, -np.inf)
        for pa, v in sc[j].items():
            m = sum(1 << p for p in pa)
            b[m] = max(b[m], v)
        for k in range(n):  # subset maximum (sum over subsets DP)
            has = (np.arange(full) >> k) & 1 == 1
            b[has] = np.maximum(b[has], b[np.arange(full)[has] ^ (1 << k)])
        best[j] = b
    F = np.full(full, -np.inf)
    F[0] = 0.0
    masks = np.arange(full)
    pc = np.array([bin(m).count("1") for m in range(full)])
    for size in range(1, n + 1):
        cur = masks[pc == size]
        val = np.full(len(cur), -np.inf)
        for j in range(n):
            has = (cur >> j) & 1 == 1
            prev = cur[has] ^ (1 << j)
            val[has] = np.maximum(val[has], F[prev] + best[j][prev])
        F[cur] = val
    glob = float(F[full - 1])
    t_dp = time.perf_counter() - t0
    print(f"{n} items, {len(y):,} complete training rows, max {max_pa} parents. BIC: greedy (our order) {greedy:,.1f}; "
          f"exact best in our order {ordered:,.1f}; exact best over all DAGs {glob:,.1f}. Gap greedy->global "
          f"{glob - greedy:,.1f} ({(glob - greedy) / abs(glob):.3%}). Seconds: greedy {t_greedy:.1f}, local scores {t_scores:.1f}, DP {t_dp:.1f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    {"lwtail": lambda: lwtail(int(a[1]), int(a[2])), "lwtail-validate": lwtail_validate, "lwtail-summary": lwtail_summary,
     "lookahead": lambda: lookahead(int(a[1]), int(a[2]), len(a) > 3 and a[3] == "open"),
     "mmap-build": lambda: mmap_build(int(a[1])), "mmap": lambda: mmap(int(a[1]), int(a[2])), "mmap-check": mmap_check,
     "structure": lambda: structure(int(a[1]))}[a[0]]()
