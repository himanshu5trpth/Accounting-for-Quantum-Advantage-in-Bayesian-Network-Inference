"""Day-one check for the head-to-head study: on real held-out patients (full admission evidence), how many samples
does likelihood weighting (LW) need to get P(top diagnosis | findings) within 10% relative error (90% of repeats),
and what would plain amplitude estimation (AE) and AE on LW's proposal with per-node weight rescaling (qLW) cost?

qLW circuit: findings clamped, other nodes sampled from their CPTs (LW's proposal), one ancilla per finding rotated
to sqrt(P(e | parents) / m_e), m_e = max over that finding's unobserved parent settings; flag = AND of ancillas.
P(flag) = P(E) / prod m_e, so AE's cost ~ 1/sqrt(P(E) / prod m_e) instead of 1/sqrt(P(E)). One state preparation =
one LW sample. AE costs are the analytic amplification law (same as the advantage stage). Prints aggregates only.

python scripts/lw_check.py BIN_INDEX [per_bin]   -> appends rows to artifacts/lw_check/rows.jsonl
python scripts/lw_check.py summary
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qddx.bn import BN, Evidence, posterior_exact, posterior_lw
from qddx.quantum import iae_emulated, iae_relative

# any learned network: QDDX_KB=artifacts/<name>/kb QDDX_VOCAB=configs/vocab_<name>.json
KB, VOC = Path(os.environ.get("QDDX_KB", "artifacts/v2.1/kb")), Path(os.environ.get("QDDX_VOCAB", "configs/vocab_v2.1.json"))
OUT = Path("artifacts/lw_check")
BINS = [(-2, -4), (-4, -6), (-6, -8), (-8, -10), (-10, -14), (-14, -40)]  # log10 P(E)
REL, REPEATS, GRID = 0.1, 10, [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
SEED, EPS0 = 2026, 0.01
SHOTS, ALPHA = int(os.environ.get("AE_SHOTS", 100)), float(os.environ.get("AE_ALPHA", 0.05))  # IAE shots per round, CI level


def setup():
    bn = BN.from_dict(json.loads((KB / "bn.json").read_text()))
    P = json.loads(VOC.read_text())["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    ev = [k for k, p in P.items() if p["kind"] != "diagnosis"]
    df = pd.read_parquet(KB / "cohort.parquet")
    subs = np.unique(df.subject_id.to_numpy())
    train = set(np.random.default_rng(SEED).permutation(subs)[: int(round(0.8 * len(subs)))].tolist())  # = kb.split_mask
    test = df[~df.subject_id.isin(train)].sample(n=4000, random_state=SEED)
    return bn, dx, ev, test


def node_max(bn: BN, j: int, e: int, obs: dict[int, int]) -> float:
    """max over the unobserved parents of P(x_j = e | parents), observed parents fixed."""
    pa = bn.struct.parents[j]
    best = 0.0
    for r in range(2 ** len(pa)):
        bits = [(r >> (len(pa) - 1 - i)) & 1 for i in range(len(pa))]  # first parent = most significant bit
        if all(obs.get(p, b) == b for p, b in zip(pa, bits)):
            q = bn.cpts[j][r, 1]
            best = max(best, q if e == 1 else 1 - q)
    return best


def ae_cost(a_e: float, a_te: float, p: float, seed: int) -> float | None:
    """Mean model evaluations for 90% of repeats to reach REL on P(t|E) = a_te / a_e (relative-precision IAE)."""
    for f in (2, 1, 0.5, 0.25, 0.125, 0.0625):
        errs, calls = [], []
        for r in range(REPEATS):
            s = seed + 7919 * r
            e1 = iae_relative(lambda eps: iae_emulated(a_e, eps, ALPHA, SHOTS, s), EPS0, f * REL)
            e2 = iae_relative(lambda eps: iae_emulated(a_te, eps, ALPHA, SHOTS, s + 1), EPS0, f * REL)
            errs.append(abs(e2["estimate"] / e1["estimate"] - p) / p if e1["estimate"] > 0 else np.inf)
            calls.append(e1["oracle_queries"] + e2["oracle_queries"])
        if np.quantile(errs, 0.9, method="higher") <= REL:
            return float(np.mean(calls))
    return None


def run_bin(b: int, per_bin: int):
    bn, dx, ev, test = setup()
    hi, lo = BINS[b]
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, (_, row) in enumerate(test.iterrows()):
        E = Evidence({k: int(row[k]) for k in ev if row[k] >= 0}, {})
        post, pe, _ = posterior_exact(bn, E, dx)
        if not (lo < np.log10(pe) <= hi):
            continue
        t = max(post, key=post.get)
        p = post[t]
        lw = None
        for n in GRID:
            errs = [abs(posterior_lw(bn, E, [t], n, SEED + 1000 * r + i)[0][t] - p) / p for r in range(REPEATS)]
            if np.quantile(errs, 0.9, method="higher") <= REL:
                lw = n
                break
        obs = {bn.struct.index(k): v for k, v in E.hard.items()}
        logW = sum(np.log(node_max(bn, j, v, obs)) for j, v in obs.items())
        rec = {"bin": b, "log10_pe": float(np.log10(pe)), "p": p, "findings": len(obs), "lw": lw,
               "log10_W": float(logW / np.log(10)),
               "ae": ae_cost(pe, pe * p, p, SEED + i), "qlw": ae_cost(pe / np.exp(logW), pe * p / np.exp(logW), p, SEED + i)}
        rec.update(shots=SHOTS, alpha=ALPHA, kb=str(KB))
        with open(OUT / "rows.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        done += 1
        if done >= per_bin:
            break
    print(f"bin {BINS[b]}: {done} patients")


def summary():
    df = pd.DataFrame([json.loads(x) for x in (OUT / "rows.jsonl").read_text().splitlines()])
    for c, v in (("kb", "artifacts/v2/kb"), ("shots", 100), ("alpha", 0.05)):  # rows written before these were recorded
        df[c] = df[c].fillna(v) if c in df else v
    df = df[(df.kb == str(KB)) & (df.shots == SHOTS) & (df.alpha == ALPHA)]
    print(f"{KB}, IAE shots {SHOTS}, alpha {ALPHA}")
    print("bin (log10 P(E)) | n | LW samples median [min-max] | AE median | qLW median | qLW/LW median | log10 prod m_e median")
    for b, g in df.groupby("bin"):
        r = (g.qlw / g.lw).median()
        print(f"{BINS[b]} | {len(g)} | {g.lw.median():,.0f} [{g.lw.min():,.0f}-{g.lw.max():,.0f}] censored {g.lw.isna().sum()} | "
              f"{g.ae.median():,.0f} | {g.qlw.median():,.0f} | {r:.2f} | {g.log10_W.median():.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run_bin(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8)
