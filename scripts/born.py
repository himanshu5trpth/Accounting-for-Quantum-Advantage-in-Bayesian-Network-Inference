"""Quantum-native model check: a circuit Born machine vs classical generative models of similar size, on 16 items
(top 10 v2.1 findings + 6 diagnoses), at training sizes 500 / 2k / 10k / 50k admissions (training-split patients;
held-out = test-split patients). Every model gives the exact joint over all 2^16 states, so log-likelihood and
P(diagnosis | findings) are exact. Aggregates only.

  python scripts/born.py run SIZE SEED classical   # network, autoregressive, RBM, logistic regression
  python scripts/born.py run SIZE SEED born        # Born machine; resumable, repeat until it prints "done"
  python scripts/born.py summary
"""
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qddx.bn import learn

# any learned network: QDDX_KB=artifacts/<name>/kb QDDX_VOCAB=configs/vocab_<name>.json
KB, VOC = Path(os.environ.get("QDDX_KB", "artifacts/v2.1/kb")), Path(os.environ.get("QDDX_VOCAB", "configs/vocab_v2.1.json"))
OUT = Path("artifacts/born")
SEED, N_EVID, N_TEST = 2026, 10, 20000
torch.set_num_threads(2)


def data():
    v = json.loads(VOC.read_text())
    P = v["propositions"]
    pmap = {p["pool_item"]: k for k, p in P.items() if "pool_item" in p}
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    evid = [pmap[i] for i in v["selection"]["items"] if i in pmap][:N_EVID]
    names = [k for k in P if k in evid or k in dx]  # vocabulary order: demographics, history, diagnoses, meds, labs
    df = pd.read_parquet(KB / "cohort.parquet")
    subs = np.unique(df.subject_id.to_numpy())
    train = set(np.random.default_rng(SEED).permutation(subs)[: int(round(0.8 * len(subs)))].tolist())
    X = df[names].to_numpy(np.int8)
    ok = (X >= 0).all(1)
    tr = df.subject_id.isin(train).to_numpy() & ok
    te = ~df.subject_id.isin(train).to_numpy() & ok
    Xte = X[te][np.random.default_rng(SEED).choice(te.sum(), N_TEST, replace=False)]
    return names, dx, X[tr], Xte


def states(n):
    s = np.arange(2 ** n)
    return np.stack([(s >> (n - 1 - j)) & 1 for j in range(n)], 1)  # names[0] = most significant bit


def index(X):
    n = X.shape[1]
    return (X.astype(np.int64) * (1 << np.arange(n - 1, -1, -1))).sum(1)


# ------------------------------------------------------------------------------------- models (log p over all states)
class FVSBN(torch.nn.Module):  # autoregressive: logistic regression on all earlier items
    def __init__(self, n):
        super().__init__()
        self.W, self.b, self.mask = torch.nn.Parameter(torch.zeros(n, n)), torch.nn.Parameter(torch.zeros(n)), torch.tril(torch.ones(n, n), -1)

    def params(self):
        n = len(self.b)
        return n * (n - 1) // 2 + n

    def forward(self, S):
        a = self.b + S @ (self.W * self.mask).T
        return (S * torch.nn.functional.logsigmoid(a) + (1 - S) * torch.nn.functional.logsigmoid(-a)).sum(1)


class RBM(torch.nn.Module):
    def __init__(self, n, h):
        super().__init__()
        self.W = torch.nn.Parameter(0.01 * torch.randn(n, h))
        self.b, self.c = torch.nn.Parameter(torch.zeros(n)), torch.nn.Parameter(torch.zeros(h))

    def params(self):
        return self.W.numel() + len(self.b) + len(self.c)

    def forward(self, S):
        u = S @ self.b + torch.nn.functional.softplus(self.c + S @ self.W).sum(1)
        return u - torch.logsumexp(u, 0)


class Born(torch.nn.Module):
    """Circuit Born machine: real RY rotations on every qubit, then CZ gates on all pairs, L times, and a final RY layer;
    p(x) = |<x|psi>|^2. First layer starts at the data marginals (product state)."""

    def __init__(self, n, L, marg, S):
        super().__init__()
        th = 0.05 * torch.randn(L + 1, n, dtype=torch.float64)
        th[0] = 2 * torch.asin(torch.sqrt(torch.as_tensor(marg, dtype=torch.float64).clamp(1e-3, 1 - 1e-3)))
        self.th, self.n, self.L = torch.nn.Parameter(th), n, L
        k = torch.as_tensor(S, dtype=torch.float64).sum(1)
        self.cz = 1 - 2 * ((k * (k - 1) / 2) % 2)  # CZ on every pair: (-1)^(number of pairs of 1s); a ring got stuck

    def params(self):
        return self.th.numel()

    def forward(self, S):
        n = self.n
        psi = torch.zeros(2 ** n, dtype=torch.float64)
        psi[0] = 1.0
        for layer in range(self.L + 1):
            for q in range(n):
                c, s = torch.cos(self.th[layer, q] / 2), torch.sin(self.th[layer, q] / 2)
                R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
                psi = torch.einsum("ij,ajb->aib", R, psi.reshape(2 ** q, 2, -1)).reshape(-1)
            if layer < self.L:
                psi = psi * self.cz
        return torch.log(psi ** 2 + 1e-300)


def fit(model, S, counts_tr, counts_va, lr, steps, patience=200, ckpt: Path | None = None, budget: float | None = None):
    """Adam on training NLL, early stopping on validation NLL. With ckpt/budget: stops after `budget` seconds, saves
    its state and returns None; the next call resumes."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad, step = math.inf, None, 0, 0
    if ckpt is not None and ckpt.exists():
        c = torch.load(ckpt)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        best, best_state, bad, step = c["best"], c["best_state"], c["bad"], c["step"]
    ntr, nva = counts_tr.sum(), counts_va.sum()
    t0 = time.perf_counter()
    while step < steps and bad <= patience:
        if budget is not None and time.perf_counter() - t0 > budget:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best, "best_state": best_state,
                        "bad": bad, "step": step}, ckpt)
            return None
        lp = model(S)
        loss = -(counts_tr * lp).sum() / ntr
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        with torch.no_grad():
            va = float(-(counts_va * lp).sum() / nva)
        if va < best - 1e-6:
            best, best_state, bad = va, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
    model.load_state_dict(best_state)
    with torch.no_grad():
        return model(S).double().numpy()


# ------------------------------------------------------------------------------------- evaluation
def auroc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks over ties
        m = s == v
        r[m] = r[m].mean()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))) if 0 < n1 < len(y) else float("nan")


def evaluate(logp, names, dx, Xte):
    n = len(names)
    ll = float(logp[index(Xte)].mean())
    ev = [i for i, k in enumerate(names) if k not in dx]
    di = [i for i, k in enumerate(names) if k in dx]
    p = np.exp(logp - logp.max()).reshape((2,) * n).transpose(ev + di).reshape(2 ** len(ev), 2 ** len(di))
    p = p / p.sum(1, keepdims=True)
    Sd = states(len(di))
    cond = p @ Sd  # [evidence pattern, diagnosis] -> P(dx = 1 | evidence)
    pe = cond[index(Xte[:, ev])]
    return ll, float(np.nanmean([auroc(Xte[:, di[k]], pe[:, k]) for k in range(len(di))]))


def logistic_auroc(Xtr, Xva, Xte, names, dx):
    ev = [i for i, k in enumerate(names) if k not in dx]
    out = []
    for k in [i for i, nm in enumerate(names) if nm in dx]:
        w = torch.zeros(len(ev) + 1, requires_grad=True)
        A = torch.as_tensor(np.c_[Xtr[:, ev], np.ones(len(Xtr))], dtype=torch.float32)
        y = torch.as_tensor(Xtr[:, k], dtype=torch.float32)
        opt = torch.optim.LBFGS([w], max_iter=200)

        def closure():
            opt.zero_grad()
            l = torch.nn.functional.binary_cross_entropy_with_logits(A @ w, y) + 1e-4 * (w[:-1] ** 2).sum()
            l.backward()
            return l
        opt.step(closure)
        s = (np.c_[Xte[:, ev], np.ones(len(Xte))] @ w.detach().numpy())
        out.append(auroc(Xte[:, k], s))
    return float(np.mean(out))


def run(size: int, seed: int, part: str):
    names, dx, Xall, Xte = data()
    rng = np.random.default_rng(seed)
    X = Xall[rng.choice(len(Xall), size, replace=False)]
    nva = max(int(0.2 * size), 100)
    Xva, Xtr = X[:nva], X[nva:]
    n = len(names)
    S = states(n)
    St = torch.as_tensor(S, dtype=torch.float32)
    ctr = torch.as_tensor(np.bincount(index(Xtr), minlength=2 ** n), dtype=torch.float32)
    cva = torch.as_tensor(np.bincount(index(Xva), minlength=2 ** n), dtype=torch.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    torch.manual_seed(seed)
    if part == "born":  # resumable: run again until it prints "done"
        ck = OUT / f"ckpt_{size}_{seed}.pt"
        m = Born(n, 9, Xtr.mean(0), S)
        t0 = time.perf_counter()
        lp = fit(m, St.double(), ctr.double(), cva.double(), 0.003, 4000, patience=400, ckpt=ck, budget=140)  # lr 0.01 was unstable
        if lp is None:
            print(f"size {size} seed {seed} born: paused")
            return
        rows.append(("Born machine (9 layers)", m.params(), lp, time.perf_counter() - t0))
    else:
        t0 = time.perf_counter()
        bn = learn(names, Xtr, 4, min_gain=1e-3)
        rows.append(("network (max 4 parents)", sum(2 ** len(pa) for pa in bn.struct.parents), np.log(bn.joint() + 1e-300),
                     time.perf_counter() - t0))
        for name, mk in (("autoregressive (FVSBN)", lambda: FVSBN(n)), ("RBM (8 hidden)", lambda: RBM(n, 8))):
            t0 = time.perf_counter()
            m = mk()
            rows.append((name, m.params(), fit(m, St, ctr, cva, 0.05, 3000), time.perf_counter() - t0))
    with open(OUT / "rows.jsonl", "a") as f:
        for name, k, lp, secs in rows:
            ll, auc = evaluate(lp, names, dx, Xte)
            f.write(json.dumps({"size": size, "seed": seed, "model": name, "params": int(k), "heldout_ll": ll,
                                "macro_auroc": auc, "seconds": secs}) + "\n")
        if part != "born":
            f.write(json.dumps({"size": size, "seed": seed, "model": "logistic regression (AUROC only)",
                                "params": (N_EVID + 1) * len(dx), "heldout_ll": None,
                                "macro_auroc": logistic_auroc(Xtr, Xva, Xte, names, dx), "seconds": 0}) + "\n")
    print(f"size {size} seed {seed} {part}: done")


def summary():
    d = pd.DataFrame([json.loads(x) for x in (OUT / "rows.jsonl").read_text().splitlines()])
    g = d.groupby(["size", "model"]).agg(seeds=("seed", "nunique"), params=("params", "median"), ll=("heldout_ll", "mean"),
                                         ll_sd=("heldout_ll", "std"), auroc=("macro_auroc", "mean"), auroc_sd=("macro_auroc", "std"),
                                         secs=("seconds", "median"))
    print(g.round(4).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary()
    else:
        run(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
