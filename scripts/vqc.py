"""Step 9 (optional): a discriminative quantum classifier, to close the reviewer question. 40 v2.1 findings in, 6 diagnosis
probabilities out, trained on the labels. Data re-uploading circuit on 10 qubits (40 binary inputs would need 40 qubits):
4 blocks per pass, 2 passes; in a block, qubit q gets RY(w * x + c) for one finding (10 findings per block), then a
trainable RY on every qubit and CZ on all pairs. P(qubit d = 1) for d = 0..5 is the probability of diagnosis d. Exact statevector simulation (real amplitudes).

Frozen v2.1 split: trained on the 350,779 inner-train admissions, early stopping on the 87,029 validation admissions
(the frozen inner/validation split; born.py drew its validation rows from its own subsample), scored once on the
108,220 test admissions against network 0.798 and logistic regression 0.827 (macro AUROC).

  python scripts/vqc.py train     # resumable: repeat until it prints "done"
  python scripts/vqc.py test
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

KB, VOC, OUT = Path("artifacts/v2.1/kb"), Path("configs/vocab_v2.1.json"), Path("artifacts/v2.1/results")
CK = Path("artifacts/vqc_ckpt.pt")
SEED, NQ, PASSES, NDX, BATCH, MAX_EPOCHS, PATIENCE = 2026, 10, 2, 6, 256, 40, 4  # batch 256: autograd keeps every state (2048 needs > 3.5 GB)
BLOCKS = 4 * PASSES                 # 40 findings = 4 blocks of 10; re-uploaded PASSES times
EPOCH_ROWS, VAL_ROWS, LR = 40000, 20000, 0.05  # an "epoch" = 40,000 sampled inner-train rows; early stopping on 20,000 validation rows
torch.set_num_threads(2)


def data():
    P = json.loads(VOC.read_text())["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    ev = [k for k, p in P.items() if p["kind"] != "diagnosis"]
    assert len(ev) == NQ * 4
    df = pd.read_parquet(KB / "cohort.parquet")
    subs = np.unique(df.subject_id.to_numpy())
    rng = np.random.default_rng(SEED)
    train = set(rng.permutation(subs)[: int(round(0.8 * len(subs)))].tolist())  # kb.split_mask
    tr = df.subject_id.isin(train).to_numpy()
    sub_tr = np.unique(df.subject_id.to_numpy()[tr])  # vocab.inner_split: 80% of training patients, seed + 1
    inner_set = set(np.random.default_rng(SEED + 1).permutation(sub_tr)[: int(round(0.8 * len(sub_tr)))].tolist())
    inner = tr & df.subject_id.isin(inner_set).to_numpy()
    val = tr & ~inner
    X = df[ev].to_numpy(np.float32)
    X[X < 0] = 0.5  # unmeasured lab: halfway angle
    Y = df[dx].to_numpy(np.float32)
    return X, Y, inner, val, ~tr, dx


class Circuit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.w = torch.nn.Parameter(math.pi * (1 + 0.1 * torch.randn(BLOCKS, NQ, generator=g)))
        self.c = torch.nn.Parameter(0.1 * torch.randn(BLOCKS, NQ, generator=g))
        self.v = torch.nn.Parameter(0.1 * torch.randn(BLOCKS, NQ, generator=g))
        s = torch.arange(2 ** NQ)
        bits = torch.stack([(s >> (NQ - 1 - q)) & 1 for q in range(NQ)], 1).float()  # qubit 0 = most significant
        k = bits.sum(1)
        self.register_buffer("cz", 1 - 2 * ((k * (k - 1) / 2) % 2))
        self.register_buffer("readout", bits[:, :NDX])

    def params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def ry(psi, q, theta):
        """RY(theta) on qubit q for a batch; theta is a scalar or [batch]."""
        B = psi.shape[0]
        x = psi.reshape(B, 2 ** q, 2, -1)
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        if c.dim() == 1:
            c, s = c[:, None, None], s[:, None, None]
        a0, a1 = x[:, :, 0, :], x[:, :, 1, :]
        return torch.stack([c * a0 - s * a1, s * a0 + c * a1], 2).reshape(B, -1)

    def forward(self, X):
        B = X.shape[0]
        psi = torch.zeros(B, 2 ** NQ)
        psi[:, 0] = 1.0
        for b in range(BLOCKS):
            for q in range(NQ):
                psi = self.ry(psi, q, self.w[b, q] * X[:, (b % 4) * NQ + q] + self.c[b, q])
            for q in range(NQ):
                psi = self.ry(psi, q, self.v[b, q])
            psi = psi * self.cz
        return (psi ** 2) @ self.readout  # [batch, 6] = P(qubit d = 1)


def bce(p, y):
    p = p.clamp(1e-6, 1 - 1e-6)
    return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()


def evaluate(m, X, Y):
    with torch.no_grad():
        P = torch.cat([m(torch.as_tensor(X[i:i + 8192])) for i in range(0, len(X), 8192)])
    return P.numpy(), float(bce(P, torch.as_tensor(Y)))


def train(budget: float = 140):
    t0 = time.perf_counter()
    X, Y, inner, val, test, dx = data()
    m = Circuit()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    st = {"epoch": 0, "best": math.inf, "bad": 0, "history": []}
    if CK.exists():
        c = torch.load(CK)
        m.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        st = c["state"]
        best_state = c["best_state"]
    else:
        best_state = None
    Xi, Yi = torch.as_tensor(X[inner]), torch.as_tensor(Y[inner])
    vi = np.flatnonzero(val)
    vi = vi[np.random.default_rng(SEED).choice(len(vi), min(VAL_ROWS, len(vi)), replace=False)]
    while st["epoch"] < MAX_EPOCHS and st["bad"] < PATIENCE:
        if time.perf_counter() - t0 > budget:
            break
        rng = np.random.default_rng(SEED + 1000 + st["epoch"])
        perm = torch.as_tensor(rng.choice(len(Xi), min(EPOCH_ROWS, len(Xi)), replace=False))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = bce(m(Xi[idx]), Yi[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        _, vl = evaluate(m, X[vi], Y[vi])
        st["epoch"] += 1
        st["history"].append(vl)
        if vl < st["best"] - 1e-5:
            st["best"], st["bad"] = vl, 0
            best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            st["bad"] += 1
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "state": st, "best_state": best_state}, CK)
        print(f"epoch {st['epoch']}: validation BCE {vl:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    finished = st["epoch"] >= MAX_EPOCHS or st["bad"] >= PATIENCE
    print("done" if finished else "paused")


def auroc(y, s):
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = (sums / cnt)[inv]  # average ranks over ties
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def test():
    X, Y, inner, val, te, dx = data()
    c = torch.load(CK)
    m = Circuit()
    m.load_state_dict(c["best_state"])
    P, tl = evaluate(m, X[te], Y[te])
    per = {d: auroc(Y[te][:, i].astype(int), P[:, i]) for i, d in enumerate(dx)}
    ek = json.loads((OUT / "eval-kb.json").read_text())["models"]
    out = {"model": f"data re-uploading circuit, {NQ} qubits, {BLOCKS} blocks ({PASSES} passes over the 40 findings), CZ on all pairs",
           "params": m.params(), "epoch_rows": EPOCH_ROWS, "early_stopping_rows": VAL_ROWS,
           "train_admissions": int(inner.sum()), "validation_admissions": int(val.sum()), "test_admissions": int(te.sum()),
           "epochs": c["state"]["epoch"], "validation_bce_history": c["state"]["history"], "test_bce": tl,
           "macro_auroc": float(np.mean(list(per.values()))), "auroc_by_diagnosis": per,
           "network_macro_auroc": ek["learned network"]["macro_auroc"]["value"],
           "logistic_macro_auroc": ek["logistic regression"]["macro_auroc"]["value"]}
    (OUT / "vqc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"train": train, "test": test}[sys.argv[1]]()
