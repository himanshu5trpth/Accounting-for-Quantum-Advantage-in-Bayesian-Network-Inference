"""Binary Bayesian networks over clinical propositions, graded (virtual) evidence, and classical engines.

All propositions are binary. Graded findings (membership m in (0,1)) enter as Pearl virtual evidence:
an extra child node v with P(v=1 | f=1) = m and P(v=1 | f=0) = 1 - m, observed as v = 1.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Struct:
    names: list[str]
    parents: list[tuple[int, ...]]

    def __post_init__(self):
        for j, pa in enumerate(self.parents):
            assert all(p < j for p in pa), "parents must precede children"
        self._bits = None
        self._flat = None

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def S(self) -> int:
        return 2 ** self.n

    @property
    def bits(self) -> np.ndarray:
        """uint8 [n, S]: value of node j in joint state s (node 0 is the most significant bit)."""
        if self._bits is None:
            s = np.arange(self.S, dtype=np.int64)
            self._bits = np.stack([((s >> (self.n - 1 - j)) & 1).astype(np.uint8) for j in range(self.n)])
        return self._bits

    @property
    def flat(self) -> list[np.ndarray]:
        if self._flat is None:
            b = self.bits
            out = []
            for j, pa in enumerate(self.parents):
                idx = np.zeros(self.S, dtype=np.int64)
                for p in pa:
                    idx = idx * 2 + b[p]
                out.append(idx * 2 + b[j])
            self._flat = out
        return self._flat

    def n_cfg(self, j: int) -> int:
        return 2 ** len(self.parents[j])

    def pa_index(self, x: np.ndarray, j: int) -> np.ndarray:
        idx = np.zeros(len(x), dtype=np.int64)
        for p in self.parents[j]:
            idx = idx * 2 + x[:, p]
        return idx

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass
class BN:
    struct: Struct
    cpts: list[np.ndarray]  # cpts[j]: [2**|pa|, 2]

    def joint(self) -> np.ndarray:
        if getattr(self, "_joint", None) is None:
            p = np.ones(self.struct.S)
            for j, c in enumerate(self.cpts):
                p *= c.reshape(-1)[self.struct.flat[j]]
            self._joint = p
        return self._joint

    def sample(self, u: np.ndarray, clamp: dict[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Ancestral sampling from uniforms u [n_samples, n]. Clamped nodes are set (likelihood weighting);
        returns (states, log-weights)."""
        x = np.zeros(u.shape, dtype=np.int64)
        logw = np.zeros(len(u))
        for j, c in enumerate(self.cpts):
            p1 = c[self.struct.pa_index(x, j), 1]
            if clamp and j in clamp:
                x[:, j] = clamp[j]
                logw += np.log(np.where(clamp[j] == 1, p1, 1 - p1) + 1e-300)
            else:
                x[:, j] = (u[:, j] < p1).astype(np.int64)
        return x, logw

    def to_dict(self) -> dict:
        return {"names": self.struct.names, "parents": [list(p) for p in self.struct.parents],
                "cpts": [c.tolist() for c in self.cpts]}

    @classmethod
    def from_dict(cls, d: dict) -> "BN":
        return cls(Struct(d["names"], [tuple(p) for p in d["parents"]]), [np.array(c, float) for c in d["cpts"]])


# ------------------------------------------------------------------------------------- learning
def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def cond_mi(y: np.ndarray, j: int, c: int, given: list[int]) -> float:
    cols = [j, c, *given]
    rows = y[(y[:, cols] >= 0).all(1)][:, cols].astype(np.int64)
    if len(rows) < 50:
        return 0.0

    def H(sub):
        if not sub:
            return 0.0
        idx = np.zeros(len(rows), dtype=np.int64)
        for s in sub:
            idx = idx * 2 + rows[:, s]
        return _entropy(np.bincount(idx))

    g = list(range(2, len(cols)))
    return H([0] + g) + H([1] + g) - H([0, 1] + g) - H(g)


def learn(names: list[str], y: np.ndarray, max_parents: int, allowed: dict[int, list[int]] | None = None,
          min_gain: float = 1e-3, alpha: float = 1.0) -> BN:
    """Greedy conditional-MI parent selection (earlier nodes only, optionally restricted by `allowed`)
    and Laplace-smoothed CPTs from rows where the node and its parents are observed (y = -1 missing)."""
    parents = []
    for j in range(len(names)):
        cand = allowed.get(j, list(range(j))) if allowed is not None else list(range(j))
        chosen: list[int] = []
        while len(chosen) < min(max_parents, len(cand)):
            gains = {c: cond_mi(y, j, c, chosen) for c in cand if c not in chosen}
            if not gains:
                break
            best = max(gains, key=gains.get)
            if gains[best] < min_gain:
                break
            chosen.append(best)
        parents.append(tuple(sorted(chosen)))
    st = Struct(names, parents)
    cpts = []
    for j, pa in enumerate(parents):
        ok = (y[:, [j, *pa]] >= 0).all(1)
        idx = st.pa_index(y[ok].astype(np.int64), j)
        counts = np.full((2 ** len(pa), 2), alpha)
        np.add.at(counts, (idx, y[ok, j].astype(np.int64)), 1)
        cpts.append(counts / counts.sum(1, keepdims=True))
    return BN(st, cpts)


# ------------------------------------------------------------------------------------- evidence
@dataclass
class Evidence:
    hard: dict[str, int]          # crisp findings (membership 0 or 1)
    soft: dict[str, float]        # graded findings, membership in (0, 1)

    @classmethod
    def from_memberships(cls, m: dict[str, float], crisp: bool = False) -> "Evidence":
        hard, soft = {}, {}
        for k, v in m.items():
            if v is None:
                continue
            if crisp or v <= 0.0 or v >= 1.0:
                hard[k] = int(v >= 0.5)
            else:
                soft[k] = float(v)
        return cls(hard, soft)


def ancestral(bn: BN, keep: list[str]) -> BN:
    """The sub-network of `keep` and their ancestors. Every other node is barren (sums to one), so any P(targets, E)
    with targets and evidence in `keep` is unchanged; circuits shrink to the qubits that matter."""
    st = bn.struct
    rel, stack = set(), [st.index(k) for k in keep]
    while stack:
        j = stack.pop()
        if j not in rel:
            rel.add(j)
            stack.extend(st.parents[j])
    idx = sorted(rel)
    pos = {j: i for i, j in enumerate(idx)}
    return BN(Struct([st.names[j] for j in idx], [tuple(pos[p] for p in st.parents[j]) for j in idx]), [bn.cpts[j] for j in idx])


def with_virtual(bn: BN, ev: Evidence) -> tuple[BN, dict[int, int]]:
    """Append one virtual-evidence child per graded finding. Returns (extended BN, observation {node: value})."""
    names, parents, cpts = list(bn.struct.names), list(bn.struct.parents), list(bn.cpts)
    obs = {bn.struct.index(k): v for k, v in ev.hard.items()}
    for k, m in ev.soft.items():
        f = bn.struct.index(k)
        names.append(f"v:{k}")
        parents.append((f,))
        cpts.append(np.array([[m, 1 - m], [1 - m, m]]))  # rows: f=0 -> P(v=1)=1-m ; f=1 -> P(v=1)=m
        obs[len(names) - 1] = 1
    return BN(Struct(names, parents), cpts), obs


def evidence_weights(bn: BN, ev: Evidence) -> np.ndarray:
    """Unnormalised P(x, E) over base states: joint x hard-evidence masks x virtual-evidence likelihoods.
    Identical to marginalising the virtual nodes of `with_virtual(bn, ev)` and conditioning on v = 1."""
    st = bn.struct
    w = bn.joint().copy()
    for k, v in ev.hard.items():
        w *= st.bits[st.index(k)] == v
    for k, m in ev.soft.items():
        b = st.bits[st.index(k)]
        w *= np.where(b == 1, m, 1 - m)
    return w


# ------------------------------------------------------------------------------------- engines
def _cost(t0, **kw) -> dict:
    return {"seconds": time.perf_counter() - t0, **kw}


def posterior_enum(bn: BN, ev: Evidence, targets: list[str]) -> tuple[dict[str, float], float, dict]:
    """Full joint enumeration (2^n states): the reference implementation, kept as a test oracle for small n."""
    t0 = time.perf_counter()
    w = evidence_weights(bn, ev)
    pe = float(w.sum())
    st = bn.struct
    out = {t: float(w @ st.bits[st.index(t)] / pe) if pe > 0 else float("nan") for t in targets}
    return out, pe, _cost(t0, model_evaluations=st.S, method="exact enumeration")


# ------------------------------------------------------------------------------------- variable elimination
_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _einsum(factors: list[tuple[tuple[int, ...], np.ndarray]], out_vars: tuple[int, ...]) -> np.ndarray:
    """Product of factors, summed onto out_vars (local letter map: only the clique's variables need letters)."""
    used = sorted({v for vs, _ in factors for v in vs} | set(out_vars))
    if len(used) > len(_LETTERS):
        raise ValueError(f"elimination clique of {len(used)} variables exceeds einsum limit")
    L = {v: _LETTERS[i] for i, v in enumerate(used)}
    spec = ",".join("".join(L[v] for v in vs) for vs, _ in factors) + "->" + "".join(L[v] for v in out_vars)
    return np.einsum(spec, *[t for _, t in factors], optimize=True)


def marginal(bn: BN, ev: Evidence, keep: list[int]) -> np.ndarray:
    """Unnormalised P(keep, E) as an array with one axis (size 2) per kept node, by variable elimination:
    CPT factors with hard evidence sliced in, graded evidence as likelihood factors [1-m, m], barren nodes pruned,
    greedy min-weight elimination order. Exact for any number of nodes; cost follows the induced tree-width."""
    st = bn.struct
    hard = {st.index(k): v for k, v in ev.hard.items()}
    soft = {st.index(k): m for k, m in ev.soft.items()}
    needed = set(keep) | set(hard) | set(soft)
    relevant = set()
    stack = list(needed)
    while stack:  # ancestors of needed nodes; everything else sums to one and is dropped
        j = stack.pop()
        if j not in relevant:
            relevant.add(j)
            stack.extend(st.parents[j])
    factors = []
    for j in sorted(relevant):
        vs = tuple(st.parents[j]) + (j,)
        t = bn.cpts[j].reshape((2,) * len(vs))
        idx = tuple(hard[v] if v in hard else slice(None) for v in vs)
        t = t[idx]
        vs = tuple(v for v in vs if v not in hard)
        factors.append((vs, t))
    for j, m in soft.items():
        if j not in hard:
            factors.append(((j,), np.array([1.0 - m, m])))
    eliminate = {v for vs, _ in factors for v in vs} - set(keep)
    while eliminate:
        def weight(v):
            return len({u for vs, _ in factors if v in vs for u in vs})
        v = min(eliminate, key=weight)
        inv = [f for f in factors if v in f[0]]
        rest = [f for f in factors if v not in f[0]]
        scope = tuple(sorted({u for vs, _ in inv for u in vs} - {v}))
        factors = rest + [(scope, _einsum(inv, scope))]
        eliminate.discard(v)
    keep_t = tuple(keep)
    const = [f for f in factors if not f[0]]
    others = [f for f in factors if f[0]]
    out = _einsum(others, keep_t) if others else np.ones((2,) * len(keep_t))
    for _, t in const:
        out = out * float(t)
    return out


def posterior_exact(bn: BN, ev: Evidence, targets: list[str]) -> tuple[dict[str, float], float, dict]:
    """-> ({target: P(target=1 | E)}, P(E), cost). Exact (variable elimination); P(E) includes graded likelihoods."""
    t0 = time.perf_counter()
    st = bn.struct
    fixed = {t: float(ev.hard[t]) for t in targets if t in ev.hard}
    free = [t for t in targets if t not in ev.hard]
    keep = [st.index(t) for t in free]
    tab = marginal(bn, ev, keep)
    pe = float(tab.sum())
    out = dict(fixed)
    for i, t in enumerate(free):
        axes = tuple(a for a in range(len(keep)) if a != i)
        m = tab.sum(axis=axes) if axes else tab
        out[t] = float(m[1] / pe) if pe > 0 else float("nan")
    return {t: out[t] for t in targets}, pe, _cost(t0, method="exact inference (variable elimination)")


def _chunks(n: int, size: int = 200_000):
    while n > 0:
        yield min(n, size)
        n -= size


def posterior_rejection(bn: BN, ev: Evidence, targets: list[str], n: int, seed: int):
    t0 = time.perf_counter()
    ext, obs = with_virtual(bn, ev)
    rng = np.random.default_rng(seed)
    idx = [ext.struct.index(t) for t in targets]
    k, hits = 0, np.zeros(len(targets))
    for m in _chunks(n):
        x, _ = ext.sample(rng.random((m, ext.struct.n)))
        acc = np.ones(m, bool)
        for j, v in obs.items():
            acc &= x[:, j] == v
        k += int(acc.sum())
        hits += x[acc][:, idx].sum(0)
    out = {t: float(hits[i] / k) if k else float("nan") for i, t in enumerate(targets)}
    return out, k / n, _cost(t0, samples=n, accepted=k, oracle_calls=n, method="rejection sampling")


def posterior_lw(bn: BN, ev: Evidence, targets: list[str], n: int, seed: int):
    t0 = time.perf_counter()
    ext, obs = with_virtual(bn, ev)
    rng = np.random.default_rng(seed)
    idx = [ext.struct.index(t) for t in targets]
    logws, num = [], []
    for m in _chunks(n):
        x, logw = ext.sample(rng.random((m, ext.struct.n)), clamp=obs)
        logws.append(logw)
        num.append(x[:, idx])
    logw = np.concatenate(logws)
    X = np.concatenate(num)
    w = np.exp(logw - logw.max())
    out = {t: float((w * X[:, i]).sum() / w.sum()) for i, t in enumerate(targets)}
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return out, float(np.exp(logw).mean()), _cost(t0, samples=n, effective_sample_size=ess, oracle_calls=n,
                                                  method="likelihood weighting")


def entropy(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def info_gain(bn: BN, ev: Evidence, candidates: list[str], targets: list[str]) -> dict[str, float]:
    """Expected reduction of summed diagnosis entropy if each unknown (or graded) finding were observed (exact)."""
    st = bn.struct
    tidx = [st.index(t) for t in targets]

    def h(tab):  # tab over targets (axes in order); summed marginal entropies
        z = tab.sum()
        if z <= 0:
            return 0.0
        tot = 0.0
        for i in range(len(tidx)):
            axes = tuple(a for a in range(len(tidx)) if a != i)
            tot += entropy(float((tab.sum(axis=axes) if axes else tab)[1] / z))
        return tot

    base = marginal(bn, ev, tidx)
    h0, z0, out = h(base), float(base.sum()), {}
    for c in candidates:
        tab = marginal(bn, ev, tidx + [st.index(c)])
        t0_, t1_ = tab[..., 0], tab[..., 1]
        pc = float(t1_.sum() / z0) if z0 > 0 else 0.0
        out[c] = h0 - (pc * h(t1_) + (1 - pc) * h(t0_))
    return out


def info_gain_enum(bn: BN, ev: Evidence, candidates: list[str], targets: list[str]) -> dict[str, float]:
    """Full-enumeration reference for info_gain (test oracle, small n)."""
    st = bn.struct
    w = evidence_weights(bn, ev)
    T = np.stack([st.bits[st.index(t)] for t in targets]).astype(float)

    def h(wv):
        z = wv.sum()
        return sum(entropy(p) for p in (T @ wv) / z) if z > 0 else 0.0

    h0, z0, out = h(w), w.sum(), {}
    for c in candidates:
        b = st.bits[st.index(c)]
        w1 = w * b
        pc = w1.sum() / z0
        out[c] = h0 - (pc * h(w1) + (1 - pc) * h(w - w1))
    return out


def dx_entropy(bn: BN, ev: Evidence, targets: list[str]) -> float:
    """Summed binary entropy of P(target | E) over the targets (the quantity info_gain expects to reduce)."""
    p, _, _ = posterior_exact(bn, ev, targets)
    return float(sum(entropy(v) for v in p.values()))


def drivers(bn: BN, ev: Evidence, target: str, k: int = 3, min_effect: float = 0.005) -> list[tuple[str, float]]:
    """Findings whose removal lowers P(target | E) the most (leave-one-out, exact)."""
    base, _, _ = posterior_exact(bn, ev, [target])
    eff = []
    for f in list(ev.hard) + list(ev.soft):
        e2 = Evidence({a: b for a, b in ev.hard.items() if a != f}, {a: b for a, b in ev.soft.items() if a != f})
        p, _, _ = posterior_exact(bn, e2, [target])
        eff.append((f, base[target] - p[target]))
    return sorted([e for e in eff if e[1] >= min_effect], key=lambda e: -e[1])[:k]
