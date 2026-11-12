"""Multi-state Bayesian networks for the quantum paper (tasks/plan_quantum_paper.md).

Representation, exact inference with pruning (min-fill variable elimination), likelihood-weighting quantities, and the
qubit-register circuit (W1): node X with k states gets ceil(log2 k) qubits (at least 1), most significant qubit first;
states k .. 2^m - 1 get zero amplitude by construction (every branch into them has probability 0).

  python scripts/msbn.py validate     # W1 gates 1-3 + binary equivalence + clamped (V2) + Grover step -> multistate_validation.json
  python scripts/msbn.py variants     # W1.3: V0-V3 CX on multi-state benchmark instances -> multistate_variants.json
"""
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED, Z90, REL = 2026, 1.645, 0.1
OUT = Path("artifacts/quantum_paper")


# ------------------------------------------------------------------------------------- representation
@dataclass
class MSBN:
    names: list
    card: list
    parents: list   # tuples of node indices, all earlier (index order is topological)
    cpts: list      # cpts[j].shape == (*[card[p] for p in parents[j]], card[j]); last axis sums to 1

    @property
    def n(self) -> int:
        return len(self.names)


def from_pgmpy(model) -> MSBN:
    import networkx as nx
    order = list(nx.lexicographical_topological_sort(model))  # deterministic
    idx = {v: i for i, v in enumerate(order)}
    card, parents, cpts = [], [], []
    for v in order:
        cpd = model.get_cpds(v)
        t = np.moveaxis(np.asarray(cpd.values, float), 0, -1)
        card.append(int(cpd.variable_card))
        parents.append(tuple(idx[e] for e in cpd.variables[1:]))
        cpts.append(t / t.sum(-1, keepdims=True))
    return MSBN(order, card, parents, cpts)


def from_binary(b) -> MSBN:
    """qddx.bn.BN (row index = parent bits, first parent most significant) -> MSBN."""
    st = b.struct
    cpts = [c.reshape((2,) * len(pa) + (2,)) for c, pa in zip(b.cpts, st.parents)]
    return MSBN(list(st.names), [2] * st.n, [tuple(p) for p in st.parents], cpts)


def random_msbn(n: int, card_max: int, k_max: int, rng, zero_frac: float = 0.0) -> MSBN:
    card = [int(rng.integers(2, card_max + 1)) for _ in range(n)]
    parents, cpts = [], []
    for j in range(n):
        pa = tuple(sorted(rng.choice(j, size=int(rng.integers(0, min(j, k_max) + 1)), replace=False).tolist())) if j else ()
        t = rng.dirichlet(np.ones(card[j]), size=int(np.prod([card[p] for p in pa], dtype=int)))
        if zero_frac:
            t[rng.random(t.shape) < zero_frac] = 0
            t[t.sum(1) == 0, 0] = 1
            t /= t.sum(1, keepdims=True)
        parents.append(pa)
        cpts.append(t.reshape(tuple(card[p] for p in pa) + (card[j],)))
    return MSBN([f"x{j}" for j in range(n)], card, parents, cpts)


def forward_sample(bn: MSBN, rng) -> list:
    x = []
    for j in range(bn.n):
        pr = bn.cpts[j][tuple(x[p] for p in bn.parents[j])]
        x.append(int(rng.choice(bn.card[j], p=pr)))
    return x


# ------------------------------------------------------------------------------------- pruning
def ancestors(bn: MSBN, nodes) -> set:
    out, stack = set(), list(nodes)
    while stack:
        j = stack.pop()
        if j not in out:
            out.add(j)
            stack.extend(bn.parents[j])
    return out


def prune(bn: MSBN, target: int, evidence: dict) -> tuple[dict, set]:
    """-> (relevant evidence, kept nodes). Evidence e is dropped if target and e are separated in the moral graph of
    An({target} U E) once E minus e is removed (d-separation, Lauritzen); d-separation satisfies intersection, so
    the dropped set can be removed jointly. Then barren nodes go: kept = An({target} U relevant evidence)."""
    anc = ancestors(bn, {target, *evidence})
    adj = {j: set() for j in anc}
    for j in anc:
        pa = [p for p in bn.parents[j]]
        for p in pa:
            adj[j].add(p)
            adj[p].add(j)
        for a in pa:
            for c in pa:
                if a != c:
                    adj[a].add(c)
    rel = {}
    for e, v in evidence.items():
        block = set(evidence) - {e}
        seen, stack = {target}, [target]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w not in seen and w not in block:
                    seen.add(w)
                    stack.append(w)
        if e in seen:
            rel[e] = v
    return rel, ancestors(bn, {target, *rel})


# ------------------------------------------------------------------------------------- exact inference
_LET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _mul(factors, out_vars):
    vs = sorted({v for f in factors for v in f[0]})
    if len(vs) > len(_LET):
        raise ValueError("factor scope too large for einsum")
    L = {v: _LET[i] for i, v in enumerate(vs)}
    spec = ",".join("".join(L[v] for v in f[0]) for f in factors) + "->" + "".join(L[v] for v in out_vars)
    return tuple(out_vars), np.einsum(spec, *[f[1] for f in factors], optimize=True)


def ve(bn: MSBN, nodes, evidence: dict, query: list, squared: bool = False, max_log10: float = 9.0):
    """Unnormalised P(query, evidence) over the given nodes (which must be closed under parents), by min-fill VE.
    squared: each evidence node's CPT entry squared (LW's second moment). -> (table, log10 largest factor, width)."""
    factors = []
    for j in sorted(nodes):
        vs = tuple(bn.parents[j]) + (j,)
        t = bn.cpts[j]
        idx = tuple(evidence[v] if v in evidence else slice(None) for v in vs)
        t = t[idx]
        if squared and j in evidence:
            t = t ** 2
        factors.append((tuple(v for v in vs if v not in evidence), t))
    hidden = {v for f in factors for v in f[0]} - set(query)
    big, width = 0.0, 0
    lc = {v: math.log10(bn.card[v]) for v in range(bn.n)}
    while hidden:
        nb = {v: set() for v in hidden}
        for vs, _ in factors:
            for v in vs:
                if v in nb:
                    nb[v].update(u for u in vs if u != v)
        edges = {frozenset((a, c)) for vs, _ in factors for a in vs for c in vs if a != c}

        def score(v):
            ns = list(nb[v])
            fill = sum(1 for i in range(len(ns)) for k in range(i + 1, len(ns)) if frozenset((ns[i], ns[k])) not in edges)
            return fill, lc[v] + sum(lc[u] for u in ns), v
        v = min(hidden, key=score)
        inv = [f for f in factors if v in f[0]]
        rest = [f for f in factors if v not in f[0]]
        scope = sorted({u for f in inv for u in f[0]})
        size = sum(lc[u] for u in scope)
        if size > max_log10:
            raise MemoryError(f"factor of 10^{size:.1f} entries")
        big, width = max(big, size), max(width, len(scope) - 1)
        factors = rest + [_mul(inv, [u for u in scope if u != v])]
        hidden.discard(v)
    tab = _mul(factors, list(query))[1] if factors else np.array(1.0)
    if query:
        big = max(big, sum(lc[u] for u in query))
    return tab, big, width


def elim_cost(bn: MSBN, nodes, evidence: dict, query: list, cap: float = 1e9) -> tuple[float, int]:
    """Symbolic version of ve(): the same min-fill order on scopes alone, no arithmetic.
    -> (log10 largest factor, width). Used for structural predictors (W7) without computing any probability."""
    scopes = []
    for j in sorted(nodes):
        scopes.append(tuple(v for v in tuple(bn.parents[j]) + (j,) if v not in evidence))
    hidden = {v for sc in scopes for v in sc} - set(query)
    lc = {v: math.log10(bn.card[v]) for v in range(bn.n)}
    big, width = 0.0, 0
    while hidden:
        nb = {v: set() for v in hidden}
        for sc in scopes:
            for v in sc:
                if v in nb:
                    nb[v].update(u for u in sc if u != v)
        edges = {frozenset((a, c)) for sc in scopes for a in sc for c in sc if a != c}

        def score(v):
            ns = list(nb[v])
            fill = sum(1 for i in range(len(ns)) for k in range(i + 1, len(ns)) if frozenset((ns[i], ns[k])) not in edges)
            return fill, lc[v] + sum(lc[u] for u in ns), v
        v = min(hidden, key=score)
        inv = [sc for sc in scopes if v in sc]
        scope = sorted({u for sc in inv for u in sc})
        size = sum(lc[u] for u in scope)
        big, width = max(big, size), max(width, len(scope) - 1)
        if size > cap:
            return math.inf, width
        scopes = [sc for sc in scopes if v not in sc] + [tuple(u for u in scope if u != v)]
        hidden.discard(v)
    if query:
        big = max(big, sum(lc[u] for u in query))
    return big, width


def node_max(bn: MSBN, j: int, v: int, evidence: dict) -> float:
    """max over j's unobserved parents of P(x_j = v | parents), observed parents fixed."""
    pa = bn.parents[j]
    idx = tuple(evidence[p] if p in evidence else slice(None) for p in pa) + (v,)
    return float(np.max(bn.cpts[j][idx]))


def lw_quantities(bn: MSBN, target: int, s: int, evidence: dict, pruned: bool = True, eps: float = REL, max_log10: float = 8.0) -> dict:
    """Exact LW and qLW quantities for P(target = s | evidence), on the pruned (default) or full network."""
    if pruned:
        ev, nodes = prune(bn, target, evidence)
    else:
        ev, nodes = dict(evidence), set(range(bn.n))
    t0 = time.perf_counter()
    tab, big, width = ve(bn, nodes, ev, [target], max_log10=max_log10)
    t_exact = time.perf_counter() - t0
    pe = float(tab.sum())
    p = float(tab[s] / pe)
    sq, _, _ = ve(bn, nodes, ev, [target], squared=True, max_log10=max_log10)
    a1 = float(sq[s])
    a0 = float(sq.sum()) - a1
    var = ((1 - p) ** 2 * a1 + p ** 2 * a0) / pe ** 2
    logW = sum(math.log(node_max(bn, e, v, ev)) for e, v in ev.items())
    aE = pe / math.exp(logW)
    return {"p": p, "p_evidence": pe, "lw_samples": (Z90 / eps) ** 2 * var / p ** 2 if p > 0 else math.inf,
            "D_T": a1 / (pe ** 2 * p) if p > 0 else math.nan, "dispersion": (a0 + a1) / pe ** 2, "logW": logW,
            "a_E": aE, "a_T": aE * p, "log10_max_factor": big, "width": width, "seconds_exact": t_exact,
            "nodes": len(nodes), "evidence_used": len(ev), "evidence_given": len(evidence)}


def qlw_cost(aE: float, aT: float, kappa: float, eps: float = REL) -> float:
    return kappa * math.pi / eps * (1 / math.sqrt(aE) + 1 / math.sqrt(aT))


# ------------------------------------------------------------------------------------- circuits
def nq(k: int) -> int:
    return max(1, (k - 1).bit_length())


def registers(bn: MSBN, nodes) -> dict:
    reg, q = {}, 0
    for j in sorted(nodes):
        reg[j] = list(range(q, q + nq(bn.card[j])))
        q += nq(bn.card[j])
    return reg


def _angles(bn: MSBN, j: int, ctrl_parents: list, ctrl_regs: list, fixed: dict, b: int, want=None):
    """UCRY angles for qubit b of node j's register (want=None), or for a weight qubit holding
    sqrt(P(x_j = want | parents)) (want = an observed value). Control i <-> bit i of the control index (qiskit);
    controls = the unobserved parents' registers (each most significant qubit first), then j's own qubits 0..b-1."""
    pa = bn.parents[j]
    m = nq(bn.card[j])
    nctl = sum(len(r) for r in ctrl_regs) + (b if want is None else 0)
    cs = np.arange(2 ** nctl)
    bits = (cs[:, None] >> np.arange(max(nctl, 1))) & 1
    pv, pos, valid = {}, 0, np.ones(len(cs), bool)
    for p, r in zip(ctrl_parents, ctrl_regs):
        v = np.zeros(len(cs), int)
        for t in range(len(r)):
            v = v * 2 + bits[:, pos + t]
        pos += len(r)
        valid &= v < bn.card[p]
        pv[p] = np.minimum(v, bn.card[p] - 1)
    idx = tuple(pv[p] if p in pv else np.full(len(cs), fixed[p]) for p in pa)
    probs = bn.cpts[j][idx] if pa else np.broadcast_to(bn.cpts[j], (len(cs), bn.card[j]))
    if want is not None:
        r1 = probs[:, want]
    else:
        prefix = np.zeros(len(cs), int)
        for t in range(b):
            prefix = prefix * 2 + bits[:, pos + t]
        s = np.arange(bn.card[j])
        inpre = (s[None, :] >> (m - b)) == prefix[:, None]
        mass = (probs * inpre).sum(1)
        mass1 = (probs * inpre * ((s >> (m - 1 - b)) & 1)[None, :]).sum(1)
        r1 = np.divide(mass1, mass, out=np.zeros_like(mass1), where=mass > 0)
    ang = 2 * np.arcsin(np.sqrt(np.clip(r1, 0, 1)))
    ang[~valid] = 0.0
    return ang.tolist()


def prep_circuit(bn: MSBN, nodes=None):
    """-> (circuit over the nodes' registers, registers). Nodes must be closed under parents."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UCRYGate
    nodes = sorted(range(bn.n) if nodes is None else nodes)
    reg = registers(bn, nodes)
    qc = QuantumCircuit(sum(len(r) for r in reg.values()), name="A")
    for j in nodes:
        cps = list(bn.parents[j])
        for b, q in enumerate(reg[j]):
            ang = _angles(bn, j, cps, [reg[p] for p in cps], {}, b)
            ctrl = [x for p in cps for x in reg[p]] + reg[j][:b]
            if ctrl:
                qc.append(UCRYGate(ang), [q] + ctrl)
            else:
                qc.ry(ang[0], q)
    return qc, reg


def literals(bn: MSBN, reg: dict, assign: dict) -> dict:
    out = {}
    for j, v in assign.items():
        m = len(reg[j])
        for t, q in enumerate(reg[j]):
            out[q] = (v >> (m - 1 - t)) & 1
    return out


def baseline_A(bn: MSBN, nodes, evidence: dict, target: int | None, s: int | None):
    """V0/V1 problem circuit: register preparation + one multi-controlled X onto the flag (last qubit)."""
    from qddx.quantum import conj_oracle, problem_circuit
    prep, reg = prep_circuit(bn, nodes)
    assign = {**evidence, **({target: s} if target is not None else {})}
    return problem_circuit(prep, conj_oracle(prep.num_qubits, literals(bn, reg, assign))), reg


def clamped_A(bn: MSBN, nodes, evidence: dict, target: int | None, s: int | None):
    """V2/V3: observed nodes are not qubits. An observed node with an unobserved parent gets one weight qubit
    rotated to sqrt(P(e | parents)); one with only observed parents contributes a constant. -> (A, C) with
    P(flag) = P(evidence [, target = s]) / C."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCXGate, UCRYGate
    nodes = sorted(nodes)
    free = [j for j in nodes if j not in evidence]
    weighted = [j for j in nodes if j in evidence and any(p not in evidence for p in bn.parents[j])]
    reg = registers(bn, free)
    nfree = sum(len(r) for r in reg.values())
    wq = {j: nfree + i for i, j in enumerate(weighted)}
    n_q = nfree + len(weighted) + 1
    qc = QuantumCircuit(n_q, name="A_clamped")
    C = 1.0
    for j in nodes:
        cps = [p for p in bn.parents[j] if p not in evidence]
        fixed = {p: evidence[p] for p in bn.parents[j] if p in evidence}
        if j in evidence and j not in wq:
            C *= float(bn.cpts[j][tuple(fixed[p] for p in bn.parents[j]) + (evidence[j],)])
            continue
        targets = [(0, wq[j])] if j in wq else list(enumerate(reg[j]))
        for b, q in targets:
            if j in wq:
                ang = _angles(bn, j, cps, [reg[p] for p in cps], fixed, 0, want=evidence[j])
                ctrl = [x for p in cps for x in reg[p]]
            else:
                ang = _angles(bn, j, cps, [reg[p] for p in cps], fixed, b)
                ctrl = [x for p in cps for x in reg[p]] + reg[j][:b]
            if ctrl:
                qc.append(UCRYGate(ang), [q] + ctrl)
            else:
                qc.ry(ang[0], q)
    lits = {wq[j]: 1 for j in weighted}
    if target is not None:
        lits.update(literals(bn, reg, {target: s}))
    if lits:
        ctrl = sorted(lits)
        qc.append(MCXGate(len(ctrl), ctrl_state=sum(lits[q] << i for i, q in enumerate(ctrl))), ctrl + [n_q - 1])
    else:
        qc.x(n_q - 1)
    return qc, C


# ------------------------------------------------------------------------------------- W1 validation
def _register_values(nqubits: int, reg: dict) -> dict:
    idx = np.arange(2 ** nqubits)
    out = {}
    for j, r in reg.items():
        v = np.zeros(len(idx), int)
        for q in r:
            v = v * 2 + ((idx >> q) & 1)
        out[j] = v
    return out


def validate_network(bn: MSBN, rng, n_ev: int = 50) -> dict:
    from qiskit.quantum_info import Statevector
    prep, reg = prep_circuit(bn)
    nqb = prep.num_qubits
    probs = Statevector(prep).probabilities()
    rv = _register_values(nqb, reg)
    g1 = g2 = 0.0
    invalid_any = np.zeros(len(probs), bool)
    for j in range(bn.n):
        mvec = np.bincount(rv[j], weights=probs, minlength=2 ** len(reg[j]))
        exact, _, _ = ve(bn, ancestors(bn, {j}), {}, [j])
        g1 = max(g1, float(np.abs(mvec[:bn.card[j]] - exact).max()))
        g2 = max(g2, float(mvec[bn.card[j]:].sum()))
        invalid_any |= rv[j] >= bn.card[j]
    g2_full = float(probs[invalid_any].sum())
    g3 = gp = 0.0
    for _ in range(n_ev):
        x = forward_sample(bn, rng)
        L = int(rng.integers(1, max(2, min(bn.n - 1, 6) + 1)))
        nodes = rng.permutation(bn.n)
        ev = {int(j): x[j] for j in nodes[:L]}
        mask = np.ones(len(probs), bool)
        for j, v in ev.items():
            mask &= rv[j] == v
        exact, _, _ = ve(bn, ancestors(bn, ev), ev, [])
        g3 = max(g3, abs(float(probs[mask].sum()) - float(exact)))
        t = int(nodes[L]) if L < bn.n else None
        if t is not None:  # pruning is exact
            s = int(rng.integers(bn.card[t]))
            a = lw_quantities(bn, t, s, ev, pruned=True)
            b_ = lw_quantities(bn, t, s, ev, pruned=False)
            gp = max(gp, abs(a["p"] - b_["p"]))
    return {"qubits": nqb, "gate1_max_marginal_error": g1, "gate2_max_invalid_mass_per_node": g2, "gate2_invalid_mass_full_state": g2_full,
            "gate3_max_P_evidence_error": g3, "pruning_max_posterior_change": gp, "evidence_configs": n_ev}


def validate():
    from qiskit.quantum_info import Statevector
    from pgmpy.utils import get_example_model
    from qddx.bn import BN, Struct
    from qddx.quantum import state_prep
    sys.path.insert(0, str(Path(__file__).parent))
    from oracle_v2 import cx, grover
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    res = {"networks": {}}
    for name in ("asia", "cancer", "earthquake", "survey", "sachs"):
        bn = from_pgmpy(get_example_model(name))
        res["networks"][name] = validate_network(bn, rng)
        print(name, res["networks"][name], flush=True)
    for i in range(20):
        while True:
            bn = random_msbn(int(rng.integers(4, 10)), 5, 3, rng, zero_frac=0.1 if i % 2 else 0.0)
            if sum(nq(k) for k in bn.card) <= 20:
                break
        res["networks"][f"random_{i:02d}"] = validate_network(bn, rng)
    print("random done", flush=True)
    # binary equivalence with the existing builder (tests/test_quantum.py covers the existing builder itself)
    eq_sv = eq_cx = 0
    for i in range(10):
        n = int(rng.integers(4, 12))
        pa = [tuple(sorted(rng.choice(j, size=int(rng.integers(0, min(j, 4) + 1)), replace=False).tolist())) if j else () for j in range(n)]
        b = BN(Struct([f"x{j}" for j in range(n)], pa), [np.stack([1 - p, p], 1) for p in (rng.uniform(0.05, 0.95, 2 ** len(x)) for x in pa)])
        old, (new, _) = state_prep(b), prep_circuit(from_binary(b))
        eq_sv = max(eq_sv, float(np.abs(Statevector(old).data - Statevector(new).data).max()))
        eq_cx = max(eq_cx, abs(cx(old) - cx(new)))
    res["binary_equivalence"] = {"cases": 10, "max_statevector_difference": eq_sv, "max_cx_difference": eq_cx}
    # oracle, clamped (V2) and one Grover step, on random multi-state networks
    worst_a = worst_c = worst_q = 0.0
    cases = 0
    while cases < 30:
        bn = random_msbn(int(rng.integers(4, 8)), 4, 3, rng)
        x = forward_sample(bn, rng)
        L = int(rng.integers(1, bn.n))
        order = rng.permutation(bn.n)
        ev = {int(j): x[j] for j in order[:L]}
        t = int(order[L])
        s = int(rng.integers(bn.card[t]))
        A0, _ = baseline_A(bn, range(bn.n), ev, t, s)
        A2, C = clamped_A(bn, range(bn.n), ev, t, s)
        if max(A0.num_qubits, A2.num_qubits) + 1 > 16:
            continue
        exact = float(ve(bn, ancestors(bn, {t, *ev}), ev, [t])[0][s])
        a0 = float(Statevector(A0).probabilities([A0.num_qubits - 1])[1])
        a2 = float(Statevector(A2).probabilities([A2.num_qubits - 1])[1])
        worst_a, worst_c = max(worst_a, abs(a0 - exact)), max(worst_c, abs(a2 * C - exact))
        for A, a, clean in ((A0, a0, 0), (A0, a0, 1), (A2, a2, 0), (A2, a2, 1)):
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(A.num_qubits + clean)
            qc.compose(A, range(A.num_qubits), inplace=True)
            qc.compose(grover(A, clean), inplace=True)
            got = float(Statevector(qc).probabilities([A.num_qubits - 1])[1])
            worst_q = max(worst_q, abs(got - math.sin(3 * math.asin(math.sqrt(a))) ** 2))
        cases += 1
    res["oracle_and_variants"] = {"cases": cases, "max_flag_error_v0": worst_a, "max_flag_error_v2_times_C": worst_c,
                                  "max_grover_step_error": worst_q}
    nets = res["networks"].values()
    res["gates"] = {"1_marginals_1e-12": max(v["gate1_max_marginal_error"] for v in nets) < 1e-12,
                    "2_invalid_mass_1e-12": max(max(v["gate2_max_invalid_mass_per_node"], v["gate2_invalid_mass_full_state"]) for v in nets) < 1e-12,
                    "3_P_evidence_1e-12": max(v["gate3_max_P_evidence_error"] for v in nets) < 1e-12,
                    "pruning_exact_1e-12": max(v["pruning_max_posterior_change"] for v in nets) < 1e-12,
                    "binary_equivalence": eq_sv < 1e-12 and eq_cx == 0,
                    "oracle_v0_v2_grover_1e-10": max(worst_a, worst_c, worst_q) < 1e-10,
                    "4_test_quantum": "run pytest tests/test_quantum.py separately"}
    (OUT / "multistate_validation.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["gates"], indent=1), json.dumps(res["binary_equivalence"]), json.dumps(res["oracle_and_variants"]))


def instance(bn: MSBN, rng) -> tuple[dict, int, int]:
    """W3.2 protocol: evidence size uniform on 1..min(10, n/4); nodes uniform without replacement; values by forward
    sampling; target uniform among the other nodes; its state uniform."""
    x = forward_sample(bn, rng)
    L = int(rng.integers(1, max(1, min(10, bn.n // 4)) + 1))
    order = rng.permutation(bn.n)
    ev = {int(j): x[j] for j in order[:L]}
    t = int(rng.choice(order[L:]))
    return ev, t, int(rng.integers(bn.card[t]))


def variants():
    """W1.3: V0-V3 CX per Grover step on multi-state benchmark instances (full network and pruned)."""
    from pgmpy.utils import get_example_model
    sys.path.insert(0, str(Path(__file__).parent))
    from oracle_v2 import cx, grover, _pad
    rng = np.random.default_rng(SEED)
    rows = []
    for name in ("insurance", "alarm", "hepar2", "child", "water"):
        bn = from_pgmpy(get_example_model(name))
        for i in range(3):
            ev, t, s = instance(bn, rng)
            for scope in ("full", "pruned"):
                if scope == "full":
                    e2, nodes = ev, list(range(bn.n))
                else:
                    e2, nodes = prune(bn, t, ev)
                A0, _ = baseline_A(bn, nodes, e2, t, s)
                A2, C = clamped_A(bn, nodes, e2, t, s)
                r = {"network": name, "instance": i, "scope": scope, "qubits_v0": A0.num_qubits, "qubits_v2": A2.num_qubits,
                     "evidence_used": len(e2)}
                for v, A, clean in (("v0", A0, 0), ("v1", A0, 1), ("v2", A2, 0), ("v3", A2, 1)):
                    r[f"A_{v}"] = cx(A) if clean == 0 else cx(_pad(A, clean))
                    r[f"Q_{v}"] = cx(grover(A, clean))
                for v in ("v0", "v1", "v2", "v3"):
                    r[f"ratio_{v}"] = r[f"Q_{v}"] / (2 * r["A_v0"])
                rows.append(r)
                print(name, i, scope, {k: round(r[k], 2) for k in r if k.startswith("ratio")}, r["qubits_v0"], flush=True)
    summ = {}
    for scope in ("full", "pruned"):
        R = [r for r in rows if r["scope"] == scope]
        summ[scope] = {v: [min(r[f"ratio_{v}"] for r in R), float(np.median([r[f"ratio_{v}"] for r in R])), max(r[f"ratio_{v}"] for r in R)]
                       for v in ("v0", "v1", "v2", "v3")}
    (OUT / "multistate_variants.json").write_text(json.dumps({"rows": rows, "ratio_min_median_max": summ,
                                                             "unit": "CX per Grover step / (2 x CX of the V0 problem circuit)"}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    {"validate": validate, "variants": variants}[sys.argv[1]]()
