"""Quantum symbolic engine: propositions are qubits, learned dependencies are multiplexed RY rotations,
graded evidence is a virtual-evidence qubit, and conjunctions of literals are MCX oracles on a flag qubit.

P(target | E) = P(target AND E) / P(E), each estimated by iterative amplitude estimation (IAE).

Two IAE executions share qiskit-algorithms' IterativeAmplitudeEstimation logic (schedule, Clopper-Pearson
intervals, oracle-query accounting):
  * circuit: real Grover circuits simulated by a Qiskit sampler (ideal shots or depolarizing noise);
  * emulated: outcomes drawn from the exact amplitude-amplification law P(1 | k) = sin^2((2k+1) theta),
    with sin^2(theta) = a computed from the validated circuit. This is exact for a noiseless device and
    is used where Grover powers are too deep to simulate gate by gate (rare evidence). The two are checked
    against each other in tests and in the report.
"""
from __future__ import annotations

import functools
import math
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate, UCRYGate
from qiskit.quantum_info import Statevector

from .bn import BN, Evidence, ancestral, posterior_exact, with_virtual


def state_prep(bn: BN) -> QuantumCircuit:
    """Qubit j <- proposition j; one multiplexed RY per node, controlled by its parent qubits."""
    st = bn.struct
    qc = QuantumCircuit(st.n, name="A")
    for j, cpt in enumerate(bn.cpts):
        pa = list(st.parents[j])
        angles = []
        for cs in range(2 ** len(pa)):  # qiskit order: bit i of cs <-> control pa[i]
            row = sum(((cs >> i) & 1) << (len(pa) - 1 - i) for i in range(len(pa)))  # first parent = MSB
            angles.append(2 * math.asin(math.sqrt(min(max(cpt[row, 1], 0.0), 1.0))))
        if not pa:
            qc.ry(angles[0], j)
        else:
            qc.append(UCRYGate(angles), [j] + pa)
    return qc


def conj_oracle(n: int, literals: dict[int, int]) -> QuantumCircuit:
    """Flag (qubit n) ^= AND_j (x_j == v_j): one multi-controlled X with a control state."""
    qc = QuantumCircuit(n + 1, name="U_E")
    if not literals:
        qc.x(n)
        return qc
    ctrl = sorted(literals)
    qc.append(MCXGate(len(ctrl), ctrl_state=sum(literals[q] << i for i, q in enumerate(ctrl))), ctrl + [n])
    return qc


def problem_circuit(prep: QuantumCircuit, orc: QuantumCircuit) -> QuantumCircuit:
    qc = QuantumCircuit(orc.num_qubits)
    qc.compose(prep, range(prep.num_qubits), inplace=True)
    qc.compose(orc, inplace=True)
    return qc


def basis_order(n: int) -> np.ndarray:
    """Statevector index of each joint state s (our node 0 = most significant bit of s; qiskit little-endian)."""
    s = np.arange(2 ** n)
    return sum((((s >> (n - 1 - j)) & 1) << j) for j in range(n))


def validate_prep(bn: BN, prep: QuantumCircuit) -> float:
    probs = Statevector(prep).probabilities()
    want = np.zeros_like(probs)
    want[basis_order(bn.struct.n)] = bn.joint()
    return float(np.abs(probs - want).max())


def _aer_gpu() -> bool:
    try:
        from qiskit_aer import AerSimulator
        return "GPU" in AerSimulator().available_devices()
    except Exception:  # noqa: BLE001 - optional accelerator
        return False


def flag_probability(circ: QuantumCircuit, method: str = "auto", sv_max: int = 26) -> float:
    """P(flag = 1) of a problem circuit, simulated gate by gate. auto: exact statevector up to sv_max qubits,
    then Aer statevector on the GPU (if qiskit-aer-gpu is installed), then Aer matrix-product-state (handles larger
    circuits when entanglement stays moderate)."""
    n = circ.num_qubits
    if method == "statevector" or (method == "auto" and n <= sv_max):
        return float(Statevector(circ).probabilities([n - 1])[1])
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    qc = circ.copy()
    qc.save_probabilities([n - 1], label="p")
    plan = [("statevector", "GPU")] if method == "auto" and _aer_gpu() else []
    plan += [("matrix_product_state", "CPU")] if method in ("auto", "mps") else [("statevector", "CPU")]
    for m, dev in plan:
        sim = AerSimulator(method=m, device=dev)
        try:
            res = sim.run(transpile(qc, sim, optimization_level=0)).result()
            return float(np.asarray(res.data()["p"])[1])
        except Exception:  # noqa: BLE001 - fall through to the next simulator (e.g. GPU out of memory)
            continue
    raise RuntimeError(f"no simulator could evaluate a {n}-qubit circuit")


def simulation_method(n_qubits: int, sv_max: int = 26) -> str:
    return "statevector" if n_qubits <= sv_max else ("aer-gpu statevector / mps" if _aer_gpu() else "matrix product state")


@functools.lru_cache(maxsize=2)
def _basis_pm():
    from qiskit.transpiler import generate_preset_pass_manager
    return generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])


def resources(qc: QuantumCircuit) -> dict:
    tq = _basis_pm().run(qc)
    ops = dict(tq.count_ops())
    return {"qubits": qc.num_qubits, "depth": tq.depth(), "gates": int(sum(ops.values())), "two_qubit_gates": int(ops.get("cx", 0))}


def make_sampler(kind: str, shots: int, seed: int, p2: float | None = None):
    if kind == "ideal":
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(default_shots=shots, seed=seed), None
    if kind == "aer":
        from qiskit_aer.primitives import SamplerV2
        return SamplerV2(default_shots=shots, seed=seed), _basis_pm()
    if kind == "depolarizing":
        from qiskit_aer.noise import NoiseModel, depolarizing_error
        from qiskit_aer.primitives import SamplerV2
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["rz", "sx", "x"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
        return SamplerV2(default_shots=shots, seed=seed, options={"backend_options": {"noise_model": nm}}), _basis_pm()
    raise ValueError(kind)


def _result(res, t0, shots: int) -> dict:
    """oracle_queries = model evaluations: each shot at Grover power k applies A (or its inverse) 2k + 1 times,
    the same unit as one classical sample of the network."""
    powers = getattr(res, "powers", None) or [0]
    return {"estimate": float(res.estimation), "ci": [float(x) for x in res.confidence_interval],
            "oracle_queries": int(2 * res.num_oracle_queries + shots * len(powers)), "max_grover_power": int(max(powers)),
            "seconds": time.perf_counter() - t0}


def iae_circuit(circ: QuantumCircuit, eps: float, alpha: float, sampler, pm=None, shots: int = 100) -> dict:
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
    t0 = time.perf_counter()
    prob = EstimationProblem(circ, objective_qubits=[circ.num_qubits - 1])
    return _result(IterativeAmplitudeEstimation(eps, alpha, sampler=sampler, transpiler=pm).estimate(prob), t0, shots)


class _Counts(dict):
    def get_counts(self):
        return self


class _AnalyticSampler:
    """Duck-typed sampler for IAE: the 'circuit' is ('Q^k', k); outcomes follow sin^2((2k+1) theta)."""

    def __init__(self, a: float, shots: int, seed: int):
        self.theta = math.asin(math.sqrt(min(max(a, 0.0), 1.0)))
        self.shots, self.rng = shots, np.random.default_rng(seed)

    def run(self, pubs):
        k = pubs[0][0][1]
        ones = int(self.rng.binomial(self.shots, math.sin((2 * k + 1) * self.theta) ** 2))
        res = type("R", (), {"data": type("D", (), {"keys": lambda s: ["c"], "c": _Counts({"1": ones, "0": self.shots - ones})})(),
                             "metadata": {"shots": self.shots}})()
        return type("J", (), {"result": lambda s: [res]})()


def iae_emulated(a: float, eps: float, alpha: float, shots: int, seed: int) -> dict:
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation

    class _IAE(IterativeAmplitudeEstimation):
        def construct_circuit(self, estimation_problem, k=0, measurement=False):
            return ("Q^k", k)

    t0 = time.perf_counter()
    dummy = QuantumCircuit(1)
    res = _IAE(eps, alpha, sampler=_AnalyticSampler(a, shots, seed)).estimate(EstimationProblem(dummy, objective_qubits=[0]))
    return _result(res, t0, shots)


def iae_relative(run, eps0: float, rel: float | None, ref: float | None = None, max_rounds: int = 8) -> dict:
    """Precision relative to the amplitude itself (or to `ref`) without knowing it in advance. qiskit's IAE stops
    on the width of its interval for theta (a = sin^2 theta), so a half-width h on a needs
    eps ~ h / (2 sqrt(a (1 - a))): for h = rel x a the cost is ~ 1/(rel sqrt(a)). Estimate at eps0, then tighten
    until the reported 95% interval's half-width is <= rel x (ref or estimate). `run(eps)` -> IAE result dict."""
    eps, calls, r = eps0, 0, None
    for _ in range(max_rounds if rel else 1):
        r = run(eps)
        calls += r["oracle_queries"]
        a, half = r["estimate"], (r["ci"][1] - r["ci"][0]) / 2
        target = rel * (ref if ref is not None else a) if rel else 0
        if not rel or (a > 0 and half <= target) or (a == 0 and ref is not None and r["ci"][1] <= target):
            break
        nxt = target / (2 * math.sqrt(a * (1 - a))) if 0 < a < 1 else eps / 10
        eps = min(nxt, eps / 2)
        if eps < 1e-9:
            break
    return {**r, "oracle_queries": calls}


def posterior_quantum(bn: BN, ev: Evidence, targets: list[str], eps: float, alpha: float, shots: int, seed: int,
                      mode: str = "circuit", sampler=None, pm=None, rel: float | None = None) -> tuple[dict[str, float], float, dict]:
    """P(t | E) = IAE[t AND E] / IAE[E] on the circuit of the virtual-evidence-extended network.

    mode: circuit  - Grover circuits run on `sampler` (default: ideal statevector shots);
          emulated - amplitude from the circuit's statevector, outcomes from the amplification law;
          analytic - same as emulated, amplitude taken from exact inference (equal to the circuit's
                     amplitude, checked by `check_circuit`); used where statevectors get large.
    rel: if set, P(E) is refined to relative precision `rel` and each P(t AND E) to rel x P(E) (see iae_relative),
         so every posterior has absolute error ~2 rel at 95%; eps is the first pass."""
    t0 = time.perf_counter()
    if mode != "analytic":  # simulate only the nodes that matter (exact; see ancestral)
        bn = ancestral(bn, [*ev.hard, *ev.soft, *targets])
    ext, obs = with_virtual(bn, ev)
    n = ext.struct.n
    prep = state_prep(ext) if mode != "analytic" else None
    if mode == "analytic":
        post, pev, _ = posterior_exact(bn, ev, targets)
        amp = {None: pev}
        amp.update({t: post[t] * pev for t in targets})
    calls = 0

    def est(literals: dict[int, int], s: int, key, ref=None) -> float:
        nonlocal calls
        if mode == "analytic":
            r = iae_relative(lambda e: iae_emulated(amp[key], e, alpha, shots, s), eps, rel, ref)
        else:
            circ = problem_circuit(prep, conj_oracle(n, literals))
            if mode == "emulated":
                a = flag_probability(circ)
                r = iae_relative(lambda e: iae_emulated(a, e, alpha, shots, s), eps, rel, ref)
            else:
                smp, p = (sampler, pm) if sampler is not None else make_sampler("ideal", shots, s)
                r = iae_relative(lambda e: iae_circuit(circ, e, alpha, smp, p, shots), eps, rel, ref)
        calls += r["oracle_queries"]
        return r["estimate"]

    pe = est(obs, seed, None)
    out = {}
    for i, t in enumerate(targets):  # P(t AND E) only needs precision relative to P(E): error on P(t|E) ~ 2 rel
        pte = est({**obs, ext.struct.index(t): 1}, seed + 1 + i, t, ref=pe if pe > 0 else None)
        out[t] = float(min(max(pte / pe, 0.0), 1.0)) if pe > 0 else float("nan")
    return out, pe, {"seconds": time.perf_counter() - t0, "oracle_calls": calls, "qubits": n + 1,
                     "method": f"iterative amplitude estimation ({mode})"}


def circuit_qubits(bn: BN, ev: Evidence, targets: list[str]) -> int:
    """Qubits of the problem circuit posterior_quantum / check_circuit simulate (ancestral sub-network + virtual + flag)."""
    return ancestral(bn, [*ev.hard, *ev.soft, *targets]).struct.n + len(ev.soft) + 1


def check_circuit(bn: BN, ev: Evidence, targets: list[str], sv_max: int = 26) -> dict:
    """Gate-level check of the encoding: flag probabilities of the evidence and target oracles (simulated) against
    exact inference. Returns the largest absolute amplitude error, the simulator used and the circuit resources."""
    bn = ancestral(bn, [*ev.hard, *ev.soft, *targets])
    ext, obs = with_virtual(bn, ev)
    prep = state_prep(ext)
    n = ext.struct.n
    post, pe, _ = posterior_exact(bn, ev, targets)
    err = abs(flag_probability(problem_circuit(prep, conj_oracle(n, obs)), sv_max=sv_max) - pe)
    for t in targets:
        circ = problem_circuit(prep, conj_oracle(n, {**obs, ext.struct.index(t): 1}))
        err = max(err, abs(flag_probability(circ, sv_max=sv_max) - post[t] * pe))
    return {"max_abs_error": float(err), "qubits": n + 1, "simulator": simulation_method(n + 1, sv_max),
            **{f"prep_{k}": v for k, v in resources(prep).items() if k != "qubits"}}
