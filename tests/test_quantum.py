import math

import numpy as np
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector

from qddx.bn import Evidence, posterior_exact
from qddx.quantum import (check_circuit, conj_oracle, flag_probability, iae_emulated, iae_relative, posterior_quantum, problem_circuit,
                          state_prep, validate_prep)
from conftest import random_bn

EV = Evidence({"x3": 1}, {"x4": 0.7})


def test_state_preparation_and_oracles_are_exact():
    bn = random_bn(6)
    assert validate_prep(bn, state_prep(bn)) < 1e-12
    assert check_circuit(bn, EV, ["x0", "x1"])["max_abs_error"] < 1e-12


def test_grover_power_law_justifies_emulation():
    bn = random_bn(5, seed=3)
    circ = problem_circuit(state_prep(bn), conj_oracle(5, {2: 1, 4: 0}))
    a = flag_probability(circ)
    theta = math.asin(math.sqrt(a))
    G = GroverOperator(conj_oracle_phase(5), state_preparation=circ.copy())
    for k in (1, 2, 3):
        qc = circ.copy()
        for _ in range(k):
            qc.compose(G, inplace=True)
        p1 = Statevector(qc).probabilities([5])[1]
        assert abs(p1 - math.sin((2 * k + 1) * theta) ** 2) < 1e-9


def conj_oracle_phase(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n + 1)
    qc.z(n)
    return qc


def test_iae_modes_agree_with_exact():
    bn = random_bn(5, seed=1)
    ex, _, _ = posterior_exact(bn, EV, ["x0"])
    for mode in ("analytic", "emulated", "circuit"):
        q, _, cost = posterior_quantum(bn, EV, ["x0"], 0.01, 0.05, 100, 7, mode=mode, rel=0.02 if mode != "circuit" else None)
        assert abs(q["x0"] - ex["x0"]) < 0.06, mode
        assert cost["oracle_calls"] > 0


def test_relative_iae_cost_scales_like_inverse_sqrt():
    """10x relative precision on a 100x rarer amplitude costs ~10x more oracle calls (rejection: ~100x)."""
    calls = []
    for a in (1e-2, 1e-4):
        rs = [iae_relative(lambda e, s=s: iae_emulated(a, e, 0.05, 100, s), 0.01, 0.1) for s in range(8)]
        assert np.median([abs(r["estimate"] - a) / a for r in rs]) < 0.1
        calls.append(np.mean([r["oracle_queries"] for r in rs]))
    assert 3 < calls[1] / calls[0] < 30


def test_mps_simulator_agrees_with_statevector():
    bn = random_bn(8, seed=2)
    ev = Evidence({"x3": 1}, {"x5": 0.7})
    a = check_circuit(bn, ev, ["x0"])
    b = check_circuit(bn, ev, ["x0"], sv_max=4)  # forces the matrix-product-state path
    assert a["max_abs_error"] < 1e-12 and b["max_abs_error"] < 1e-9 and b["simulator"] != "statevector"
