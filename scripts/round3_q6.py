import json, sys
sys.path[:0] = ["src", "scripts"]  # Q6: does the saving need clean qubits? Run from the repo root.
import round3_q as R
from qiskit import QuantumCircuit
from oracle_v2 import cx
ex = {r["i"]: r for r in json.load(open("artifacts/quantum_paper/cost_exceptions.json"))}
out = []
for i, (bn, ev, t, s, meta) in enumerate(R._grid_cases()):
    if i not in ex:
        continue
    A, _ = R.baseline_A(bn, range(bn.n), ev, t, s)
    n = A.num_qubits
    Ad = A.inverse()
    # position of the first multi-qubit gate in A-dagger, and which qubits are untouched before it
    touched = set()
    first = None
    for k, ins in enumerate(Ad.data):
        qs = [Ad.find_bit(q).index for q in ins.qubits]
        if len(qs) > 2 and first is None:
            first = (k, ins.operation.name, len(qs), sorted(set(range(n)) - touched - set(qs)))
            break
        touched |= set(qs)
    chi = QuantumCircuit(n); chi.z(n - 1); chi.compose(Ad, inplace=True)
    dirty = QuantumCircuit(n)
    for q in range(n):
        dirty.ry(0.1, q)
    d2 = dirty.copy(); d2.compose(chi, inplace=True)
    rec = {"i": i, "saved": ex[i]["saving_applied"], "chi_Ad": ex[i]["seg"]["chi_Ad"], "A": ex[i]["seg"]["A"],
           "chi_Ad_dirty": cx(d2), "first_big_gate": first[1:3] if first else None, "clean_before_first_big": len(first[3]) if first else None,
           "ops_before_first_big": first[0] if first else None}
    out.append(rec)
json.dump(out, open("artifacts/quantum_paper/cost_exceptions_probe.json", "w"), indent=1)
import collections
for sv in (True, False):
    g = [r for r in out if r["saved"] == sv]
    print(sv, len(g), "dirty removes saving:", sum(r["chi_Ad_dirty"] == r["A"] for r in g),
          "first gate:", collections.Counter(str(r["first_big_gate"]) for r in g).most_common(4),
          "clean qubits before it:", collections.Counter(r["clean_before_first_big"] for r in g).most_common(6))
