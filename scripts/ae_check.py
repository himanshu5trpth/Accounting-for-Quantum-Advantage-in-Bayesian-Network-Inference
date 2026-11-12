"""Is amplitude estimation's fixed overhead real, and how much of it is a choice of settings?

  python scripts/ae_check.py circuits [N_PROFILES]   # real IAE circuits (Aer, <= 22 qubits) vs the amplification law
  python scripts/ae_check.py sweep                   # model evaluations to 10% relative error vs shots / alpha / amplitude

Cost unit: one application of the state preparation A or its inverse (= one pass through the network). A shot at
Grover power k uses A 2k + 1 times. Success: 90% of repeats within 10% relative error.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qddx.bn import BN, Evidence, ancestral, posterior_exact, with_virtual
from qddx.quantum import conj_oracle, iae_circuit, iae_emulated, iae_relative, make_sampler, problem_circuit, state_prep

KB, VOC = Path(os.environ.get("QDDX_KB", "artifacts/v2.1/kb")), Path(os.environ.get("QDDX_VOCAB", "configs/vocab_v2.1.json"))
OUT, SEED = Path("artifacts/ae_check"), 2026
OUT.mkdir(parents=True, exist_ok=True)


def circuits(n_prof: int, repeats: int = 8, eps: float = 0.01):
    """Same amplitude, same IAE settings (eps, alpha 0.05, 100 shots): real circuits on Aer vs the analytic law."""
    bn = BN.from_dict(json.loads((KB / "bn.json").read_text()))
    P = json.loads(VOC.read_text())["propositions"]
    dx = [k for k, p in P.items() if p["kind"] == "diagnosis"]
    ev = [k for k, p in P.items() if p["kind"] != "diagnosis"]
    rng = np.random.default_rng(SEED)
    rows, tries = [], 0
    while len(rows) < n_prof and tries < 500:
        tries += 1
        t = dx[rng.integers(len(dx))]
        E = Evidence({k: int(rng.random() < 0.3) for k in rng.choice(ev, int(rng.integers(1, 4)), replace=False)}, {})
        sub = ancestral(bn, [*E.hard, t])
        ext, obs = with_virtual(sub, E)
        if not 6 <= ext.struct.n + 1 <= 16:  # Aer runs of deep Grover powers get slow above this on 2 cores
            continue
        lit = {**obs, ext.struct.index(t): 1}
        a = posterior_exact(sub, E, [t])[0][t] * posterior_exact(sub, E, [t])[1]
        if a < 1e-3:
            continue
        circ = problem_circuit(state_prep(ext), conj_oracle(ext.struct.n, lit))
        rec = {"qubits": ext.struct.n + 1, "a": a, "circuit": [], "law": []}
        for r in range(repeats):
            smp, pm = make_sampler("aer", 100, SEED + r)
            c = iae_circuit(circ, eps, 0.05, smp, pm, 100)
            e = iae_emulated(a, eps, 0.05, 100, SEED + r)
            rec["circuit"].append((c["estimate"], c["oracle_queries"]))
            rec["law"].append((e["estimate"], e["oracle_queries"]))
        rows.append(rec)
        cc, ll = np.array(rec["circuit"]), np.array(rec["law"])
        print(f"{rec['qubits']} qubits, a={a:.4f}: circuit cost {cc[:, 1].mean():,.0f} abs err {np.abs(cc[:, 0] - a).mean():.4f} | "
              f"law cost {ll[:, 1].mean():,.0f} abs err {np.abs(ll[:, 0] - a).mean():.4f}", flush=True)
    (OUT / "circuits.json").write_text(json.dumps(rows))


def cost_rel(a: float, shots: int, alpha: float, repeats: int = 30, rel: float = 0.1) -> float | None:
    """Mean model evaluations for 90% of repeats within `rel` of a (relative-precision IAE, law)."""
    for f in (2, 1, 0.5, 0.25, 0.125, 0.0625):
        errs, calls = [], []
        for r in range(repeats):
            e = iae_relative(lambda eps: iae_emulated(a, eps, alpha, shots, SEED + 31 * r), 0.01, f * rel)
            errs.append(abs(e["estimate"] - a) / a)
            calls.append(e["oracle_queries"])
        if np.quantile(errs, 0.9, method="higher") <= rel:
            return float(np.mean(calls))
    return None


def sweep():
    rows = []
    for a in (0.25, 0.025, 0.0025):  # the qLW amplitudes seen on real patients; fewer shots get very slow in qiskit
        for shots in (16, 100):
            for alpha in (0.05, 0.2):
                t0 = time.perf_counter()
                c = cost_rel(a, shots, alpha)
                rows.append({"a": a, "shots": shots, "alpha": alpha, "cost": c})
                print(f"a={a:g} shots={shots} alpha={alpha}: {c if c is None else f'{c:,.0f}'} ({time.perf_counter() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "sweep.csv", index=False)
    base = d[(d.shots == 100) & (d.alpha == 0.05)].set_index("a").cost
    best = d.dropna().loc[d.dropna().groupby("a").cost.idxmin()].set_index("a")
    for a in base.index:
        print(f"a={a:g}: used settings {base[a]:,.0f}; best {best.cost[a]:,.0f} (shots {best.shots[a]}, alpha {best.alpha[a]}); "
              f"Heisenberg-limit reference pi/(rel*sqrt(a)) = {math.pi / (0.1 * math.sqrt(a)):,.0f}")


if __name__ == "__main__":
    {"circuits": lambda: circuits(int(sys.argv[2]) if len(sys.argv) > 2 else 4), "sweep": sweep}[sys.argv[1]]()
