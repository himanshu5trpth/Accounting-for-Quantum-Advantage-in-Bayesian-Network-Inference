import numpy as np

from qddx.bn import Evidence, info_gain, learn, posterior_exact, posterior_lw, posterior_rejection, with_virtual, drivers
from conftest import random_bn

EV = Evidence({"x4": 1, "x6": 0}, {"x7": 0.8, "x5": 0.3})
T = ["x0", "x1", "x2"]


def test_virtual_evidence_matches_extended_network():
    bn = random_bn()
    p, pe, _ = posterior_exact(bn, EV, T)
    ext, obs = with_virtual(bn, EV)
    hard = Evidence({ext.struct.names[j]: v for j, v in obs.items()}, {})
    p2, pe2, _ = posterior_exact(ext, hard, T)
    assert abs(pe - pe2) < 1e-12 and all(abs(p[t] - p2[t]) < 1e-12 for t in T)


def test_sampling_engines_agree_with_exact():
    bn = random_bn()
    p, _, _ = posterior_exact(bn, EV, T)
    for f in (posterior_rejection, posterior_lw):
        q, _, _ = f(bn, EV, T, 400_000, 1)
        assert all(abs(p[t] - q[t]) < 0.03 for t in T)


def test_learn_recovers_dependency():
    rng = np.random.default_rng(0)
    a = rng.random(20000) < 0.3
    b = np.where(a, rng.random(20000) < 0.9, rng.random(20000) < 0.1)
    c = rng.random(20000) < 0.5
    y = np.stack([a, c, b], 1).astype(np.int8)
    y[:500, 2] = -1  # missing values are skipped
    bn = learn(["a", "c", "b"], y, max_parents=2)
    assert bn.struct.parents[2] == (0,)
    assert abs(bn.cpts[2][1, 1] - 0.9) < 0.02


def test_info_gain_and_drivers():
    bn = random_bn()
    g = info_gain(bn, EV, ["x3", "x5"], T)
    assert all(v >= -1e-12 for v in g.values())
    d = drivers(bn, EV, "x0", min_effect=0.0)
    assert all(f in ("x4", "x6", "x7", "x5") for f, _ in d)


def test_variable_elimination_matches_enumeration():
    from qddx.bn import info_gain_enum, posterior_enum
    for seed in range(15):
        bn = random_bn(10, max_pa=3, seed=seed)
        rng = np.random.default_rng(seed)
        names = bn.struct.names
        hard = {names[j]: int(rng.integers(0, 2)) for j in rng.choice(10, 3, replace=False)}
        soft = {n: float(rng.uniform(0.1, 0.9)) for n in names if n not in hard and rng.random() < 0.3}
        ev = Evidence(hard, soft)
        T = [n for n in names if n not in hard][:4] + list(hard)[:1]
        a, pa, _ = posterior_exact(bn, ev, T)
        b, pb, _ = posterior_enum(bn, ev, T)
        assert abs(pa - pb) < 1e-12 and all(abs(a[t] - b[t]) < 1e-12 for t in T)
        cand = [n for n in names if n not in hard and n not in T]
        g1, g2 = info_gain(bn, ev, cand, T[:4]), info_gain_enum(bn, ev, cand, T[:4])
        assert all(abs(g1[c] - g2[c]) < 1e-10 for c in cand)


def test_variable_elimination_scales_to_large_vocabularies():
    import time
    for n in (24, 64, 128):
        bn = random_bn(n, max_pa=4, seed=1)
        names = bn.struct.names
        rng = np.random.default_rng(0)
        hard = {names[j]: int(rng.integers(0, 2)) for j in rng.choice(n, n // 2, replace=False)}
        soft = {x: 0.7 for x in names[: n // 8] if x not in hard}
        T = [x for x in names if x not in hard][:6]
        t0 = time.perf_counter()
        p, pe, _ = posterior_exact(bn, Evidence(hard, soft), T)
        assert time.perf_counter() - t0 < 1.0 and 0 < pe < 1 and all(0 <= v <= 1 for v in p.values())


def test_ancestral_subnetwork_keeps_posteriors():
    from conftest import random_bn
    from qddx.bn import Evidence, ancestral, posterior_exact
    bn = random_bn(n=12, max_pa=3, seed=4)
    nm = bn.struct.names
    ev, t = Evidence({nm[5]: 1}, {nm[7]: 0.7}), nm[9]
    sub = ancestral(bn, [nm[5], nm[7], t])
    assert sub.struct.n < bn.struct.n
    assert abs(posterior_exact(bn, ev, [t])[0][t] - posterior_exact(sub, ev, [t])[0][t]) < 1e-12
