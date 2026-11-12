import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pairs import Ranked, d_separated, pair_from_table  # noqa: E402

from qddx.bn import Evidence, marginal, posterior_exact


def test_weighted_metrics_match_sklearn_with_ties():
    rng = np.random.default_rng(0)
    s = np.round(rng.random(3000), 2)  # many ties, like network scores
    y = (rng.random(3000) < 0.1 + 0.5 * s).astype(int)
    w = rng.integers(0, 4, 3000).astype(float)  # bootstrap-style weights, including zeros
    ap, auc, rec = Ranked(s, y).all(w, 0.1)
    assert abs(ap - average_precision_score(y, s, sample_weight=w)) < 1e-9
    assert abs(auc - roc_auc_score(y, s, sample_weight=w)) < 1e-9
    o = np.argsort(-s, kind="mergesort")
    k = np.searchsorted(np.cumsum(w[o]), 0.1 * w.sum())
    assert abs(rec - (w[o][: k + 1] * y[o][: k + 1]).sum() / (w * y).sum()) < 1e-12


def test_d_separation_implies_joint_equals_product():
    from conftest import random_bn
    rng = np.random.default_rng(1)
    found = 0
    for seed in range(30):
        bn = random_bn(n=9, max_pa=2, seed=seed)
        a, b = 7, 8
        given = {int(g) for g in rng.choice(7, size=int(rng.integers(0, 4)), replace=False)}
        E = Evidence({bn.struct.names[g]: int(rng.integers(0, 2)) for g in given}, {})
        t = marginal(bn, E, [a, b])
        t = t / t.sum()
        indep = abs(t[1, 1] - t[1].sum() * t[:, 1].sum()) < 1e-12
        if d_separated(bn.struct.parents, a, b, given):
            found += 1
            assert indep
    assert found > 0


def test_pair_from_table():
    from conftest import random_bn
    bn = random_bn(n=6, max_pa=2, seed=3)
    tab = marginal(bn, Evidence({}, {}), list(range(6))).reshape(1, -1)
    both, pi, pj = pair_from_table(tab, 1, 4)
    post = posterior_exact(bn, Evidence({bn.struct.names[1]: 1}, {}), [bn.struct.names[4]])[0][bn.struct.names[4]]
    assert abs(pi[0] - tab[0].reshape((2,) * 6)[:, 1].sum()) < 1e-12 and abs(both[0] - pi[0] * post) < 1e-12
