from pathlib import Path

import numpy as np
import pytest

from qddx.bn import BN, Struct

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    from qddx.cli import load_config
    return load_config(str(ROOT / "configs" / "qddx.yaml"), [])


def random_bn(n=8, max_pa=2, seed=0) -> BN:
    rng = np.random.default_rng(seed)
    parents = [tuple(sorted(rng.choice(j, size=min(j, int(rng.integers(0, max_pa + 1))), replace=False).tolist())) for j in range(n)]
    cpts = []
    for pa in parents:
        p1 = rng.uniform(0.05, 0.95, 2 ** len(pa))
        cpts.append(np.stack([1 - p1, p1], 1))
    return BN(Struct([f"x{j}" for j in range(n)], parents), cpts)
