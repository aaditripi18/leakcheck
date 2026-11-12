"""The GPU search (leakcheck.checks.near_torch) gives near.nearest's answers, including on
features scaled into the hundreds of thousands, where float32 would lose a 0.01 distance."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from leakcheck import Split  # noqa: E402
from leakcheck.checks import near, near_torch  # noqa: E402


def _split(rng, scale_spread: bool) -> Split:
    n_tr, n_te, d = 3000, 600, 12
    x = rng.normal(size=(n_tr, d))
    if scale_spread:
        x[:, :3] = rng.choice([0.0, 1e-6], size=(n_tr, 3)) + rng.normal(0, 5.0, size=(n_tr, 3)) * (rng.random((n_tr, 3)) < 0.02)
    te = x[rng.choice(n_tr, n_te)] + rng.normal(0, 1e-3, size=(n_te, d)) * (rng.random((n_te, 1)) < 0.5)
    cat_tr, cat_te = rng.choice(["a", "b", "c"], n_tr), rng.choice(["a", "b", "c", "z"], n_te)
    tr = pd.DataFrame(x, columns=[f"f{i}" for i in range(d)]).assign(c=cat_tr, y=rng.integers(0, 2, n_tr))
    tst = pd.DataFrame(te, columns=[f"f{i}" for i in range(d)]).assign(c=cat_te, y=rng.integers(0, 2, n_te))
    return Split(tr, tst, "y")


@pytest.mark.parametrize("scale_spread", [False, True])
def test_same_decisions_and_distances_as_near(scale_spread):
    s = _split(np.random.default_rng(7), scale_spread)
    d0, _ = near.nearest(s)
    d1, _ = near_torch.nearest(s, device="cpu", block=1000, rows=128)
    assert np.array_equal(np.isinf(d0), np.isinf(d1))          # unmatched categorical buckets agree
    fin = np.isfinite(d0)
    assert np.array_equal(d0[fin] <= 0.01, d1[fin] <= 0.01)
    assert np.allclose(d0[fin], d1[fin], rtol=0, atol=1e-6)
    if scale_spread:
        xtr, _ = near.embed(s)
        assert np.abs(xtr).max() > 1e4                          # the case float32 gets wrong
