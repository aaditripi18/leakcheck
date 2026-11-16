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


def test_many_small_blocks_give_the_same_answer():
    """One training block on the device at a time: the shortlists merged across many blocks
    must give the answer of one large block."""
    s = _split(np.random.default_rng(11), False)
    d0, n0 = near_torch.nearest(s, device="cpu", block=10_000, rows=600)
    d1, n1 = near_torch.nearest(s, device="cpu", block=97, rows=53)
    assert np.array_equal(d0, d1) and np.array_equal(n0, n1)


@pytest.mark.parametrize("scale_spread", [False, True])
def test_float32_pass_gives_the_float64_answer(scale_spread):
    """The certified float32 shortlist returns exactly what the float64 shortlist returns, and
    certifies most rows on ordinary data; rows it cannot certify are redone in float64."""
    s = _split(np.random.default_rng(3), scale_spread)
    d64, n64 = near_torch.nearest(s, device="cpu", block=700, rows=128, fast=False)
    d32, n32 = near_torch.nearest(s, device="cpu", block=700, rows=128, fast=True)
    assert np.array_equal(d64, d32) and np.array_equal(n64, n32)
    if not scale_spread:
        assert near_torch.nearest.redone < 0.05 * len(s.test)


def test_float32_pass_redoes_rows_it_cannot_certify():
    """Features near 3e5 (RT-IoT2022): float32 cannot resolve a 0.01 distance there, so those rows
    must go through float64, and the answer still equals near.nearest."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=(800, 6))
    x[:, 0] = 3e5 + rng.normal(0, 1e-3, 800)
    te = x[:100] + rng.normal(0, 1e-4, size=(100, 6))
    tr = pd.DataFrame(x, columns=[f"f{i}" for i in range(6)]).assign(y=0)
    tst = pd.DataFrame(te, columns=[f"f{i}" for i in range(6)]).assign(y=0)
    s = Split(tr, tst, "y")
    d0, _ = near.nearest(s)
    d1, _ = near_torch.nearest(s, device="cpu", block=300, rows=40)
    assert np.allclose(d0, d1, rtol=0, atol=1e-9)
    assert np.array_equal(d0 <= 0.01, d1 <= 0.01)
