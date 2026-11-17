"""The GPU search (leakcheck.checks.near_torch) gives near.nearest's answers, including on
features scaled into the hundreds of thousands, where float32 would lose a 0.01 distance."""
import os

import numpy as np
import pandas as pd
import pytest

# CI's gpu-search job sets LEAKCHECK_REQUIRE_TORCH=1: there these tests must run, not skip
torch = pytest.importorskip("torch") if not os.environ.get("LEAKCHECK_REQUIRE_TORCH") else __import__("torch")

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


def _repeated(rng, copies: int, cats: bool) -> Split:
    """Training rows that are copies of a few hundred distinct rows (a split without identifier
    columns, as CSE-CIC-IDS2018), shuffled; test rows near those rows but none an exact copy."""
    base = rng.normal(size=(300, 8))
    x = np.repeat(base, copies, axis=0)
    order = rng.permutation(len(x))
    x = x[order]
    te = base[rng.choice(len(base), 400)] + rng.normal(0, 1e-3, size=(400, 8))
    cols = [f"f{i}" for i in range(8)]
    tr = pd.DataFrame(x, columns=cols).assign(y=0)
    tst = pd.DataFrame(te, columns=cols).assign(y=0)
    if cats:  # one category per distinct row, so the copies share a bucket
        tr["c"] = np.repeat(rng.choice(["a", "b"], len(base)), copies)[order]
        tst["c"] = rng.choice(["a", "b"], len(te))
    return Split(tr, tst, "y")


@pytest.mark.parametrize("cats", [False, True])
def test_repeated_training_rows_are_searched_once(cats):
    """A nearest row with more than K copies could not be certified, so every test row went through
    float64. Searched once, the rows are certified, and the distances are near.nearest's exactly;
    the row returned may be any copy of near.nearest's row (identical vector)."""
    s = _repeated(np.random.default_rng(13), copies=3 * near_torch.K, cats=cats)
    d0, n0 = near.nearest(s)
    d1, n1 = near_torch.nearest(s, device="cpu", block=1000, rows=128, distinct=False)
    assert near_torch.nearest.redone > 0.9 * len(s.test)     # the failure being fixed
    d2, n2 = near_torch.nearest(s, device="cpu", block=1000, rows=128)
    assert near_torch.nearest.redone < 0.05 * len(s.test)
    assert near_torch.nearest.searched == 300
    xtr, _ = near.embed(s)
    for d, n in ((d1, n1), (d2, n2)):
        assert np.array_equal(d0, d)
        assert np.array_equal(xtr[n0], xtr[n])
    # the first copy is returned
    th = s.train_hash
    assert all(n == np.flatnonzero(th == th[n])[0] for n in n2)


def test_rows_equal_only_after_rounding_are_both_kept():
    """Rows whose hash matches (features equal to 30 mantissa bits) but whose vectors differ are both
    searched: the distance found is the exact minimum over every training row. (The differences,
    1e-12, are below what near.nearest's expanded-formula ranking resolves, so the reference here is
    the direct computation.)"""
    rng = np.random.default_rng(17)
    base = rng.normal(size=(50, 4))
    x = np.vstack([base, base * (1 + 1e-12)] * 6)
    te = base * (1 + 1e-12) + 1e-13 * (rng.random((50, 4)) < 0.5)
    cols = [f"f{i}" for i in range(4)]
    s = Split(pd.DataFrame(x, columns=cols).assign(y=0), pd.DataFrame(te, columns=cols).assign(y=0), "y")
    assert len(np.unique(s.train_hash)) == 50
    xtr, xte = near.embed(s)
    exact = np.sqrt(((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(2))
    d1, n1 = near_torch.nearest(s, device="cpu", block=97, rows=16)
    assert np.array_equal(d1, exact.min(1))
    assert np.array_equal(d1, exact[np.arange(len(te)), n1])
    assert near_torch.nearest.searched == 100


def test_distinct_compacts_the_whole_matrix_in_place():
    rng = np.random.default_rng(19)
    base = rng.normal(size=(40, 3)).astype(np.float32)
    x = base[rng.integers(0, 40, 1000)]
    x0 = x.copy()
    th = np.asarray(pd.util.hash_pandas_object(pd.DataFrame(x), index=False))
    keep_ref = np.sort(np.unique(th, return_index=True)[1])
    members, xm = near_torch._distinct(x, np.arange(len(x)), th, chunk=64)
    assert np.array_equal(members, keep_ref)
    assert np.shares_memory(xm, x) and len(xm) == len(members)
    assert np.array_equal(xm, np.asarray(pd.DataFrame(x0)).astype(np.float32)[members])
    assert len(np.unique(xm, axis=0)) == len(members)


@pytest.mark.parametrize("spread", [1e-2, 1e-4, 1e-6, 1e-7])
@pytest.mark.parametrize("offset", [0.0, 50.0])
def test_float32_product_bound_holds_in_tight_clusters(spread, offset):
    """The float32 pass uses |q|^2 + |t|^2 - 2 q.t, whose error grows with the norms, not the distance.
    Clusters of training rows closer together than that error (and far from the centre) must still
    give exactly the float64 answer: rows the bound cannot separate are redone."""
    rng = np.random.default_rng(int(spread * 1e9) + int(offset))
    centres = rng.normal(size=(30, 10)) + offset
    x = np.repeat(centres, 40, axis=0) + rng.normal(0, spread, size=(1200, 10))
    te = centres[rng.integers(0, 30, 300)] + rng.normal(0, spread, size=(300, 10))
    cols = [f"f{i}" for i in range(10)]
    s = Split(pd.DataFrame(x, columns=cols).assign(y=0), pd.DataFrame(te, columns=cols).assign(y=0), "y")
    xtr, xte = near.embed(s)
    exact = np.sqrt(((xte[:, None, :] - xtr[None, :, :]) ** 2).sum(2)).min(1)
    d64, n64 = near_torch.nearest(s, device="cpu", block=500, rows=64, fast=False)
    d32, n32 = near_torch.nearest(s, device="cpu", block=500, rows=64, fast=True)
    assert np.allclose(d32, exact, rtol=1e-12, atol=0)       # the true minimum (summation order aside)
    assert np.array_equal(d64, d32) and np.array_equal(n64, n32)


# --- The hard cases, against near.nearest and a direct computation --------------------------------

def _direct(s):
    """Each test row's distance to every training row of its bucket, computed one test row at a time
    by the formula near.nearest uses: the reference both searches must reproduce bit for bit."""
    xtr, xte = near.embed(s)
    ktr, kte = near._cat_keys(s)
    out = []
    for i in range(len(xte)):
        m = np.flatnonzero(ktr == kte[i])
        out.append((m, np.sqrt(np.sum((xtr[m] - xte[i]) ** 2, axis=1))))
    return out


def _hard(kind, rng, n_filler=6000):
    d = 8
    if kind == "large_norms":   # robust scaling to ~1e5 and beyond (RT-IoT2022): tiny IQRs, rare far values
        x = rng.normal(size=(n_filler, d))
        x[:, 0] = 3e5 + rng.normal(0, 1e-3, n_filler)
        x[:, 1] = rng.choice([0.0, 1e-6], n_filler) + rng.normal(0, 5.0, n_filler) * (rng.random(n_filler) < 0.02)
        te = x[rng.integers(0, n_filler, 500)] + rng.normal(0, 1e-4, size=(500, d)) * (rng.random((500, 1)) < 0.7)
    elif kind in ("near_ties", "exact_ties"):
        # column 0 has median 0 and the transformation is odd (signed log, then scaling about the
        # median), so training rows at +1 and -(1 + rel) in column 0, equal to the test row elsewhere,
        # are exactly (rel = 0) or almost equally far from it; filler rows sit far away
        base = rng.normal(size=(500, d))
        base[:, 0] = 0.0
        rel = 0.0 if kind == "exact_ties" else rng.choice([1e-3, 1e-5, 1e-6, 1e-7], 500)
        plus, minus = base.copy(), base.copy()
        plus[:, 0], minus[:, 0] = 1.0, -(1.0 + rel)
        filler = rng.normal(size=(n_filler, d)) + 8.0
        filler[:, 0] = np.where(rng.random(n_filler) < 0.6, 0.0, rng.choice([-2.0, 2.0], n_filler))
        x = np.vstack([plus, minus, filler])
        te = base
    else:                       # duplicates: copies of 300 rows; some test rows are exact copies
        base = rng.normal(size=(300, d))
        x = base[rng.integers(0, 300, n_filler)]
        te = base[rng.integers(0, 300, 500)] + rng.normal(0, 1e-3, size=(500, d)) * (rng.random((500, 1)) < 0.6)
    cols = [f"f{i}" for i in range(d)]
    x = x[rng.permutation(len(x))]
    return Split(pd.DataFrame(x, columns=cols).assign(y=0), pd.DataFrame(te, columns=cols).assign(y=0), "y")


def _assert_exact(s, ref, d, n):
    """Every distance is the true minimum, bit for bit, and the row returned is at that distance
    (any of equidistant rows is allowed)."""
    best = np.array([r.min() for _, r in ref])
    assert np.array_equal(d, best)
    for i, (m, r) in enumerate(ref):
        j = np.searchsorted(m, n[i])
        assert m[j] == n[i] and r[j] == d[i]


KINDS = ["large_norms", "near_ties", "exact_ties", "duplicates"]


@pytest.mark.parametrize("float32_embedding", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_hard_cases_equal_near_nearest(kind, float32_embedding, monkeypatch):
    """Large feature norms, near-ties, exact ties and exact duplicates, in one bucket searched in
    several blocks, with the embedding in float64 and (as on the giant cases) float32: the GPU search's
    distances equal near.nearest's and the direct computation's exactly, and the rows match."""
    if float32_embedding:
        monkeypatch.setattr(near, "LARGE", 0)
    s = _hard(kind, np.random.default_rng(1))
    ref = _direct(s)
    xtr, _ = near.embed(s)
    assert xtr.dtype == (np.float32 if float32_embedding else np.float64)
    # the hard condition is really there
    if kind == "large_norms":
        assert np.abs(xtr).max() > 1e4
    if kind == "exact_ties":
        assert sum((r == r.min()).sum() >= 2 for _, r in ref) == 500
    if kind == "near_ties":
        gaps = [np.partition(r, 1)[1] / np.partition(r, 1)[0] - 1 for _, r in ref]
        assert sum(g < 1e-5 for g in gaps) >= 200 and min(gaps) > 0
    if kind == "duplicates":
        assert len(np.unique(s.train_hash)) == 300 and (np.isin(s.test_hash, s.train_hash)).sum() >= 100
    d0, n0 = near.nearest(s)
    d1, n1 = near_torch.nearest(s, device="cpu", block=997, rows=128)
    _assert_exact(s, ref, d0, n0)
    _assert_exact(s, ref, d1, n1)
    assert np.array_equal(d0, d1)
    if kind == "large_norms":
        assert near_torch.nearest.redone > 0      # float32 cannot certify these: float64 decided them


def test_one_large_bucket_in_many_blocks(monkeypatch):
    """One bucket of about 38,000 rows (no categorical columns, as CSE-CIC-IDS2018 without identifiers and
    CIC-IoT2023), searched in 38 training blocks and 4 query blocks, with every hard case mixed in and
    the float32 embedding: the same answers as near.nearest."""
    monkeypatch.setattr(near, "LARGE", 0)
    rng = np.random.default_rng(2)
    parts = [_hard(k, rng, n_filler=9000) for k in KINDS]
    tr = pd.concat([p.train for p in parts], ignore_index=True)
    te = pd.concat([p.test for p in parts], ignore_index=True)
    s = Split(tr.sample(frac=1, random_state=0).reset_index(drop=True), te, "y")
    assert len(s.train) > 37_000 and near.categorical_columns(s) == []
    ref = _direct(s)
    d0, n0 = near.nearest(s)
    d1, n1 = near_torch.nearest(s, device="cpu", block=997, rows=512)
    _assert_exact(s, ref, d0, n0)
    _assert_exact(s, ref, d1, n1)
    assert np.array_equal(d0, d1)


def test_a_large_search_logs_its_progress(monkeypatch, capsys):
    """A bucket worth watching prints a line every `progress` training blocks (rows done, elapsed time,
    time to go), and its float64 pass and its end, so a slow part shows in the log while it runs."""
    monkeypatch.setattr(near_torch, "PROGRESS_MIN", 1)
    s = _hard("large_norms", np.random.default_rng(4))
    near_torch.nearest(s, device="cpu", block=500, rows=128, progress=3)
    out = capsys.readouterr().out
    assert "near_torch: bucket of 6000 training rows" in out
    assert "float32 pass, 500 test rows: training block 3/12 (1500 of 6000 rows)" in out
    assert "not certified by float32: float64 pass" in out and "bucket done in" in out
    near_torch.nearest(s, device="cpu", block=500, rows=128, progress=0)
    assert capsys.readouterr().out == ""
