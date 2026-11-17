"""The near search in parts (experiments.near_parts) gives the whole search's answers: per row, for
clean_split, and for the re-check of the cleaned split, which reads the stored distances."""
import numpy as np
import pandas as pd
import pytest

from experiments import near_parts
from leakcheck import Split
from leakcheck.checks import near
from leakcheck.clean import CleanOptions, clean_split, verify


def _split(seed=3) -> Split:
    rng = np.random.default_rng(seed)
    n_tr, n_te, d = 800, 300, 6
    x = rng.normal(size=(n_tr, d))
    te = x[rng.choice(n_tr, n_te)] + rng.normal(0, 1e-3, size=(n_te, d)) * (rng.random((n_te, 1)) < 0.5)
    te[:40] = te[40:80]                                    # repeated test rows: searched once
    cols = [f"f{i}" for i in range(d)]
    tr = pd.DataFrame(x, columns=cols).assign(c=rng.choice(["a", "b"], n_tr), y=rng.integers(0, 2, n_tr))
    ts = pd.DataFrame(te, columns=cols).assign(c=rng.choice(["a", "b", "z"], n_te), y=rng.integers(0, 2, n_te))
    return Split(tr, ts, "y")


def _parts(tmp_path, s, n):
    whole = near.nearest
    for i in range(n):
        f = near_parts.install(whole, tmp_path, (i, n))
        with pytest.raises(near_parts.PartDone):
            f(s)
    return near_parts.install(whole, tmp_path)


def test_parts_equal_the_whole_search(tmp_path):
    s = _split()
    d0, n0 = near.nearest(s)
    merged = _parts(tmp_path, s, 3)
    d1, n1 = merged(s)
    assert np.array_equal(d0, d1, equal_nan=True)
    fin = np.isfinite(d0)
    assert np.array_equal(np.isinf(d0), np.isinf(d1))
    # the nearest row may differ only between training rows at the same distance
    xtr, xte = near.embed(s)
    assert np.allclose(np.sqrt(((xte[fin] - xtr[n1[fin]]) ** 2).sum(1)), d0[fin], atol=1e-12)
    assert merged.stats["computed"] == 0 and merged.stats["unique_rows"] < len(s.test)
    sub = merged(s, np.array([5, 7, 41]))                 # a subset: read, not searched
    assert np.array_equal(sub[0], d0[[5, 7, 41]])


def test_clean_split_and_recheck_from_the_parts(tmp_path, monkeypatch):
    s = _split(4)
    opts = CleanOptions(near_eps=0.01, text_tau=None, conflicts="keep")
    want = clean_split(s, opts)
    merged = _parts(tmp_path, s, 4)
    monkeypatch.setattr(near, "nearest", merged)
    got = clean_split(s, opts)
    assert got.removed == want.removed and len(want.removed["near_tabular"]["test"]) > 0
    check = verify(Split(got.train, got.test, s.label), opts)
    assert check["L2.near"] == {"ok": True, "value": 0}
    assert merged.stats["computed"] == 0                  # the re-check read every distance


def test_a_missing_part_is_computed_not_invented(tmp_path):
    s = _split(5)
    whole = near.nearest
    f = near_parts.install(whole, tmp_path, (0, 2))
    with pytest.raises(near_parts.PartDone):
        f(s)
    merged = near_parts.install(whole, tmp_path)
    d, _ = merged(s)
    assert np.array_equal(d, whole(s)[0]) and merged.stats["computed"] > 0


def test_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LEAKCHECK_NEAR_CACHE", raising=False)
    assert near_parts.from_env(near.nearest) is near.nearest
    monkeypatch.setenv("LEAKCHECK_NEAR_CACHE", str(tmp_path))
    monkeypatch.setenv("LEAKCHECK_NEAR_PART", "2/2")
    with pytest.raises(ValueError):
        near_parts.from_env(near.nearest)


def test_parts_with_the_gpu_search(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from leakcheck.checks import near_torch
    s = _split(6)

    def gpu(split, test_idx=None):
        return near_torch.nearest(split, test_idx, device="cpu", block=200, rows=64)

    d0, _ = gpu(s)
    for i in range(3):
        with pytest.raises(near_parts.PartDone):
            near_parts.install(gpu, tmp_path, (i, 3))(s)
    d1, _ = near_parts.install(gpu, tmp_path)(s)
    assert np.array_equal(d0, d1)


def test_exact_copies_are_not_searched(tmp_path):
    s = _split(8)
    calls = []

    def counted(split, test_idx=None):
        calls.append(len(test_idx))
        return near.nearest(split, test_idx)

    f = near_parts.install(counted, tmp_path)
    d, nn = f(s)
    d0, _ = near.nearest(s)
    assert np.array_equal(d, d0)
    copies = f.stats["exact_copies"]
    assert copies > 0 and calls == [f.stats["unique_rows"] - copies]
    hit = np.isin(s.test_hash, s.train_hash)
    assert (d[hit] == 0).all() and np.array_equal(s.train_hash[nn[hit]], s.test_hash[hit])
