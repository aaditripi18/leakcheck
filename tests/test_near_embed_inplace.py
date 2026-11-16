"""near.embed transforms its matrices in place, a block of rows at a time (memory: CIC-IoT2023).
It must give exactly what the whole-matrix formula gives, including the column-major layout
pandas returns, which decides the summation order of the scaling statistics."""
import warnings

import numpy as np
import pandas as pd
import pytest

from leakcheck import Split
from leakcheck.checks import near


def _reference(split, dtype):
    cols = near.numeric_columns(split)
    tr = near._slog(split.train[cols].to_numpy(dtype=dtype, copy=True))
    te = near._slog(split.test[cols].to_numpy(dtype=dtype, copy=True))
    if dtype == "float32":
        med, iqr, std = near._column_stats(tr)
    else:
        fin = np.where(np.isfinite(tr), tr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(fin, axis=0)
            iqr = np.nanpercentile(fin, 75, axis=0) - np.nanpercentile(fin, 25, axis=0)
            std = np.nanstd(fin, axis=0)
    scale = np.nan_to_num(np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0)), nan=1.0)
    med = np.nan_to_num(med)

    def z(x):
        out = (x - med.astype(x.dtype)) / scale.astype(x.dtype)
        out[np.isnan(x)] = -near.MISSING
        out[np.isposinf(x)] = 2 * near.MISSING
        out[np.isneginf(x)] = -2 * near.MISSING
        out /= np.sqrt(max(len(cols), 1))
        return out
    return z(tr), z(te)


@pytest.mark.parametrize("big", [False, True])
def test_in_place_embedding_equals_whole_matrix_formula(big, monkeypatch):
    rng = np.random.default_rng(1)
    n = 5000
    df = pd.DataFrame({"a": rng.normal(0, 1e4, n), "b": rng.integers(0, 70000, n), "c": rng.exponential(3, n),
                       "d": rng.choice([0.0, 1.0, np.nan, np.inf, -np.inf], n), "y": rng.integers(0, 2, n)})
    s = Split(df.iloc[:4000].reset_index(drop=True), df.iloc[4000:].reset_index(drop=True), "y")
    monkeypatch.setattr(near, "CHUNK", 333)
    monkeypatch.setattr(near, "LARGE", 1 if big else 10**12)
    got = near.embed(s)
    want = _reference(s, "float32" if big else "float64")
    for g, w in zip(got, want):
        assert g.dtype == w.dtype and g.flags.f_contiguous == w.flags.f_contiguous
        assert np.array_equal(g, w, equal_nan=True)
