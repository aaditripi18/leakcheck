"""Chunked row hashing gives exactly the unchunked hashes (needed for 47M-row data)."""
import numpy as np
import pandas as pd

from leakcheck import split as split_mod
from leakcheck.split import canonical_row_hashes, canonicalize, row_hashes


def test_chunked_hashes_equal_unchunked(monkeypatch):
    rng = np.random.default_rng(0)
    n = 1003
    df = pd.DataFrame({
        "f": rng.standard_normal(n),
        "i": pd.array(rng.integers(0, 5, n), dtype="Int64"),   # NaN only in the last rows below
        "b": rng.integers(0, 2, n).astype(bool),
        "s": rng.choice([" a", "b ", None], n),
        "f32": rng.standard_normal(n).astype("float32"),
    })
    df.loc[n - 3:, "i"] = pd.NA  # a chunk without NaN must still be treated like the column
    want = row_hashes(canonicalize(df))
    monkeypatch.setattr(split_mod, "HASH_CHUNK", 100)
    assert np.array_equal(canonical_row_hashes(df), want)
