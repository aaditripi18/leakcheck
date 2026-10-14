"""A train/test split plus the canonical row hashing every check relies on."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

# Floats are rounded to this many mantissa bits before hashing, so values that
# differ only by float-printing noise (0.1 + 0.2 vs 0.3) hash equal. 30 bits is
# about 9 significant decimal digits.
MANTISSA_BITS = 30


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, skipinitialspace=True)


def canonicalize(df: pd.DataFrame, mantissa_bits: int = MANTISSA_BITS) -> pd.DataFrame:
    """Return a copy whose equal-looking rows are bit-identical."""
    out = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            out[col] = s.astype("int64")
        elif pd.api.types.is_integer_dtype(s) and not s.isna().any():
            out[col] = s.astype("int64")
        elif pd.api.types.is_numeric_dtype(s):
            out[col] = pd.Series(_round_mantissa(s.to_numpy(dtype="float64"), mantissa_bits), index=s.index)
        else:
            # Strings / mixed: strip whitespace, unify missing values.
            t = s.astype("string").str.strip()
            out[col] = t.fillna("<NA>")
    return pd.DataFrame(out, index=df.index)


def _round_mantissa(x: np.ndarray, bits: int) -> np.ndarray:
    x = x.copy()
    finite = np.isfinite(x)
    m, e = np.frexp(x[finite])
    scale = float(2**bits)
    x[finite] = np.ldexp(np.round(m * scale) / scale, e)
    x[x == 0] = 0.0  # -0.0 -> 0.0
    x[np.isnan(x)] = np.nan  # one canonical NaN
    return x


def row_hashes(df: pd.DataFrame) -> np.ndarray:
    """64-bit hash per row of an already-canonicalized frame (column order matters)."""
    if df.shape[1] == 0:
        return np.zeros(len(df), dtype=np.uint64)
    return pd.util.hash_pandas_object(df, index=False).to_numpy()


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    label: str
    time: str | None = None
    group: str | None = None
    ignore: list[str] = field(default_factory=list)
    text: str | None = None  # a free-text feature (URL, message) for MinHash near-duplicates

    def __post_init__(self) -> None:
        for name, df in (("train", self.train), ("test", self.test)):
            if self.label not in df.columns:
                raise ValueError(f"label column {self.label!r} missing from {name}")
        missing = set(self.train.columns) ^ set(self.test.columns)
        if missing:
            raise ValueError(f"train and test have different columns: {sorted(missing)}")
        for c in [self.time, self.group, self.text, *self.ignore]:
            if c is not None and c not in self.train.columns:
                raise ValueError(f"column {c!r} not found")
        if self.text is not None and self.text in {self.label, self.time, self.group, *self.ignore}:
            raise ValueError(f"text column {self.text!r} must be a feature")
        self.train = self.train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)

    @classmethod
    def from_files(cls, train: str | Path, test: str | Path, label: str, **kw) -> "Split":
        return cls(read_table(train), read_table(test), label, **kw)

    @property
    def features(self) -> list[str]:
        """Feature columns: everything except label, time, group and ignored.

        Ordered by the train frame so train and test hash identically even if
        their CSV column orders differ.
        """
        drop = {self.label, self.time, self.group, *self.ignore}
        return [c for c in self.train.columns if c not in drop]

    @cached_property
    def _hashes(self) -> np.ndarray:
        # Canonicalize train and test together so a column that is int in one
        # file and float in the other (e.g. because of a NaN) hashes the same.
        both = pd.concat([self.train[self.features], self.test[self.features]], ignore_index=True)
        return row_hashes(canonicalize(both))

    @property
    def train_hash(self) -> np.ndarray:
        return self._hashes[: len(self.train)]

    @property
    def test_hash(self) -> np.ndarray:
        return self._hashes[len(self.train) :]

    @property
    def y_train(self) -> np.ndarray:
        return self.train[self.label].astype("string").to_numpy()

    @property
    def y_test(self) -> np.ndarray:
        return self.test[self.label].astype("string").to_numpy()
