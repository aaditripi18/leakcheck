"""E11 (PLAN.md, phase 3b): EMBER on the official 2,381-dimension feature vectors.

The vectors are made by EMBER's own code (elastic/ember @ d97a0b5, AGPL), run
unmodified outside this repository (C:/Projects/_ember_official: vectorize.py, with a
documented scikit-learn compatibility shim) and verified there: EMBER's published
2018 LightGBM model scores ROC AUC 0.996429 on them, the figure EMBER's own notebook
reports (0.9964289468). This module only reads the resulting files (raw float32
memmaps and metadata CSVs); no EMBER code is imported or copied.

Rows are joined to leakcheck's EMBER loader by SHA-256, so the official temporal split
(E1) and E2's random split select exactly the same files as for leakcheck's own view;
only the features differ, which is what H15 compares.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import data

VEC_DIR = Path(os.environ.get("LEAKCHECK_EMBER_VECTORS", data.ROOT.parent / "_ember_official" / "data" / "ember2018"))
DIM = 2381
FILES = {"X_train.dat": (800_000, DIM), "X_test.dat": (200_000, DIM)}
# SHA-256 of the files E11 used (recorded 2026-11-04, after the AUC check)
SHA256 = {
    "X_train.dat": "50a9e2541e42cca90433be97acc03303038c4aebdbc110c86935d5f6edc45d28",
    "X_test.dat": "9bf3c905240c749c40e8e7461e9c38f4be4b8f0440f0aea4d56d259e679d8a18",
    "train_metadata.csv": "3da0d30135a1305669f4a3a2091310fad25206c7dab78c9439a6be28fbb1c520",
    "test_metadata.csv": "9ff00493351e05b2c27332cde97a39bed601330c003f925769a929c90491d589",
}


def verify() -> None:
    for name, digest in SHA256.items():
        got = data.sha256_file(VEC_DIR / name)
        if got != digest:
            raise data.HashMismatch(f"{VEC_DIR / name}: expected {digest}, got {got}")


def load_vectors() -> pd.DataFrame:
    """Labelled samples: 2,381 float32 feature columns plus sha256 and label, train then test."""
    verify()
    frames = []
    for part, rows in (("train", 800_000), ("test", 200_000)):
        X = np.memmap(VEC_DIR / f"X_{part}.dat", dtype=np.float32, mode="r", shape=(rows, DIM))
        meta = pd.read_csv(VEC_DIR / f"{part}_metadata.csv", usecols=["sha256", "label"])
        keep = (meta["label"] != -1).to_numpy()
        f = pd.DataFrame(np.asarray(X[keep]), columns=[f"v{i}" for i in range(DIM)])
        f["sha256"] = meta.loc[keep, "sha256"].to_numpy()
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def aligned(base: pd.DataFrame) -> pd.DataFrame:
    """The official vectors in the row order of `base` (leakcheck's labelled EMBER frame),
    with base's split, label, appeared and family columns. Every base row must match."""
    vec = load_vectors().set_index("sha256")
    missing = ~base["sha256"].isin(vec.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} labelled EMBER files have no official vector")
    out = vec.loc[base["sha256"].to_numpy()].reset_index(drop=True)
    for c in ("split", "label", "appeared", "family"):
        if c in base:
            out[c] = base[c].to_numpy()
    return out
