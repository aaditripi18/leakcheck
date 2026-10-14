"""L2 tabular near-duplicates: test rows that sit almost on top of a training row.

Space: each numeric feature goes through a signed log (network counters are
heavy-tailed), then is robust-scaled with the training median and IQR. The
distance is the RMS per-feature difference in that space, so "0.01" reads as
"on average, features differ by 1% of their interquartile range".
Non-numeric features must match exactly: neighbours are only searched within
rows that share all categorical values.

Nearest neighbours are exact (brute force), and the reported distance is
recomputed directly for the matched pair, so it is not affected by the
rounding of the fast squared-distance formula used for the search.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from ..findings import Finding, grade, wilson
from ..split import Split, canonicalize, row_hashes

EPS = 0.01
EPS_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
SAMPLE = 20_000
MISSING = 1e3  # NaN / inf sit this many IQRs away, so they only match each other
N_EXAMPLES = 5


def numeric_columns(split: Split) -> list[str]:
    return [
        c
        for c in split.features
        if c != split.text
        and pd.api.types.is_numeric_dtype(split.train[c])
        and pd.api.types.is_numeric_dtype(split.test[c])
    ]


def categorical_columns(split: Split) -> list[str]:
    num = set(numeric_columns(split))
    return [c for c in split.features if c not in num and c != split.text]


def _slog(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sign(x) * np.log1p(np.abs(x))


def embed(split: Split) -> tuple[np.ndarray, np.ndarray]:
    """Scaled numeric matrices (train, test), RMS-normalised."""
    cols = numeric_columns(split)
    tr = _slog(split.train[cols].to_numpy(dtype="float64", copy=True))
    te = _slog(split.test[cols].to_numpy(dtype="float64", copy=True))
    fin = np.where(np.isfinite(tr), tr, np.nan)
    with warnings.catch_warnings():  # all-NaN columns
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(fin, axis=0)
        iqr = np.nanpercentile(fin, 75, axis=0) - np.nanpercentile(fin, 25, axis=0)
        std = np.nanstd(fin, axis=0)
    scale = np.where(iqr > 0, iqr, np.where(std > 0, std, 1.0))
    med = np.nan_to_num(med)
    scale = np.nan_to_num(scale, nan=1.0)

    def z(x: np.ndarray) -> np.ndarray:
        out = (x - med) / scale
        out[np.isnan(x)] = -MISSING
        out[np.isposinf(x)] = 2 * MISSING
        out[np.isneginf(x)] = -2 * MISSING
        return out / np.sqrt(max(len(cols), 1))

    return z(tr), z(te)


def _cat_keys(split: Split) -> tuple[np.ndarray, np.ndarray]:
    cols = categorical_columns(split)
    both = pd.concat([split.train[cols], split.test[cols]], ignore_index=True)
    h = row_hashes(canonicalize(both))
    return h[: len(split.train)], h[len(split.train) :]


def nearest(split: Split, test_idx: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """For each chosen test row: RMS distance to, and index of, its nearest training row.

    Rows with no training row in the same categorical bucket get distance inf
    and index -1.
    """
    if test_idx is None:
        test_idx = np.arange(len(split.test))
    xtr, xte = embed(split)
    ktr, kte = _cat_keys(split)
    dist = np.full(len(test_idx), np.inf)
    nn = np.full(len(test_idx), -1, dtype=np.int64)
    if xtr.shape[1] == 0 or len(split.train) == 0:
        return dist, nn

    tr_groups = pd.Series(np.arange(len(ktr))).groupby(ktr).indices
    sub_keys = kte[test_idx]
    for key, pos in pd.Series(np.arange(len(sub_keys))).groupby(sub_keys).indices.items():
        members = tr_groups.get(key)
        if members is None:
            continue
        q = xte[test_idx[pos]]
        model = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(xtr[members])
        _, j = model.kneighbors(q)
        match = members[j[:, 0]]
        dist[pos] = np.sqrt(np.sum((q - xtr[match]) ** 2, axis=1))
        nn[pos] = match
    return dist, nn


def _sample(n: int, k: int, seed: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))


def check_near(split: Split, eps: float = EPS, sample: int = SAMPLE, seed: int = 0) -> Finding | None:
    if not numeric_columns(split):
        return None
    idx = _sample(len(split.test), sample, seed)
    dist, nn = nearest(split, idx)
    exact = np.isin(split.test_hash[idx], split.train_hash)
    near = (dist <= eps) & ~exact
    k, n = int(near.sum()), len(idx)
    rate = k / n if n else 0.0

    grid = sorted(set(EPS_GRID) | {eps})
    curve = {str(e): float(((dist <= e) & ~exact).mean()) if n else 0.0 for e in grid}
    baseline = _train_baseline(split, min(sample, len(split.train) // 2), seed, grid)

    ex = np.flatnonzero(near)[:N_EXAMPLES]
    return Finding(
        code="L2.near",
        severity=grade(rate, high=0.05, medium=0.01),
        value=rate,
        summary=f"{rate:.2%} of test rows are near-duplicates of a training row "
        f"(RMS scaled distance <= {eps}, not exact)",
        details={
            "eps": eps,
            "n_checked": n,
            "n_near": k,
            "sampled": n < len(split.test),
            "ci95": wilson(k, n),
            "exact_or_near_rate": float(((dist <= eps) | exact).mean()) if n else 0.0,
            "curve": curve,
            "train_baseline_curve": baseline,
            # > 1: test sits closer to training than training sits to itself.
            # Read the severity together with this: clustered data (many
            # near-identical flows) has a high baseline even under a random split.
            "vs_train_baseline": (rate / baseline[str(eps)]) if baseline.get(str(eps)) else None,
            "numeric_features": len(numeric_columns(split)),
            "categorical_features": len(categorical_columns(split)),
        },
        examples=[{"test": int(idx[i]), "train": int(nn[i]), "distance": float(dist[i])} for i in ex],
    )


def _train_baseline(split: Split, k: int, seed: int, grid=EPS_GRID) -> dict[str, float]:
    """The same curve for held-out training rows against the rest of training.

    Tells you what "near" looks like inside the training data itself, e.g. for
    benchmarks built from many repeated flows.
    """
    if k < 1:
        return {}
    rng = np.random.default_rng(seed + 7)
    perm = rng.permutation(len(split.train))
    held, rest = perm[:k], perm[k:]
    sub = Split(split.train.iloc[rest], split.train.iloc[held], split.label,
                time=split.time, group=split.group, ignore=list(split.ignore), text=split.text)
    dist, _ = nearest(sub)
    exact = np.isin(sub.test_hash, sub.train_hash)
    return {str(e): float(((dist <= e) & ~exact).mean()) for e in grid}
