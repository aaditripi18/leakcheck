"""Synthetic splits with exactly known leakage, for tests and experiment E0.

Base rows are continuous Gaussian draws, so they are unique with probability
one; every duplicate, conflict and shortcut in the output was put there on
purpose and is counted in `truth`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_split(
    n_train: int = 2000,
    n_test: int = 1000,
    n_features: int = 8,
    n_classes: int = 2,
    cross_dup: float = 0.0,
    train_dup: float = 0.0,
    conflict: float = 0.0,
    shortcut: bool = False,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (train, test, truth).

    cross_dup: fraction of test rows replaced by copies of training rows (with
      their training label).
    train_dup: fraction of training rows replaced by copies of other training rows.
    conflict: fraction of training rows that get a copy with a different label
      appended to the training set.
    shortcut: add a column 'leaky' that equals the label plus small noise.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(n_features, n_classes))

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, n_features))
        logits = x @ w + rng.gumbel(size=(n, n_classes)) * 2.0
        return x, logits.argmax(1)

    xtr, ytr = draw(n_train)
    xte, yte = draw(n_test)
    if shortcut:
        # Leaky column is part of the row, so copies below carry it along.
        xtr = np.column_stack([xtr, ytr + rng.normal(scale=0.05, size=n_train)])
        xte = np.column_stack([xte, yte + rng.normal(scale=0.05, size=n_test)])
    used = np.zeros(n_train, dtype=bool)  # rows touched by a copy, kept out of later pools

    k_tr = int(round(train_dup * n_train))
    if k_tr:
        dst = rng.choice(n_train, k_tr, replace=False)
        keep = np.setdiff1d(np.arange(n_train), dst)
        src = rng.choice(keep, k_tr, replace=True)
        xtr[dst], ytr[dst] = xtr[src], ytr[src]
        used[dst] = used[src] = True

    k_x = int(round(cross_dup * n_test))
    if k_x:
        dst = rng.choice(n_test, k_x, replace=False)
        src = rng.choice(n_train, k_x, replace=False)
        xte[dst], yte[dst] = xtr[src], ytr[src]
        used[src] = True

    k_c = int(round(conflict * n_train))
    if k_c:
        if n_classes < 2:
            raise ValueError("conflicts need at least two classes")
        # Only untouched rows, so every conflict group has exactly two rows.
        pool = np.flatnonzero(~used)
        src_c = rng.choice(pool, k_c, replace=False)
        flipped = (ytr[src_c] + rng.integers(1, n_classes, size=k_c)) % n_classes
        xtr = np.vstack([xtr, xtr[src_c]])
        ytr = np.concatenate([ytr, flipped])

    cols = [f"f{i}" for i in range(n_features)] + (["leaky"] if shortcut else [])
    train = pd.DataFrame(xtr, columns=cols)
    test = pd.DataFrame(xte, columns=cols)
    train["label"] = ytr
    test["label"] = yte

    truth = {
        "cross_dup_rows": k_x,
        "cross_dup_rate": k_x / n_test if n_test else 0.0,
        "train_dup_rows": k_tr,
        "conflict_pairs": k_c,
        "conflict_rows": 2 * k_c,
        "n_train": len(train),
        "n_test": len(test),
    }
    return train, test, truth
