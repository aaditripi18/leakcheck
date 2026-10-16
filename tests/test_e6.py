"""E6's count-based macro-F1 and bootstrap cells agree with scikit-learn."""
import numpy as np
from sklearn.metrics import f1_score

from experiments.e6_analysis import Cells, gaps, macro_f1_from_counts


def test_macro_f1_matches_sklearn_including_absent_classes():
    rng = np.random.default_rng(0)
    for k in (2, 3, 7):
        y = rng.integers(0, k, 500)
        p = np.where(rng.random(500) < 0.7, y, rng.integers(0, k + 1, 500))  # includes a class absent from y
        labels = np.unique(np.concatenate([y, p]))
        cm = np.zeros((len(labels),) * 2)
        np.add.at(cm, (np.searchsorted(labels, y), np.searchsorted(labels, p)), 1)
        assert abs(macro_f1_from_counts(cm) - f1_score(y, p, average="macro", zero_division=0)) < 1e-12


def test_gap_and_paired_cells_match_direct_computation():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 2000)
    a = np.where(rng.random(2000) < 0.8, y, rng.integers(0, 3, 2000))
    b = np.where(rng.random(2000) < 0.6, y, rng.integers(0, 3, 2000))
    leaked = rng.random(2000) < 0.3
    direct = lambda p: f1_score(y, p, average="macro") - f1_score(y[~leaked], p[~leaked], average="macro")  # noqa: E731
    one = Cells(y, [a], leaked)
    assert abs(gaps(one.counts.reshape(one.shape), 1, 1) - direct(a)) < 1e-12
    two = Cells(y, [a, b], leaked)
    full = two.counts.reshape(two.shape)
    assert abs(gaps(full, 1, 2) - direct(a)) < 1e-12 and abs(gaps(full, 2, 2) - direct(b)) < 1e-12
    draws = two.resample(np.random.default_rng(2), 50)
    assert draws.shape == (50, *two.shape) and np.all(draws.sum(axis=tuple(range(1, draws.ndim))) == 2000)
