"""L4 temporal leakage: training data from after the test period.

Follows the first TESSERACT constraint (Pendlebury et al. 2019): every training
sample should precede every test sample. Also scores the timestamp as a
one-feature classifier, because when benign and malicious samples were
collected in different periods, time alone separates them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..findings import Finding, grade
from ..split import Split
from .shortcut import HIGH, MEDIUM, single_feature_score


def to_seconds(s: pd.Series) -> tuple[np.ndarray, bool]:
    """Times as float seconds (NaN where unparseable), and whether they were dates."""
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype="float64", copy=True), False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    secs = (t - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return secs.to_numpy(dtype="float64", copy=True), True


def _fmt(x: float, is_date: bool) -> str | float:
    if not np.isfinite(x):
        return "n/a"
    return str(pd.Timestamp(x, unit="s", tz="UTC")) if is_date else float(x)


def check_temporal(split: Split) -> list[Finding]:
    if split.time is None:
        return []
    ttr, d1 = to_seconds(split.train[split.time])
    tte, d2 = to_seconds(split.test[split.time])
    is_date = d1 or d2
    ok_tr, ok_te = np.isfinite(ttr), np.isfinite(tte)
    if not ok_tr.any() or not ok_te.any():
        return [Finding("L4.after_test_start", "info", 0.0, f"time column {split.time!r} could not be parsed",
                        details={"unparsed_train": int((~ok_tr).sum()), "unparsed_test": int((~ok_te).sum())})]
    a, b = ttr[ok_tr], tte[ok_te]
    t0, tmed, t1 = b.min(), np.median(b), b.max()
    after = float((a > t0).mean())

    per_class = {}
    labels = np.concatenate([split.y_train[ok_tr], split.y_test[ok_te]])
    times = np.concatenate([a, b])
    for c in pd.unique(labels):
        tc = times[labels == c]
        per_class[str(c)] = {"min": _fmt(tc.min(), is_date), "max": _fmt(tc.max(), is_date), "n": int(len(tc))}

    findings = [
        Finding(
            code="L4.after_test_start",
            severity=grade(after, high=0.10, medium=0.01),
            value=after,
            summary=f"{after:.2%} of training rows are newer than the earliest test row "
            f"({(a > tmed).mean():.2%} newer than the median, {(a > t1).mean():.2%} newer than all test rows)",
            details={
                "train_range": [_fmt(a.min(), is_date), _fmt(a.max(), is_date)],
                "test_range": [_fmt(t0, is_date), _fmt(t1, is_date)],
                "after_test_median": float((a > tmed).mean()),
                "after_test_end": float((a > t1).mean()),
                "unparsed_train": int((~ok_tr).sum()),
                "unparsed_test": int((~ok_te).sum()),
                "per_class": per_class,
            },
        )
    ]

    score = single_feature_score(
        pd.Series(ttr), split.y_train, pd.Series(tte), split.y_test
    )
    findings.append(
        Finding(
            code="L4.time_shortcut",
            severity="high" if score >= HIGH else "medium" if score >= MEDIUM else "ok",
            value=score,
            summary=f"the timestamp alone reaches {score:.2%} balanced test accuracy",
        )
    )
    return findings
