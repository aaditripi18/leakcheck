"""L5 shortcuts: single features that give the label away, and identifier-like columns."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from ..findings import Finding
from ..split import Split

MAX_ROWS = 200_000  # per split, for speed on multi-million-row IDS datasets
MAX_LEAVES = 32
HIGH, MEDIUM = 0.95, 0.90
TOP_K = 10

ID_NAME = re.compile(
    r"(^|[\s_.-])(id|uid|uuid|ip|addr|address|port|mac|timestamp|time|date|hash|md5|sha\d*)($|[\s_.-])",
    re.IGNORECASE,
)
# A name match only counts when the column could actually identify something:
# binary/ternary flags such as "having_IP_Address" are features, not identifiers.
ID_MIN_UNIQUE = 50


def _sample(n: int, seed: int) -> np.ndarray:
    if n <= MAX_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS, replace=False))


def single_feature_score(xtr: pd.Series, ytr: np.ndarray, xte: pd.Series, yte: np.ndarray) -> float:
    """Balanced test accuracy of the best simple classifier that sees one feature.

    Numeric: a small decision tree (handles thresholds and NaN).
    Other: majority training label per category, unseen categories get the
    overall majority label.
    """
    if pd.api.types.is_numeric_dtype(xtr) and pd.api.types.is_numeric_dtype(xte):
        a = xtr.to_numpy(dtype="float64", copy=True).reshape(-1, 1)
        b = xte.to_numpy(dtype="float64", copy=True).reshape(-1, 1)
        a[np.isinf(a)] = np.nan
        b[np.isinf(b)] = np.nan
        tree = DecisionTreeClassifier(max_leaf_nodes=MAX_LEAVES, random_state=0).fit(a, ytr)
        pred = tree.predict(b)
    else:
        a = xtr.astype("string").fillna("<NA>").to_numpy()
        b = xte.astype("string").fillna("<NA>").to_numpy()
        df = pd.DataFrame({"x": a, "y": ytr})
        majority = df.groupby("x")["y"].agg(lambda s: s.value_counts().index[0])
        fallback = pd.Series(ytr).value_counts().index[0]
        pred = pd.Series(b).map(majority).fillna(fallback).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(yte, pred))


def check_shortcuts(split: Split, seed: int = 0) -> Finding:
    itr, ite = _sample(len(split.train), seed), _sample(len(split.test), seed + 1)
    ytr, yte = split.y_train[itr], split.y_test[ite]
    scores = {}
    for col in split.features:
        xtr = split.train[col].iloc[itr]
        xte = split.test[col].iloc[ite]
        scores[col] = single_feature_score(xtr, ytr, xte, yte)

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_col, best = ranked[0] if ranked else (None, 0.0)
    n_classes = len(pd.unique(ytr))
    flagged = [c for c, s in ranked if s >= HIGH]
    sev = "high" if best >= HIGH else "medium" if best >= MEDIUM else "ok"
    return Finding(
        code="L5.shortcut",
        severity=sev,
        value=best,
        summary=f"best single feature {best_col!r} reaches {best:.2%} balanced test accuracy "
        f"(chance {1 / max(n_classes, 1):.2%}); {len(flagged)} feature(s) >= {HIGH:.0%}",
        details={
            "top": [{"feature": c, "balanced_acc": s} for c, s in ranked[:TOP_K]],
            "flagged": flagged,
            "n_classes": n_classes,
            "rows_used": {"train": len(itr), "test": len(ite)},
        },
    )


def check_identifiers(split: Split) -> Finding:
    """Columns that look like identifiers or capture context, not behaviour."""
    hits = []
    n = len(split.train)
    for col in split.features:
        if col == split.text:  # free text is high-cardinality by nature
            continue
        s = split.train[col]
        reasons = []
        if ID_NAME.search(str(col)) and s.nunique(dropna=True) >= ID_MIN_UNIQUE:
            reasons.append("name")
        if not pd.api.types.is_numeric_dtype(s) and n and s.nunique(dropna=True) / n > 0.5:
            reasons.append("high-cardinality text")
        if reasons:
            hits.append({"feature": col, "reasons": reasons})
    return Finding(
        code="L5.identifier",
        severity="medium" if hits else "ok",
        value=float(len(hits)),
        summary=f"{len(hits)} feature(s) look like identifiers or context (IPs, ports, IDs, timestamps)",
        details={"columns": hits},
    )
