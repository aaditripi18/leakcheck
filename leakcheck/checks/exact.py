"""L1 exact duplicates and L3 label conflicts, both from canonical row hashes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..findings import Finding, grade
from ..split import Split

N_EXAMPLES = 5


def dup_rate(h: np.ndarray) -> float:
    """Fraction of rows that repeat an earlier row."""
    return 0.0 if len(h) == 0 else 1.0 - len(np.unique(h)) / len(h)


def check_within(split: Split) -> list[Finding]:
    out = []
    for name, h in (("train", split.train_hash), ("test", split.test_hash)):
        r = dup_rate(h)
        out.append(
            Finding(
                code=f"L1.within_{name}",
                severity="info" if r > 0 else "ok",
                value=r,
                summary=f"{r:.2%} of {name} rows repeat another {name} row",
                details={"n_rows": len(h), "n_unique": int(len(np.unique(h)))},
            )
        )
    return out


def check_cross(split: Split) -> Finding:
    """Test rows whose features exactly match some training row."""
    tr, te = split.train_hash, split.test_hash
    hit = np.isin(te, tr)
    rate = float(hit.mean()) if len(te) else 0.0

    uniq = np.unique(te)
    uniq_rate = float(np.isin(uniq, tr).mean()) if len(uniq) else 0.0

    y = split.y_test
    per_class = {
        str(c): {"n": int((y == c).sum()), "rate": float(hit[y == c].mean())}
        for c in pd.unique(y)
    }

    first_train = pd.Series(np.arange(len(tr))).groupby(tr).first()
    idx = np.flatnonzero(hit)[:N_EXAMPLES]
    examples = [{"test": int(i), "train": int(first_train[te[i]])} for i in idx]

    return Finding(
        code="L1.cross",
        severity=grade(rate, high=0.05, medium=0.01),
        value=rate,
        summary=f"{rate:.2%} of test rows ({int(hit.sum())}) exactly duplicate a training row",
        details={"n_hit": int(hit.sum()), "unique_test_rate": uniq_rate, "per_class": per_class},
        examples=examples,
    )


def check_conflicts(split: Split) -> list[Finding]:
    """Identical features with different labels.

    L3.conflict: rows (train+test) in a feature group with more than one label.
    L3.ceiling: the best accuracy any classifier can reach on the test set,
      since rows with identical features must get the same prediction.
    L3.cross_contradict: test rows that match training rows, none of which
      carry the test row's label (memorisation is guaranteed to be wrong).
    """
    h = np.concatenate([split.train_hash, split.test_hash])
    y = np.concatenate([split.y_train, split.y_test])
    df = pd.DataFrame({"h": h, "y": y})
    n_labels = df.groupby("h")["y"].nunique()
    bad = n_labels.index[n_labels > 1]
    in_bad = df["h"].isin(bad).to_numpy()
    rate = float(in_bad.mean()) if len(df) else 0.0
    examples = [int(i) for i in np.flatnonzero(in_bad[: len(split.train_hash)])[:N_EXAMPLES]]

    te = pd.DataFrame({"h": split.test_hash, "y": split.y_test})
    if len(te):
        best = te.groupby(["h", "y"]).size().groupby(level="h").max().sum()
        ceiling = float(best / len(te))
    else:
        ceiling = 1.0

    tr = pd.DataFrame({"h": split.train_hash, "y": split.y_train})
    same_pair = pd.MultiIndex.from_frame(te).isin(pd.MultiIndex.from_frame(tr))
    contra = np.isin(te["h"].to_numpy(), tr["h"].to_numpy()) & ~same_pair
    contra_rate = float(contra.mean()) if len(te) else 0.0

    return [
        Finding(
            code="L3.conflict",
            severity=grade(rate, high=0.05, medium=0.01),
            value=rate,
            summary=f"{rate:.2%} of rows share identical features with a differently labelled row ({len(bad)} groups)",
            details={"n_groups": int(len(bad)), "n_rows": int(in_bad.sum())},
            examples=examples,
        ),
        Finding(
            code="L3.ceiling",
            severity="medium" if ceiling < 0.99 else ("low" if ceiling < 1.0 else "ok"),
            value=ceiling,
            summary=f"no classifier on these features can exceed {ceiling:.2%} test accuracy",
        ),
        Finding(
            code="L3.cross_contradict",
            severity=grade(contra_rate, high=0.01, medium=0.001),
            value=contra_rate,
            summary=f"{contra_rate:.2%} of test rows match training rows that all carry a different label",
            details={"n_rows": int(contra.sum())},
            examples=[int(i) for i in np.flatnonzero(contra)[:N_EXAMPLES]],
        ),
    ]
