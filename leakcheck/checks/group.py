"""L6 group overlap: the same host / family / session / user in both splits."""

from __future__ import annotations

import numpy as np

from ..findings import Finding, grade
from ..split import Split

N_EXAMPLES = 5


def group_values(split: Split) -> tuple[np.ndarray, np.ndarray]:
    g = split.group
    return (
        split.train[g].astype("string").str.strip().fillna("<NA>").to_numpy(),
        split.test[g].astype("string").str.strip().fillna("<NA>").to_numpy(),
    )


def check_group(split: Split) -> Finding | None:
    if split.group is None:
        return None
    gtr, gte = group_values(split)
    hit = np.isin(gte, gtr)
    rate = float(hit.mean()) if len(gte) else 0.0
    shared = np.intersect1d(np.unique(gtr), np.unique(gte))
    n_test_groups = len(np.unique(gte))
    return Finding(
        code="L6.group",
        severity=grade(rate, high=0.05, medium=0.01),
        value=rate,
        summary=f"{rate:.2%} of test rows belong to a {split.group!r} value also present in training "
        f"({len(shared)} of {n_test_groups} test groups)",
        details={"n_hit": int(hit.sum()), "shared_groups": int(len(shared)), "test_groups": n_test_groups},
        examples=[str(x) for x in shared[:N_EXAMPLES]],
    )
