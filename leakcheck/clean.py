"""leakcheck clean: remove leakage, then prove it is gone by re-scanning.

Order matters for the guarantee:
  1. resolve label conflicts (may drop training rows),
  2. drop within-split duplicates (optional),
  3. drop test rows that leak from the now-final training set:
     exact copies, group overlap, tabular near-copies, text near-copies.
Training data does not change after step 2, so a re-scan of the output sees
exactly the training set the test rows were checked against.

Every removed row is logged by its 0-based row number in the input file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .checks import exact, group, near, near_text
from .split import Split

CONFLICT_MODES = ("drop", "majority", "keep")


@dataclass
class CleanOptions:
    near_eps: float | None = None  # drop tabular near-duplicates within this distance
    text_tau: float | None = None  # drop text near-duplicates at or above this Jaccard
    drop_group_overlap: bool = False
    dedup_within: bool = False
    conflicts: str = "drop"

    def __post_init__(self) -> None:
        if self.conflicts not in CONFLICT_MODES:
            raise ValueError(f"conflicts must be one of {CONFLICT_MODES}")


@dataclass
class CleanResult:
    train: pd.DataFrame
    test: pd.DataFrame
    removed: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    def counts(self) -> dict[str, dict[str, int]]:
        return {k: {s: len(v) for s, v in d.items()} for k, d in self.removed.items()}


def _like(split: Split, train: pd.DataFrame, test: pd.DataFrame) -> Split:
    return Split(train, test, split.label, time=split.time, group=split.group,
                 ignore=list(split.ignore), text=split.text)


def clean_split(split: Split, opts: CleanOptions) -> CleanResult:
    n_tr, n_te = len(split.train), len(split.test)
    tr_keep = np.ones(n_tr, dtype=bool)
    te_keep = np.ones(n_te, dtype=bool)
    removed: dict[str, dict[str, list[int]]] = {}

    def log(reason: str, tr_mask: np.ndarray | None = None, te_mask: np.ndarray | None = None) -> None:
        entry = removed.setdefault(reason, {"train": [], "test": []})
        if tr_mask is not None:
            new = tr_mask & tr_keep
            entry["train"] += np.flatnonzero(new).tolist()
            tr_keep[new] = False
        if te_mask is not None:
            new = te_mask & te_keep
            entry["test"] += np.flatnonzero(new).tolist()
            te_keep[new] = False

    # 1. label conflicts, over train + test together
    if opts.conflicts != "keep":
        h = np.concatenate([split.train_hash, split.test_hash])
        y = np.concatenate([split.y_train, split.y_test])
        drop = _conflict_drops(h, y, opts.conflicts)
        log(f"conflict_{opts.conflicts}", drop[:n_tr], drop[n_tr:])

    # 2. within-split duplicates: keep the first occurrence
    if opts.dedup_within:
        for name, hh, keep in (("train", split.train_hash, tr_keep), ("test", split.test_hash, te_keep)):
            dup = np.zeros(len(hh), dtype=bool)
            idx = np.flatnonzero(keep)
            dup[idx] = pd.Series(hh[idx]).duplicated().to_numpy()
            log("dedup_within", dup if name == "train" else None, dup if name == "test" else None)

    # 3. test rows that leak from the final training set
    final_train_hash = split.train_hash[tr_keep]
    log("cross_exact", te_mask=np.isin(split.test_hash, final_train_hash))

    if opts.drop_group_overlap and split.group is not None:
        gtr, gte = group.group_values(split)
        log("group_overlap", te_mask=np.isin(gte, gtr[tr_keep]))

    if opts.near_eps is not None or opts.text_tau is not None:
        sub = _like(split, split.train[tr_keep], split.test)
        if opts.near_eps is not None and near.numeric_columns(sub):
            dist, _ = near.nearest(sub)
            log("near_tabular", te_mask=dist <= opts.near_eps)
        if opts.text_tau is not None and split.text is not None:
            log("near_text", te_mask=_text_hits(sub, opts.text_tau))

    return CleanResult(
        train=split.train[tr_keep].reset_index(drop=True),
        test=split.test[te_keep].reset_index(drop=True),
        removed=removed,
    )


def _conflict_drops(h: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    df = pd.DataFrame({"h": h, "y": y})
    n_labels = df.groupby("h")["y"].transform("nunique").to_numpy()
    in_conflict = n_labels > 1
    if mode == "drop":
        return in_conflict
    # majority: keep rows carrying the group's most common label; drop the whole group on a tie
    counts = df.groupby(["h", "y"])["y"].transform("size").to_numpy()
    top = df.assign(c=counts).groupby("h")["c"].transform("max").to_numpy()
    n_top = (
        df.assign(c=counts, is_top=counts == top)
        .drop_duplicates(["h", "y"])
        .groupby("h")["is_top"]
        .sum()
    )
    tie = df["h"].map(n_top).to_numpy() > 1
    return in_conflict & ((counts < top) | tie)


def _text_hits(split: Split, tau: float) -> np.ndarray:
    sim, _ = near_text.nearest_text(split.train[split.text], split.test[split.text], tau=tau)
    tr_norm = set(split.train[split.text].fillna("").map(near_text.normalize))
    same = split.test[split.text].fillna("").map(near_text.normalize).isin(tr_norm).to_numpy()
    return (sim >= tau) | same


def verify(split: Split, opts: CleanOptions) -> dict[str, dict]:
    """Re-check a cleaned split on every row (no sampling). Each entry has ok + value."""
    out: dict[str, dict] = {}
    cross = exact.check_cross(split)
    out["L1.cross"] = {"ok": cross.details["n_hit"] == 0, "value": cross.details["n_hit"]}
    if opts.conflicts != "keep":
        c = {f.code: f for f in exact.check_conflicts(split)}["L3.conflict"]
        out["L3.conflict"] = {"ok": c.details["n_groups"] == 0, "value": c.details["n_groups"]}
    if opts.dedup_within:
        for f in exact.check_within(split):
            n_dup = f.details["n_rows"] - f.details["n_unique"]
            out[f.code] = {"ok": n_dup == 0, "value": n_dup}
    if opts.drop_group_overlap and split.group is not None:
        g = group.check_group(split)
        out["L6.group"] = {"ok": g.details["n_hit"] == 0, "value": g.details["n_hit"]}
    if opts.near_eps is not None and near.numeric_columns(split):
        dist, _ = near.nearest(split)
        n = int((dist <= opts.near_eps).sum())
        out["L2.near"] = {"ok": n == 0, "value": n}
    if opts.text_tau is not None and split.text is not None:
        n = int(_text_hits(split, opts.text_tau).sum())
        out["L2.near_text"] = {"ok": n == 0, "value": n}
    return out


def options_dict(opts: CleanOptions) -> dict:
    return asdict(opts)
