"""Are a test set's leaked rows memorised or merely easy? (PLAN.md, phase 4: H19 and
`leakcheck explain`)

E13 answered this by training every model twice, with and without the leaked rows'
training copies. This module answers it without training anything: it removes the copies
(the same removal E9/E13 use) and asks whether each leaked row's nearest remaining
training row still carries the right label. If the leaked rows stay as predictable as the
rest of the test set, they are easy rows; if they only had the right neighbour because of
their copy, a model can only have got them right by memorising it.

The leak detection and copy removal moved here from experiments/e2_retrain.py and
experiments/e9_controls.py unchanged, so the experiments and the CLI share one code path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from leakcheck.checks import near, near_text
from leakcheck.split import Split

NEAR_EPS = 0.01
TEXT_TAU = 0.8
ROUNDS = 20
SAMPLE = 20_000

# H19's threshold on S, fixed on the eight eligible dev datasets (PLAN.md amendment
# 2026-11-09) by scripts/h19_threshold.py; None until that has run.
S_THRESHOLD: float | None = 0.07211209852651596
THRESHOLD_NOTE = ("The threshold was fixed on eight datasets whose answer is known (E13, 24 model fits each); "
                  "how often it is right on new data is H19 (PLAN.md, docs/RESULTS.md).")


def with_train(split: Split, train: pd.DataFrame, test: pd.DataFrame | None = None) -> Split:
    return Split(train, split.test if test is None else test, split.label, time=split.time, group=split.group,
                 ignore=list(split.ignore), text=split.text)


def leak_masks(split: Split) -> dict[str, np.ndarray]:
    """exact / near (exact or near) per test row, against the training data used."""
    exact = np.isin(split.test_hash, split.train_hash)
    near_any = exact.copy()
    todo = np.flatnonzero(~exact)
    if near.numeric_columns(split) and len(todo):
        dist, _ = near.nearest(split, todo)
        near_any[todo[dist <= NEAR_EPS]] = True
    if split.text is not None and len(todo):
        sim, _ = near_text.nearest_text(split.train[split.text], split.test[split.text], todo, TEXT_TAU)
        tr_norm = set(split.train[split.text].fillna("").map(near_text.normalize))
        same = split.test[split.text].fillna("").map(near_text.normalize).iloc[todo].isin(tr_norm).to_numpy()
        near_any[todo[(sim >= TEXT_TAU) | same]] = True
    return {"exact": exact, "near": near_any}


def _radius(split: Split, test_idx: np.ndarray, eps: float, keep: np.ndarray | None = None):
    """Radius search for the chosen test rows against the (kept) training rows, in the
    scaled space and categorical buckets of near.nearest, fitted on the *whole*
    training set: the space in which leak_masks called a test row leaked.
    Returns (training rows within eps of any test row, test rows with any such row)."""
    from sklearn.neighbors import NearestNeighbors

    xtr, xte = near.embed(split)
    ktr, kte = near._cat_keys(split)
    keep = np.ones(len(split.train), bool) if keep is None else keep
    train_hit = np.zeros(len(split.train), bool)
    test_hit = np.zeros(len(test_idx), bool)
    tr_groups = pd.Series(np.arange(len(ktr))).groupby(ktr).indices
    sub = kte[test_idx]
    for key, pos in pd.Series(np.arange(len(sub))).groupby(sub).indices.items():
        members = tr_groups.get(key)
        if members is None:
            continue
        members = members[keep[members]]
        if not len(members):
            continue
        nn = NearestNeighbors(radius=eps, algorithm="brute").fit(xtr[members])
        for c in range(0, len(pos), 2000):
            chunk = pos[c:c + 2000]
            for i, found in zip(chunk, nn.radius_neighbors(xte[test_idx[chunk]], return_distance=False)):
                if len(found):
                    train_hit[members[found]] = True
                    test_hit[i] = True
    return train_hit, test_hit


def copies_in_train(split: Split, leaked: np.ndarray) -> tuple[np.ndarray, int]:
    """Training rows to keep so that no leaked test row keeps an exact or near copy:
    exact copies by hash, every row within eps by radius search, text near copies by
    repeated MinHash nearest neighbour (at most ROUNDS rounds). Returns (keep mask,
    leaked test rows still with a copy afterwards: a check, expected 0)."""
    keep = ~np.isin(split.train_hash, split.test_hash[leaked])
    todo = np.flatnonzero(leaked)
    if near.numeric_columns(split) and len(todo):
        keep &= ~_radius(split, todo, NEAR_EPS)[0]
    if split.text is not None and len(todo):
        for _ in range(ROUNDS):
            sub = with_train(split, split.train[keep])
            pos = np.flatnonzero(keep)
            sim, nn = near_text.nearest_text(sub.train[split.text], sub.test[split.text], todo, TEXT_TAU)
            close = (sim >= TEXT_TAU) & (nn >= 0)
            if not close.any():
                break
            keep[pos[nn[close]]] = False
            todo = todo[close]
    # check, in the same space: no leaked row may keep an exact or numeric near copy
    left = np.isin(split.test_hash[leaked], split.train_hash[keep])
    if near.numeric_columns(split) and leaked.any():
        left |= _radius(split, np.flatnonzero(leaked), NEAR_EPS, keep)[1]
    return keep, int(left.sum())


def neighbour_votes(split: Split, test_idx: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """The label of each chosen test row's nearest (kept) training row: MinHash nearest
    neighbour when the split has a text column, otherwise leakcheck's scaled numeric space
    within categorical buckets; the training majority class where no neighbour is found."""
    sub = split if keep is None else with_train(split, split.train[keep])
    y_tr = sub.train[split.label].to_numpy()
    majority = pd.Series(y_tr).mode().iloc[0]
    if split.text is not None:
        _, nn = near_text.nearest_text(sub.train[split.text], sub.test[split.text], test_idx, 0.0)
    elif near.numeric_columns(split):
        _, nn = near.nearest(sub, test_idx)
    else:
        nn = np.full(len(test_idx), -1)
    votes = np.empty(len(test_idx), dtype=object)
    found = nn >= 0
    votes[found] = y_tr[nn[found]]
    votes[~found] = majority
    return votes


def statistic(split: Split, leaked: np.ndarray, keep: np.ndarray, sample: int = SAMPLE, seed: int = 0) -> dict:
    """S = neighbour-vote accuracy on the leaked rows (copies removed) minus on the rest
    (full training set), each on a random sample of at most `sample` rows."""
    rng = np.random.default_rng(seed)
    y = split.test[split.label].to_numpy()

    def pick(mask):
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, sample, replace=False)) if len(idx) > sample else idx

    li, ri = pick(leaked), pick(~leaked)
    # with every training row a copy of some leaked row, nothing is left to vote: S is undefined
    acc_l = float((neighbour_votes(split, li, keep) == y[li]).mean()) if len(li) and keep.any() else None
    acc_r = float((neighbour_votes(split, ri) == y[ri]).mean()) if len(ri) else None
    s = acc_l - acc_r if acc_l is not None and acc_r is not None else None
    return {"n_leaked": int(leaked.sum()), "n_rest": int((~leaked).sum()), "sampled_leaked": int(len(li)),
            "sampled_rest": int(len(ri)), "vote_acc_leaked": acc_l, "vote_acc_rest": acc_r, "S": s}


def explain(split: Split, sample: int = SAMPLE, seed: int = 0) -> dict:
    """Leak detection, copy removal and S for one train/test pair, with H19's verdict."""
    masks = leak_masks(split)
    leaked = masks["near"]
    out = {"n_test": int(len(split.test)), "exact": int(masks["exact"].sum()), "leaked": int(leaked.sum())}
    if not leaked.any():
        return {**out, "verdict": "no leaked rows"}
    keep, left = copies_in_train(split, leaked)
    st = statistic(split, leaked, keep, sample, seed)
    out.update(st, train_rows_removed=int((~keep).sum()), copies_left=left, threshold=S_THRESHOLD)
    if not keep.any():
        out["verdict"] = "no verdict (every training row is a copy of a leaked test row)"
    elif S_THRESHOLD is None or st["S"] is None:
        out["verdict"] = "no verdict (threshold not fixed)"
    else:
        out["verdict"] = "leaked rows look easy" if st["S"] > S_THRESHOLD else "leaked rows look memorised"
    return out
