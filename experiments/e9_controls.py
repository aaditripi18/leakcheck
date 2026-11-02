"""E9 + E10 (PLAN.md, phase 3b): are the copies memorised, and is the gap more
than the effect of removing rows?

For each dataset, seed (0-2) and E2 baseline model, on E2's split and training cap:

  E10  the full model's macro-F1 on the whole test set, on the test set without
       its leaked rows (E2's no_near), and on the test set without a *random*
       subset of the same size and class mix (seed-matched): the leakage gap
       next to a random-removal control gap.
  E9   a second model trained without every training row that copies a leaked
       test row (exact rows by hash; near rows by repeated nearest-neighbour
       removal until no leaked test row has a training row within eps, at most
       ROUNDS rounds); on the leaked rows, a row is *memorised* if the full
       model gets it right and the deduplicated model does not.

    python -m experiments.e9_controls kdd99 nslkdd ...
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments import data
from experiments.e2_retrain import (MODELS, NEAR_EPS, TEXT_TAU, fit_predict, leak_masks, like, original_split,
                                    scores, subsample)
from experiments.e1_audit import PREP
from experiments.ledger import record
from leakcheck import Split
from leakcheck.checks import near, near_text

SEEDS = (0, 1, 2)
ROUNDS = 20


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
            sub = like(split, split.train[keep], split.test)
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


def run_one(name: str, seed: int) -> dict:
    split = subsample(original_split(name, seed), seed)
    masks = leak_masks(split)
    leaked = masks["near"]
    y = split.y_test
    rng = np.random.default_rng(seed)
    # random rows of the same size and class mix as the leaked ones
    rand = np.zeros(len(y), bool)
    for c in np.unique(y[leaked]):
        pool = np.flatnonzero(y == c)
        rand[rng.choice(pool, int((y[leaked] == c).sum()), replace=False)] = True
    keep, left = copies_in_train(split, leaked) if leaked.any() else (np.ones(len(split.train), bool), 0)
    dedup = like(split, split.train[keep], split.test)
    out = {"dataset": name, "seed": seed, "n_test": int(len(y)), "n_leaked": int(leaked.sum()),
           "train_rows_removed": int((~keep).sum()), "leaked_rows_still_near": int(left), "models": {}}
    for m in MODELS:
        pred, _ = fit_predict(split, m, seed)
        r = {"original": scores(y, pred), "no_near": scores(y[~leaked], pred[~leaked]),
             "random_removed": scores(y[~rand], pred[~rand])}
        if leaked.any():
            pred_d, _ = fit_predict(dedup, m, seed)
            ok_full, ok_dedup = pred[leaked] == y[leaked], pred_d[leaked] == y[leaked]
            r["memorised_share"] = float((ok_full & ~ok_dedup).mean())
            r["correct_either_way_share"] = float((ok_full & ok_dedup).mean())
            r["leaked_acc_full"], r["leaked_acc_dedup"] = float(ok_full.mean()), float(ok_dedup.mean())
        out["models"][m] = r
        f = lambda s: r[s]["macro_f1"] if r[s]["macro_f1"] is not None else float("nan")  # noqa: E731
        print(f"   {name} seed {seed} {m}: gap {100 * (f('original') - f('no_near')):+.2f} pt, "
              f"random {100 * (f('original') - f('random_removed')):+.2f} pt, "
              f"memorised {r.get('memorised_share', float('nan')):.3f}", flush=True)
    return out


def summarize(per_seed: list[dict]) -> dict:
    out = {}
    for m in MODELS:
        rows = [r["models"][m] for r in per_seed]
        g = [r["original"]["macro_f1"] - r["no_near"]["macro_f1"] for r in rows
             if r["no_near"]["macro_f1"] is not None]
        c = [r["original"]["macro_f1"] - r["random_removed"]["macro_f1"] for r in rows]
        out[m] = {"leakage_gap": float(np.mean(g)) if g else None, "random_gap": float(np.mean(c)),
                  "memorised_share": float(np.mean([r["memorised_share"] for r in rows]))
                  if "memorised_share" in rows[0] else None,
                  "correct_either_way_share": float(np.mean([r["correct_either_way_share"] for r in rows]))
                  if "correct_either_way_share" in rows[0] else None}
    return out


def main(names: list[str]) -> None:
    for name in names:
        print(f"== {name}", flush=True)
        per_seed = [run_one(name, s) for s in SEEDS]
        prep = PREP[name.partition(":")[0]]()
        record("e9_controls:" + name, {"seeds": list(SEEDS), "near_eps": NEAR_EPS, "text_tau": TEXT_TAU,
                                        "rounds": ROUNDS, "models": list(MODELS)},
               {"per_seed": per_seed, "summary": summarize(per_seed)},
               {d: data.dataset_hashes(d) for d in prep.datasets})


if __name__ == "__main__":
    main(sys.argv[1:])
