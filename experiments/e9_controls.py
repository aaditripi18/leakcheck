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
from leakcheck.checks import near, near_text  # noqa: F401
from leakcheck.explain import ROUNDS, _radius, copies_in_train  # noqa: F401

SEEDS = (0, 1, 2)


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
