"""E13 (PLAN.md, amendment 2026-11-05): are duplicated test rows simply easy?

Same splits, seeds, models and copy removal as E9. For the full model and the copy-free
model (trained without every training row within eps of a leaked test row), records:
macro-F1 on the whole test set and without the leaked rows, and accuracy on the leaked
rows and on the rest. H17 compares the two models' leakage gaps; H18 compares the
copy-free model's accuracy on leaked and remaining rows.

    python -m experiments.e13_difficulty kdd99 nslkdd ...
"""
from __future__ import annotations

import sys

import numpy as np

from experiments import data
from experiments.e1_audit import PREP
from experiments.e2_retrain import MODELS, NEAR_EPS, TEXT_TAU, fit_predict, leak_masks, like, original_split, \
    scores, subsample
from experiments.e9_controls import ROUNDS, SEEDS, copies_in_train
from experiments.ledger import record


def _acc(y, pred, mask) -> float | None:
    return float((pred[mask] == y[mask]).mean()) if mask.any() else None


def run_one(name: str, seed: int) -> dict:
    split = subsample(original_split(name, seed), seed)
    leaked = leak_masks(split)["near"]
    y = split.y_test
    out = {"dataset": name, "seed": seed, "n_test": int(len(y)), "n_leaked": int(leaked.sum()),
           "leaked_share": float(leaked.mean()), "models": {}}
    if not leaked.any():
        return out
    keep, left = copies_in_train(split, leaked)
    out["train_rows_removed"], out["leaked_rows_still_near"] = int((~keep).sum()), int(left)
    dedup = like(split, split.train[keep], split.test)
    for m in MODELS:
        r = {}
        for tag, s in (("full", split), ("copy_free", dedup)):
            pred, _ = fit_predict(s, m, seed)
            r[tag] = {"whole": scores(y, pred)["macro_f1"], "without_leaked": scores(y[~leaked], pred[~leaked])["macro_f1"],
                      "acc_leaked": _acc(y, pred, leaked), "acc_rest": _acc(y, pred, ~leaked)}
            w, wo = r[tag]["whole"], r[tag]["without_leaked"]
            r[tag]["gap"] = (w - wo) if w is not None and wo is not None else None
        out["models"][m] = r
        f = lambda v: f"{v:+.4f}" if v is not None else "  n/a"  # noqa: E731
        print(f"   {name} seed {seed} {m}: gap full {f(r['full']['gap'])} copy-free {f(r['copy_free']['gap'])}; "
              f"copy-free acc leaked {r['copy_free']['acc_leaked']:.4f} rest {r['copy_free']['acc_rest']:.4f}",
              flush=True)
    return out


def main(names: list[str]) -> None:
    for name in names:
        print(f"== {name}", flush=True)
        per_seed = [run_one(name, s) for s in SEEDS]
        prep = PREP[name.partition(":")[0]]()
        record("e13_difficulty:" + name, {"seeds": list(SEEDS), "near_eps": NEAR_EPS, "text_tau": TEXT_TAU,
                                           "rounds": ROUNDS, "models": list(MODELS)},
               {"per_seed": per_seed}, {d: data.dataset_hashes(d) for d in prep.datasets})


if __name__ == "__main__":
    main(sys.argv[1:])
