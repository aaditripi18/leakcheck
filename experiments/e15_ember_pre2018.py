"""E15 (PLAN.md amendment 2026-11-12 (2)): the effect of EMBER 2018's 50,000 pre-2018 training files.

    python -m experiments.e15_ember_pre2018

Official time split, leakcheck's numeric view, the four baselines, seeds 0-2. Per seed, two
training sets of 300,000 files (stratified by label): from all training files (as published)
and from the 2018 files only. Both are scored on the whole official test set.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import train_test_split

from experiments import data
from experiments.e1_audit import PREP
from experiments.e2_retrain import MAX_TRAIN, MODELS, fit_predict, scores
from experiments.e9_controls import SEEDS
from experiments.ledger import record
from leakcheck.explain import with_train


def draw(train, y, seed: int, n: int = MAX_TRAIN):
    idx, _ = train_test_split(np.arange(len(train)), train_size=n, random_state=seed, stratify=y)
    return train.iloc[np.sort(idx)].reset_index(drop=True)


def main() -> None:
    prep = PREP["ember2018"]()
    s = {v: x for v, x, _ in prep.splits}["numeric"]
    early = s.train["appeared"].astype(str) < "2018"
    out = {"n_train": int(len(s.train)), "n_pre2018": int(early.sum()),
           "pre2018_labels": {str(k): int(v) for k, v in s.train.loc[early, s.label].value_counts().items()},
           "n_test": int(len(s.test)), "per_seed": []}
    y = s.y_test
    recent = s.train[~early].reset_index(drop=True)
    for seed in SEEDS:
        sets = {"as_published": draw(s.train, s.train[s.label], seed),
                "2018_only": draw(recent, recent[s.label], seed)}
        row = {"seed": seed, "models": {}}
        for m in MODELS:
            r = {}
            for tag, tr in sets.items():
                t0 = time.time()
                pred, _ = fit_predict(with_train(s, tr), m, seed)
                r[tag] = scores(y, pred)["macro_f1"]
                print(f"   seed {seed} {m} {tag}: macro-F1 {r[tag]:.4f} ({time.time() - t0:.0f} s)", flush=True)
            r["effect"] = r["as_published"] - r["2018_only"]
            row["models"][m] = r
        out["per_seed"].append(row)
    out["mean_effect"] = {m: float(np.mean([p["models"][m]["effect"] for p in out["per_seed"]])) for m in MODELS}
    out["h21"] = all(abs(v) < 0.005 for v in out["mean_effect"].values())
    print("mean effect (points):", {m: round(v * 100, 2) for m, v in out["mean_effect"].items()}, "H21", out["h21"])
    record("e15_ember_pre2018", {"seeds": list(SEEDS), "train_size": MAX_TRAIN, "models": list(MODELS),
                                 "split": prep.split_policy, "view": "numeric"},
           out, {d: data.dataset_hashes(d) for d in prep.datasets})


if __name__ == "__main__":
    main()
