"""E0a: does leakcheck recover planted leakage?

Part 1  exact recovery. Grid of planted rates x 5 seeds. For every cell the
        recovered count must equal the planted count; the result reports the
        maximum absolute error per check (expected: 0).
Part 2  sampled L2 interval coverage. 200 runs with the near-duplicate rate
        estimated from a 500-row sample: the 95% Wilson interval should cover
        the true rate in about 95% of runs.
Part 3  MinHash recall. Text pairs with controlled true Jaccard: recall of
        L2.near_text per Jaccard bin at tau = 0.8. False positives are
        impossible by construction (every reported pair is verified exactly),
        and the run checks that too.

  python -m experiments.e0_synthetic
"""

from __future__ import annotations

import itertools
import string

import numpy as np
import pandas as pd

from experiments.ledger import record
from leakcheck import Split
from leakcheck.checks import exact, near, near_text
from leakcheck.synth import make_split

SEEDS = range(5)
CROSS = (0.0, 0.001, 0.01, 0.05, 0.2, 0.5)
NEAR = (0.0, 0.01, 0.05, 0.2)
CONFLICT = (0.0, 0.01, 0.05)
TRAIN_DUP = (0.0, 0.1)


def part1() -> tuple[pd.DataFrame, dict]:
    rows = []
    for seed, c, nr, cf, td in itertools.product(SEEDS, CROSS, NEAR, CONFLICT, TRAIN_DUP):
        if c + nr > 0.9:
            continue
        train, test, t = make_split(n_train=3000, n_test=1500, n_features=10, n_classes=3, cross_dup=c,
                                    near_dup=nr, conflict=cf, train_dup=td, seed=seed)
        s = Split(train, test, "label")
        cross = exact.check_cross(s).details["n_hit"]
        within = {f.code: f for f in exact.check_within(s)}["L1.within_train"]
        conf = {f.code: f for f in exact.check_conflicts(s)}["L3.conflict"]
        nn = near.check_near(s, sample=10**9).details["n_near"]
        rows.append({
            "seed": seed, "cross": c, "near": nr, "conflict": cf, "train_dup": td,
            "err_cross": cross - t["cross_dup_rows"],
            "err_near": nn - t["near_dup_rows"],
            "err_conflict_groups": conf.details["n_groups"] - t["conflict_pairs"],
            # each planted conflict appends a feature-identical copy, so it is also a within-train repeat
            "err_within_train": (within.details["n_rows"] - within.details["n_unique"])
            - (t["train_dup_rows"] + t["conflict_pairs"]),
        })
    df = pd.DataFrame(rows)
    errs = [c for c in df.columns if c.startswith("err_")]
    return df, {"cells": len(df), "max_abs_error": {c: int(df[c].abs().max()) for c in errs}}


def part2(runs: int = 200, sample: int = 500, rate: float = 0.1) -> dict:
    covered = []
    for seed in range(runs):
        train, test, t = make_split(n_train=2000, n_test=4000, near_dup=rate, seed=1000 + seed)
        f = near.check_near(Split(train, test, "label"), sample=sample, seed=seed)
        lo, hi = f.details["ci95"]
        covered.append(lo <= t["near_dup_rows"] / len(test) <= hi)
    return {"runs": runs, "sample": sample, "true_rate": rate, "coverage": float(np.mean(covered))}


def _mutate(s: str, k: int, rng: np.random.Generator) -> str:
    chars = list(s)
    for i in rng.choice(len(chars), k, replace=False):
        chars[i] = "#"
    return "".join(chars)


def part3(n_per_k: int = 200, length: int = 120) -> dict:
    rng = np.random.default_rng(7)
    alphabet = np.array(list(string.ascii_lowercase + string.digits + "/.-_"))
    pairs = []
    base = ["".join(rng.choice(alphabet, length)) for _ in range(n_per_k * 8)]
    i = 0
    for k in range(0, 8):  # k character substitutions
        for _ in range(n_per_k):
            pairs.append((base[i], _mutate(base[i], k, rng)))
            i += 1
    train = pd.DataFrame({"t": [a for a, _ in pairs], "label": 0})
    test = pd.DataFrame({"t": [b for _, b in pairs], "label": 0})
    sim, nn = near_text.nearest_text(train["t"], test["t"], tau=near_text.TAU)
    true_j = np.array([near_text.jaccard(near_text.shingles(a), near_text.shingles(b)) for a, b in pairs])
    found = (sim >= near_text.TAU) & (nn == np.arange(len(pairs)))
    bins = [(0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0001)]
    recall = {}
    for lo, hi in bins:
        m = (true_j >= lo) & (true_j < hi)
        recall[f"{lo:.2f}-{min(hi, 1):.2f}"] = {"n": int(m.sum()), "recall": float(found[m].mean()) if m.any() else None}
    reported = sim >= near_text.TAU
    false_pos = int((reported & (true_j < near_text.TAU) & (nn == np.arange(len(pairs)))).sum())
    # any reported pair must truly be >= tau (verified Jaccard), whichever train row it matched
    verified = all(
        near_text.jaccard(near_text.shingles(test["t"][i]), near_text.shingles(train["t"][nn[i]])) >= near_text.TAU
        for i in np.flatnonzero(reported)
    )
    return {"pairs": len(pairs), "length": length, "tau": near_text.TAU, "recall_by_jaccard": recall,
            "false_positives": false_pos, "all_reported_verified": verified}


def main() -> dict:
    df, p1 = part1()
    df.to_csv("results/e0_synthetic_grid.csv", index=False)
    results = {"exact_recovery": p1, "interval_coverage": part2(), "minhash": part3()}
    record("e0_synthetic", {"seeds": list(SEEDS), "cross": CROSS, "near": NEAR, "conflict": CONFLICT,
                            "train_dup": TRAIN_DUP}, results)
    import json
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
