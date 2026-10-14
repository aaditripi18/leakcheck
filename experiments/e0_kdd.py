"""E0b: rediscover the KDD99 redundancy published by Tavallaee et al. (2009).

Reference values are Tables I and II of the paper ("Statistics of redundant
records in the KDD train / test set"), copied from the PDF on 2026-10-14.
A record is the full line: 41 features + label. "Distinct" counts unique
records; attack/normal counts are per class.

Then the leakcheck view the paper does not give: cross-split overlap between
KDD99 train and test, label conflicts, and the same scan on NSL-KDD.

  python -m experiments.e0_kdd
"""

from __future__ import annotations

import json

import numpy as np

from experiments.data import KDD_FEATURES, dataset_hashes, load_kdd99, load_nslkdd
from experiments.ledger import record
from leakcheck import Split, scan
from leakcheck.split import canonicalize, row_hashes

TAVALLAEE = {
    "train": {"attack": (3_925_650, 262_178), "normal": (972_781, 812_814), "total": (4_898_431, 1_074_992)},
    "test": {"attack": (250_436, 29_378), "normal": (60_591, 47_911), "total": (311_027, 77_289)},
}


def redundancy(df) -> dict:
    h = row_hashes(canonicalize(df[KDD_FEATURES + ["label"]]))
    attack = df["is_attack"].to_numpy() == 1
    out = {}
    for name, m in (("attack", attack), ("normal", ~attack), ("total", np.ones(len(df), bool))):
        n, d = int(m.sum()), int(len(np.unique(h[m])))
        out[name] = {"records": n, "distinct": d, "reduction": 1 - d / n}
    return out


def compare(ours: dict, paper: dict) -> dict:
    return {
        k: {"ours": [ours[k]["records"], ours[k]["distinct"]], "paper": list(paper[k]),
            "diff": [ours[k]["records"] - paper[k][0], ours[k]["distinct"] - paper[k][1]]}
        for k in paper
    }


def kdd_split(train, test) -> Split:
    return Split(train.drop(columns=["label"]), test.drop(columns=["label"]), "is_attack")


def main() -> dict:
    full, test = load_kdd99("full"), load_kdd99("test")
    nsl_tr, nsl_te = load_nslkdd("train"), load_nslkdd("test")

    red = {"train": redundancy(full), "test": redundancy(test)}
    vs_paper = {s: compare(red[s], TAVALLAEE[s]) for s in red}

    # leakcheck on the binary attack/normal task (label column dropped so the
    # attack name cannot leak into the features)
    kdd_report = scan(kdd_split(full, test), checks=("L1", "L3"))
    nsl_report = scan(Split(nsl_tr.drop(columns=["label"]), nsl_te.drop(columns=["label"]), "is_attack",
                            ignore=["difficulty"]))

    def pick(rep, codes):
        return {c: {"value": rep.get(c).value, "severity": rep.get(c).severity, **{
            k: v for k, v in rep.get(c).details.items() if k in ("n_hit", "n_groups", "n_rows", "n_near", "ci95", "top", "per_class",
                                          "curve", "train_baseline_curve", "vs_train_baseline")}}
            for c in codes}

    results = {
        "tavallaee_comparison": vs_paper,
        "kdd99_leakcheck": pick(kdd_report, ["L1.within_train", "L1.within_test", "L1.cross", "L3.conflict",
                                             "L3.ceiling", "L3.cross_contradict"]),
        "nslkdd_leakcheck": pick(nsl_report, ["L1.within_train", "L1.within_test", "L1.cross", "L2.near",
                                              "L3.conflict", "L3.ceiling", "L3.cross_contradict", "L5.shortcut"]),
    }
    results["nslkdd_leakcheck"]["L5.shortcut"]["top"] = results["nslkdd_leakcheck"]["L5.shortcut"]["top"][:5]
    datasets = {n: dataset_hashes(n) for n in ("kdd99", "kdd99_test", "nslkdd")}
    record("e0_kdd", {"task": "binary is_attack", "nsl_ignore": ["difficulty"], "near_eps": 0.01}, results, datasets)
    print(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
