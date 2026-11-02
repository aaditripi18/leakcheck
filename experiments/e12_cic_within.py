"""E12 (PLAN.md, phase 3b): CIC-IDS2017's time split without unseen attack types.

E2's chronological split of CIC-IDS2017 trains on the early capture files and
tests on the late ones, so most test attack types never appear in training, and
the models collapse. That mixes two things: time and unseen attacks. Here the
split is chronological *within each label*: for every label, its earliest 80% of
flows (by capture file, then row order) train and its latest 20% test, so every
test label also appears in training. The E2 baselines score this split and the
stratified random split, with the same training cap, seeds 0-2.

    python -m experiments.e12_cic_within
"""
from __future__ import annotations

import numpy as np

from experiments import data
from experiments.e1_audit import random_split
from experiments.e2_retrain import MODELS, fit_predict, scores, subsample
from experiments.ledger import record
from leakcheck import Split

SEEDS = (0, 1, 2)
TRAIN_FRACTION = 0.8


def frame():
    df = data.cached(data.load_cicids2017)
    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    return df


def within_label_time_split(df):
    order = np.lexsort((np.arange(len(df)), df["capture_order"].to_numpy()))  # capture file, then row
    df = df.iloc[order].reset_index(drop=True)
    rank = df.groupby("Label").cumcount()
    size = df.groupby("Label")["Label"].transform("size")
    train = rank < np.floor(TRAIN_FRACTION * size)
    return df[train].reset_index(drop=True), df[~train].reset_index(drop=True)


def split_of(tr, te) -> Split:
    return Split(tr.drop(columns=["Label"]), te.drop(columns=["Label"]), "is_attack", time="capture_order")


def main() -> None:
    df = frame()
    tr_t, te_t = within_label_time_split(df)
    unseen = sorted(set(te_t["Label"]) - set(tr_t["Label"]))
    per_seed = []
    for seed in SEEDS:
        tr_r, te_r = random_split(df, df["Label"], seed)
        res = {"seed": seed, "models": {}}
        for name, (tr, te) in (("random", (tr_r, te_r)), ("time_within_label", (tr_t, te_t))):
            s = subsample(split_of(tr, te), seed)
            for m in MODELS:
                pred, _ = fit_predict(s, m, seed)
                res["models"].setdefault(m, {})[name] = scores(s.y_test, pred)
                print(f"   seed {seed} {name:18s} {m}: macro-F1 {res['models'][m][name]['macro_f1']:.4f}",
                      flush=True)
        per_seed.append(res)
    summary = {m: {"random": float(np.mean([r["models"][m]["random"]["macro_f1"] for r in per_seed])),
                   "time_within_label": float(np.mean([r["models"][m]["time_within_label"]["macro_f1"]
                                                       for r in per_seed]))} for m in MODELS}
    for v in summary.values():
        v["cost"] = v["random"] - v["time_within_label"]
    record("e12_cic_within", {"seeds": list(SEEDS), "train_fraction": TRAIN_FRACTION, "models": list(MODELS)},
           {"unseen_test_labels": unseen, "n_train": len(tr_t), "n_test": len(te_t), "per_seed": per_seed,
            "summary": summary}, {"cicids2017": data.dataset_hashes("cicids2017")})
    print({m: round(100 * v["cost"], 2) for m, v in summary.items()})


if __name__ == "__main__":
    main()
