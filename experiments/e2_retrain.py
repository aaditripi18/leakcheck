"""E2: how much of a baseline's score comes from leaked test rows?

For each dataset, seed and model, train once and predict the whole test set,
then score it on:
  original   every test row
  no_exact   test rows with no exact copy in the training data the model saw
  no_near    test rows with neither an exact nor a near copy
             (tabular RMS distance <= 0.01; text Jaccard >= 0.8 or identical)
  leaked     the rows removed for no_near (memorisation shows up here)
and, where the data has a timestamp, retrain on a chronological split:
  time       train on earlier capture files, test on later ones

Settings fixed before any E2 result (see PLAN.md, "E2 operationalisation"):
seeds 0-4; random-split datasets are re-split per seed, official splits are
fixed; training sets above 300,000 rows are stratified-subsampled per seed.

  python -m experiments.e2_retrain uci_phishing sms_spam unsw_nb15 kdd99 cicids2017
"""

from __future__ import annotations

import csv
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from experiments import data
from experiments.e1_audit import PREP
from experiments.ledger import git_state, record
from experiments.models import MODELS, build
from leakcheck import Split
from leakcheck.checks import near, near_text

SEEDS = tuple(range(5))
MAX_TRAIN = 300_000
NEAR_EPS = 0.01
TEXT_TAU = 0.8
TIME_FRACTION = 0.8
OUT = data.ROOT / "results" / "e2"
RETRAIN_CSV = data.ROOT / "results" / "retrain.csv"


def like(split: Split, train: pd.DataFrame, test: pd.DataFrame) -> Split:
    return Split(train, test, split.label, time=split.time, group=split.group, ignore=list(split.ignore),
                 text=split.text)


def subsample(split: Split, seed: int, cap: int = MAX_TRAIN) -> Split:
    if len(split.train) <= cap:
        return split
    idx, _ = train_test_split(np.arange(len(split.train)), train_size=cap, random_state=seed,
                              stratify=split.train[split.label])
    return like(split, split.train.iloc[np.sort(idx)], split.test)


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


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "balanced_accuracy": None}
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def fit_predict(split: Split, model: str, seed: int) -> tuple[np.ndarray, float]:
    Xtr, Xte = split.train[split.features], split.test[split.features]
    t0 = time.time()
    pipe = build(model, Xtr, split.text, seed).fit(Xtr, split.y_train)
    return pipe.predict(Xte), time.time() - t0


# --- chronological splits -------------------------------------------------------------


def chrono_cut(order: pd.Series, fraction: float = TIME_FRACTION) -> int:
    """Last training period: the boundary whose cumulative row share is closest to `fraction`."""
    sizes = order.value_counts().sort_index()
    cum = sizes.cumsum() / sizes.sum()
    return int((cum - fraction).abs().idxmin())


def time_split_cic(seed: int) -> Split:
    """Train on capture files up to the boundary closest to 80% of rows, test on the rest."""
    df = data.cached(data.load_cicids2017)
    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    cut = chrono_cut(df["capture_order"])
    tr, te = df[df["capture_order"] <= cut], df[df["capture_order"] > cut]
    return Split(tr.drop(columns=["Label"]).reset_index(drop=True), te.drop(columns=["Label"]).reset_index(drop=True),
                 "is_attack", time="capture_order")


EMBER_TEST = 200_000


def _ember_labelled() -> pd.DataFrame:
    from experiments.ember import CATEGORICAL, columns, load_ember

    df = data.cached(load_ember)
    df = df[df["label"] != -1]
    return df[["split", "appeared"] + columns() + list(CATEGORICAL) + ["label"]].reset_index(drop=True)


def ember_random(seed: int) -> Split:
    """All 800k labelled files, stratified random split with the official sizes (600k / 200k)."""
    df = _ember_labelled()
    tr, te = train_test_split(df, test_size=EMBER_TEST, random_state=seed, stratify=df["label"])
    return Split(tr.drop(columns=["split"]).reset_index(drop=True), te.drop(columns=["split"]).reset_index(drop=True),
                 "label", time="appeared")


def ember_temporal(seed: int) -> Split:
    """The official split: train up to Oct 2018, test Nov-Dec 2018."""
    df = _ember_labelled()
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    return Split(tr.drop(columns=["split"]).reset_index(drop=True), te.drop(columns=["split"]).reset_index(drop=True),
                 "label", time="appeared")


TIME_SPLITS = {"cicids2017": time_split_cic, "ember2018": ember_temporal}
# E2 "original" split where it differs from E1's audit split: for EMBER the
# original is a random split, so the official temporal split can play "time".
E2_SPLITS = {"ember2018": ember_random}


def original_split(name: str, seed: int) -> Split:
    if name in E2_SPLITS:
        return E2_SPLITS[name](seed)
    (_, split, _), *_ = PREP[name](seed).splits
    return split


# --- one (dataset, seed) ----------------------------------------------------------------


def run_one(name: str, seed: int) -> dict:
    cache = OUT / f"{name}_seed{seed}.json"
    commit = git_state()["commit"]
    if cache.exists():
        res = json.loads(cache.read_text(encoding="utf-8"))
        if res.get("commit") == commit:  # results from other code versions are recomputed
            return res
    t0 = time.time()
    split = subsample(original_split(name, seed), seed)
    masks = leak_masks(split)
    y = split.y_test
    subsets = {"original": np.ones(len(y), bool), "no_exact": ~masks["exact"], "no_near": ~masks["near"],
               "leaked": masks["near"]}
    out = {"dataset": name, "seed": seed, "commit": commit, "n_train": len(split.train), "n_test": len(y),
           "leak_rate": {"exact": float(masks["exact"].mean()), "near": float(masks["near"].mean())}, "models": {}}
    for m in MODELS:
        pred, secs = fit_predict(split, m, seed)
        out["models"][m] = {c: scores(y[s], pred[s]) for c, s in subsets.items()}
        out["models"][m]["fit_seconds"] = round(secs, 1)
        print(f"   {name} seed {seed} {m}: original {out['models'][m]['original']['macro_f1']:.4f} "
              f"no_near {out['models'][m]['no_near']['macro_f1'] or float('nan'):.4f} ({secs:.0f} s)", flush=True)
    if name in TIME_SPLITS:
        ts = subsample(TIME_SPLITS[name](seed), seed)
        out["time"] = {"n_train": len(ts.train), "n_test": len(ts.test),
                       "exact_overlap": float(np.isin(ts.test_hash, ts.train_hash).mean()), "models": {}}
        for m in MODELS:
            pred, secs = fit_predict(ts, m, seed)
            out["time"]["models"][m] = {"time": scores(ts.y_test, pred), "fit_seconds": round(secs, 1)}
            print(f"   {name} seed {seed} {m}: time {out['time']['models'][m]['time']['macro_f1']:.4f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --- aggregate --------------------------------------------------------------------------


def rows(res: dict) -> list[dict]:
    out = []
    for m, conds in res["models"].items():
        for c, s in conds.items():
            if c != "fit_seconds":
                out.append({"dataset": res["dataset"], "seed": res["seed"], "model": m, "condition": c, **s})
    for m, conds in res.get("time", {}).get("models", {}).items():
        out.append({"dataset": res["dataset"], "seed": res["seed"], "model": m, "condition": "time", **conds["time"]})
    return out


def summarize(per_seed: list[dict]) -> dict:
    df = pd.DataFrame([r for res in per_seed for r in rows(res)])
    for k in ("accuracy", "macro_f1", "balanced_accuracy"):
        df[k] = pd.to_numeric(df[k], errors="coerce")  # empty subsets score None
    agg = {}
    for (m, c), g in df.groupby(["model", "condition"]):
        agg.setdefault(m, {})[c] = {
            "n_mean": float(g["n"].mean()),
            **{f"{k}_mean": float(g[k].mean()) for k in ("accuracy", "macro_f1", "balanced_accuracy")},
            **{f"{k}_std": float(g[k].std(ddof=1)) if len(g) > 1 else 0.0 for k in ("accuracy", "macro_f1")},
        }
    gaps = {m: {"orig_minus_no_exact": v["original"]["macro_f1_mean"] - v["no_exact"]["macro_f1_mean"],
                "orig_minus_no_near": v["original"]["macro_f1_mean"] - v["no_near"]["macro_f1_mean"],
                **({"orig_minus_time": v["original"]["macro_f1_mean"] - v["time"]["macro_f1_mean"]} if "time" in v else {})}
            for m, v in agg.items()}
    leak = {k: float(np.mean([r["leak_rate"][k] for r in per_seed])) for k in ("exact", "near")}
    return {"by_model": agg, "macro_f1_gap": gaps, "leak_rate_mean": leak,
            "n_train": per_seed[0]["n_train"], "n_test_mean": float(np.mean([r["n_test"] for r in per_seed]))}


def main(names: list[str]) -> None:
    for name in names:
        print(f"== {name}", flush=True)
        per_seed = [run_one(name, s) for s in SEEDS]
        summary = summarize(per_seed)
        prep = PREP[name]()
        policy = ("random 600k/200k of all labelled files (original) vs official temporal (time)"
                  if name in E2_SPLITS else prep.split_policy)
        record("e2_retrain:" + name, {"seeds": list(SEEDS), "max_train": MAX_TRAIN, "near_eps": NEAR_EPS,
                                       "text_tau": TEXT_TAU, "models": list(MODELS), "split_policy": policy},
               summary, {d: data.dataset_hashes(d) for d in prep.datasets})
        all_rows = [r for res in per_seed for r in rows(res)]
        existing = []
        if RETRAIN_CSV.exists():
            existing = [r for r in csv.DictReader(open(RETRAIN_CSV, encoding="utf-8")) if r["dataset"] != name]
        with open(RETRAIN_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(existing + all_rows)
        for m, g in summary["macro_f1_gap"].items():
            print(f"   {m}: " + ", ".join(f"{k} {v * 100:+.2f} pt" for k, v in g.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
