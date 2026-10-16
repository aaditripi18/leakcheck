"""E5: per-row predictions of uncapped and tuned models (PLAN.md, phase 2).

For each dataset and seed 0-2, on the E2 split but with the *full* training set:
train lr, rf, hgb, mlp (the E2 settings) and hgb_tuned, and save for every test
row its label, each model's prediction, whether it has an exact copy in the
training data, the RMS distance to its nearest training row (tabular), and its
best Jaccard to a training document (text). Where the dataset has a
chronological split, the same models are also trained and scored on it.

Saved to results/e5/<dataset>_seed<s>.npz (not committed: regenerate with this
script) with a small JSON beside it (commit, timings, tuned parameters), which
the analysis (experiments/e6_analysis.py) reads. Cached by git commit, as E2.

  python -m experiments.e5_predictions uci_phishing sms_spam ...
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from experiments import data
from experiments.e2_retrain import TIME_SPLITS, fit_predict, original_split
from experiments.ledger import git_state
from experiments.models import MODELS, build
from leakcheck import Split
from leakcheck.checks import near, near_text

SEEDS = (0, 1, 2)
ALL_MODELS = MODELS + ("hgb_tuned",)
TUNE_ROWS = 200_000
TUNE_ITER = 12
MIN_TAU = 0.6  # the lowest Jaccard the analysis sweeps to
OUT = data.ROOT / "results" / "e5"

HGB_SPACE = {
    "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [None, 6, 10, 16],
    "model__max_leaf_nodes": [15, 31, 63, 127],
    "model__l2_regularization": [0.0, 0.1, 1.0],
    "model__max_iter": [200, 400, 800],
}


def tune_hgb(split: Split, seed: int) -> tuple[dict, float]:
    """Randomised search on a stratified subsample of the training set."""
    X, y = split.train[split.features], split.y_train
    if len(X) > TUNE_ROWS:
        idx, _ = train_test_split(np.arange(len(X)), train_size=TUNE_ROWS, random_state=seed, stratify=y)
        X, y = X.iloc[np.sort(idx)], y[np.sort(idx)]
    search = RandomizedSearchCV(build("hgb", X, split.text, seed), HGB_SPACE, n_iter=TUNE_ITER, scoring="f1_macro",
                                cv=StratifiedKFold(3, shuffle=True, random_state=seed), random_state=seed, n_jobs=-1,
                                refit=False)
    search.fit(X, y)
    return search.best_params_, float(search.best_score_)


def fit_predict_any(split: Split, model: str, seed: int, tuned: dict | None) -> tuple[np.ndarray, float]:
    if model != "hgb_tuned":
        return fit_predict(split, model, seed)
    Xtr, Xte = split.train[split.features], split.test[split.features]
    t0 = time.time()
    pipe = build("hgb", Xtr, split.text, seed).set_params(**tuned).fit(Xtr, split.y_train)
    return pipe.predict(Xte), time.time() - t0


def row_evidence(split: Split) -> dict[str, np.ndarray]:
    """Per test row: exact copy, nearest RMS distance, best text Jaccard."""
    n = len(split.test)
    exact = np.isin(split.test_hash, split.train_hash)
    dist = np.full(n, np.nan)
    sim = np.full(n, np.nan)
    todo = np.flatnonzero(~exact)
    if near.numeric_columns(split) and len(todo):
        dist[todo], _ = near.nearest(split, todo)
    if split.text is not None and len(todo):
        s, _ = near_text.nearest_text(split.train[split.text], split.test[split.text], todo, MIN_TAU)
        tr_norm = set(split.train[split.text].fillna("").map(near_text.normalize))
        same = split.test[split.text].fillna("").map(near_text.normalize).iloc[todo].isin(tr_norm).to_numpy()
        sim[todo] = np.where(same, 1.0, s)
    dist[exact] = 0.0
    sim[exact] = 1.0
    return {"exact": exact, "dist": dist, "sim": sim}


# The code that determines E5's predictions. Results are reused while these
# files are unchanged, so unrelated commits do not force hours of retraining.
CODE = ["experiments/e5_predictions.py", "experiments/e2_retrain.py", "experiments/models.py",
        "experiments/data.py", "experiments/e1_audit.py", "experiments/ember.py",
        "leakcheck/split.py", "leakcheck/checks/near.py", "leakcheck/checks/near_text.py"]


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for rel in CODE:
        f = data.ROOT / rel
        body = f.read_bytes().replace(b"\r\n", b"\n") if f.exists() else b""
        h.update(rel.encode() + b"\0" + body + b"\0")
    return h.hexdigest()[:16]


def run_one(name: str, seed: int) -> dict:
    meta_path = OUT / f"{name}_seed{seed}.json"
    commit = git_state()["commit"]
    fp = code_fingerprint()
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("code") == fp and (OUT / meta["npz"]).exists():
            return meta
    t0 = time.time()
    split = original_split(name, seed)  # full training set: no subsample
    arrays = {"y": np.asarray(split.y_test), **row_evidence(split)}
    meta = {"dataset": name, "seed": seed, "commit": commit, "code": fp, "n_train": len(split.train), "n_test": len(split.test),
            "npz": f"{name}_seed{seed}.npz", "fit_seconds": {}, "evidence_seconds": round(time.time() - t0, 1)}
    print(f"   {name} seed {seed}: evidence in {meta['evidence_seconds']} s "
          f"(train {len(split.train):,}, test {len(split.test):,})", flush=True)
    tuned, cv = tune_hgb(split, seed)
    meta["hgb_tuned"] = {"params": {k.split("__", 1)[1]: v for k, v in tuned.items()}, "cv_macro_f1": cv}
    print(f"   {name} seed {seed}: tuned {meta['hgb_tuned']}", flush=True)
    for m in ALL_MODELS:
        pred, secs = fit_predict_any(split, m, seed, tuned)
        arrays[f"pred_{m}"] = np.asarray(pred)
        meta["fit_seconds"][m] = round(secs, 1)
        print(f"   {name} seed {seed} {m}: {secs:.0f} s", flush=True)
    if name in TIME_SPLITS:
        ts = TIME_SPLITS[name](seed)
        arrays["time_y"] = np.asarray(ts.y_test)
        meta["time"] = {"n_train": len(ts.train), "n_test": len(ts.test)}
        for m in ALL_MODELS:
            pred, _ = fit_predict_any(ts, m, seed, tuned)
            arrays[f"time_pred_{m}"] = np.asarray(pred)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / meta["npz"], **arrays)
    meta["seconds"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


def main(names: list[str]) -> None:
    for name in names:
        print(f"== {name}", flush=True)
        for s in SEEDS:
            run_one(name, s)


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main(sys.argv[1:])
