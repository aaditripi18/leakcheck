"""E14 (PLAN.md, amendment 2026-11-09): H19's statistic S and H20's per-class gaps.

    python -m experiments.e14_explain stat kdd99 nslkdd ...    S only (seed 0; no model trained)
    python -m experiments.e14_explain full ton_network ...     E13 + per-class F1 + S

`stat` records `e14_stat:<case>`. `full` records `e14:<case>`: per seed and model, the
E13 quantities plus the full model's per-class F1 on the whole test set and without the
leaked rows. A fresh case's H19 prediction must be in the ledger (`h19_prediction:<case>`,
scripts/h19_predict.py) before `full` runs on it; `full` refuses otherwise.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

from experiments import data
from experiments.e1_audit import PREP
from experiments.e13_difficulty import _acc
from experiments.e2_retrain import MODELS, fit_predict, original_split, scores, subsample
from experiments.e9_controls import SEEDS
from experiments.ledger import LEDGER, record
from leakcheck.explain import NEAR_EPS, ROUNDS, TEXT_TAU, copies_in_train, leak_masks, statistic, with_train

FRESH_PREFIXES = ("ton_", "botiot", "xiiotid")


def _hashes(name: str) -> dict:
    prep = PREP[name.partition(":")[0]]()
    return {d: data.dataset_hashes(d) for d in prep.datasets}


def run_stat(name: str) -> dict:
    t0 = time.time()
    split = subsample(original_split(name, 0), 0)
    leaked = leak_masks(split)["near"]
    keep, left = copies_in_train(split, leaked) if leaked.any() else (np.ones(len(split.train), bool), 0)
    st = statistic(split, leaked, keep, seed=0) if leaked.any() else {"S": None}
    st.update(leaked_share=float(leaked.mean()), train_rows_removed=int((~keep).sum()), copies_left=left,
              seconds=round(time.time() - t0))
    print(f"   {name}: S {st['S']}, leaked {st['leaked_share']:.4f}, {st['seconds']} s", flush=True)
    return st


def _per_class(y, pred, mask) -> dict:
    labels = np.unique(y)
    f_all = f1_score(y, pred, labels=labels, average=None, zero_division=0)
    out = {}
    for c, fa in zip(labels, f_all):
        rest = ~mask
        n_rest = int((y[rest] == c).sum())
        fr = (f1_score(y[rest], pred[rest], labels=[c], average=None, zero_division=0)[0]
              if n_rest else None)
        out[str(c)] = {"n_test": int((y == c).sum()), "n_leaked": int(((y == c) & mask).sum()),
                       "f1_whole": float(fa), "f1_without_leaked": None if fr is None else float(fr)}
    return out


def run_full_seed(name: str, seed: int) -> dict:
    split = subsample(original_split(name, seed), seed)
    leaked = leak_masks(split)["near"]
    y = split.y_test
    out = {"dataset": name, "seed": seed, "n_test": int(len(y)), "n_leaked": int(leaked.sum()),
           "leaked_share": float(leaked.mean()), "models": {}}
    keep, left = copies_in_train(split, leaked) if leaked.any() else (np.ones(len(split.train), bool), 0)
    out["train_rows_removed"], out["leaked_rows_still_near"] = int((~keep).sum()), int(left)
    dedup = with_train(split, split.train[keep], split.test)
    for m in MODELS:
        r = {}
        for tag, s in (("full", split), ("copy_free", dedup)):
            if tag == "copy_free" and not leaked.any():
                continue
            pred, _ = fit_predict(s, m, seed)
            r[tag] = {"whole": scores(y, pred)["macro_f1"],
                      "without_leaked": scores(y[~leaked], pred[~leaked])["macro_f1"] if (~leaked).any() else None,
                      "acc_leaked": _acc(y, pred, leaked), "acc_rest": _acc(y, pred, ~leaked)}
            w, wo = r[tag]["whole"], r[tag]["without_leaked"]
            r[tag]["gap"] = (w - wo) if w is not None and wo is not None else None
            if tag == "full":
                r["per_class"] = _per_class(y, pred, leaked)
        out["models"][m] = r
        print(f"   {name} seed {seed} {m}: gap full {r['full']['gap']} copy-free "
              f"{r.get('copy_free', {}).get('gap')}", flush=True)
    return out


def _predicted(case: str) -> bool:
    if not LEDGER.exists():
        return False
    return any(json.loads(ln).get("experiment") == f"h19_prediction:{case}"
               for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip())


def main(mode: str, names: list[str]) -> None:
    for name in names:
        print(f"== {name} ({mode})", flush=True)
        if mode == "stat":
            record("e14_stat:" + name, {"seed": 0, "near_eps": NEAR_EPS, "text_tau": TEXT_TAU, "rounds": ROUNDS},
                   run_stat(name), _hashes(name))
        else:
            if name.startswith(FRESH_PREFIXES) and not _predicted(name):
                sys.exit(f"{name}: no H19 prediction in the ledger yet (scripts/h19_predict.py first)")
            per_seed = [run_full_seed(name, s) for s in SEEDS]
            record("e14:" + name, {"seeds": list(SEEDS), "near_eps": NEAR_EPS, "text_tau": TEXT_TAU,
                                   "rounds": ROUNDS, "models": list(MODELS)},
                   {"per_seed": per_seed}, _hashes(name))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
