"""E4: retrain without the shortcut features that E1 flagged.

Drop sets are read from the latest E1 ledger line of each dataset, not typed
by hand:
  minus_top      E1's best single feature (L5.shortcut top-1)
  minus_flagged  every feature E1 flagged at >= 0.95 balanced accuracy, plus
                 identifier-like columns (L5.identifier), minus known false
                 positives listed below
Identical drop sets are run once. After each ablation, the single-feature
check is re-run (seed 0) to show which feature takes over.

Same harness as E2: 4 untuned models, seeds 0-4, random splits re-drawn per
seed, training sets capped at 300k; macro-F1 on the full test set.

  python -m experiments.e4_ablation phiusiil cicids2017
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from experiments import data
from experiments.e1_audit import PREP
from experiments.e2_retrain import fit_predict, like, scores, subsample
from experiments.ledger import LEDGER, git_state, record
from experiments.models import MODELS
from leakcheck import Split
from leakcheck.checks import shortcut

SEEDS = tuple(range(5))
# identifier-heuristic hits that are features, not identifiers (FINDINGS.md section 10)
KNOWN_FALSE_IDENTIFIERS = {"kdd99": {"dst_host_same_src_port_rate"}}
OUT = data.ROOT / "results" / "e4"


def e1_findings(name: str) -> dict:
    last = None
    for line in open(LEDGER, encoding="utf-8"):
        d = json.loads(line)
        if d["experiment"] == f"e1_audit:{name}":
            last = d
    if last is None:
        raise KeyError(f"no E1 ledger line for {name}")
    f = {}
    for v in last["results"]["views"].values():
        f.update(v["findings"])
    return f


def drop_sets(name: str) -> dict[str, list[str]]:
    f = e1_findings(name)
    top = [f["L5.shortcut"]["top"][0]["feature"]]
    ids = [c["feature"] for c in f.get("L5.identifier", {}).get("columns", [])
           if c["feature"] not in KNOWN_FALSE_IDENTIFIERS.get(name, set()) and "name" in c["reasons"]]
    flagged = list(dict.fromkeys(f["L5.shortcut"].get("flagged", []) + ids))
    sets = {"full": [], "minus_top": top}
    if set(flagged) != set(top) and flagged:
        sets["minus_flagged"] = flagged
    return sets


def without(split: Split, cols: list[str]) -> Split:
    s = like(split, split.train, split.test)
    s.ignore = list(split.ignore) + [c for c in cols if c not in split.ignore]
    return s


def run_one(name: str, seed: int, sets: dict[str, list[str]]) -> dict:
    cache = OUT / f"{name}_seed{seed}.json"
    commit = git_state()["commit"]
    if cache.exists():
        res = json.loads(cache.read_text(encoding="utf-8"))
        if res.get("commit") == commit:
            return res
    t0 = time.time()
    (_, split, _), *_ = PREP[name](seed).splits
    split = subsample(split, seed)
    out = {"dataset": name, "seed": seed, "commit": commit, "conditions": {}}
    for cond, cols in sets.items():
        s = without(split, cols)
        out["conditions"][cond] = {"dropped": cols, "models": {}}
        for m in MODELS:
            pred, secs = fit_predict(s, m, seed)
            out["conditions"][cond]["models"][m] = {**scores(s.y_test, pred), "fit_seconds": round(secs, 1)}
            print(f"   {name} seed {seed} {cond} {m}: macro-F1 "
                  f"{out['conditions'][cond]['models'][m]['macro_f1']:.4f}", flush=True)
        if seed == 0:
            f = shortcut.check_shortcuts(s, seed=seed)
            out["conditions"][cond]["best_single_feature_after"] = f.details["top"][:3]
    out["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def summarize(per_seed: list[dict]) -> dict:
    conds = per_seed[0]["conditions"]
    agg = {}
    for c in conds:
        agg[c] = {"dropped": conds[c]["dropped"], "models": {}}
        for m in MODELS:
            v = np.array([r["conditions"][c]["models"][m]["macro_f1"] for r in per_seed])
            agg[c]["models"][m] = {"macro_f1_mean": float(v.mean()), "macro_f1_std": float(v.std(ddof=1))}
        agg[c]["best_single_feature_after"] = per_seed[0]["conditions"][c].get("best_single_feature_after")
    full = agg["full"]["models"]
    drops = {c: {m: full[m]["macro_f1_mean"] - agg[c]["models"][m]["macro_f1_mean"] for m in MODELS}
             for c in agg if c != "full"}
    return {"conditions": agg, "macro_f1_drop_vs_full": drops}


def main(names: list[str]) -> None:
    for name in names:
        sets = drop_sets(name)
        print(f"== {name}: {sets}", flush=True)
        per_seed = [run_one(name, s, sets) for s in SEEDS]
        summary = summarize(per_seed)
        prep = PREP[name]()
        record("e4_ablation:" + name, {"seeds": list(SEEDS), "drop_sets": sets, "models": list(MODELS),
                                        "split_policy": prep.split_policy},
               summary, {d: data.dataset_hashes(d) for d in prep.datasets})
        for c, d in summary["macro_f1_drop_vs_full"].items():
            print(f"   {c}: " + ", ".join(f"{m} {v * 100:+.2f} pt" for m, v in d.items()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
