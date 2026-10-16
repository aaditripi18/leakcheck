"""E7: UNSW-NB15 raw records, chronological split vs random split (PLAN.md, phase 2).

1. Provenance gate (experiments/unsw_raw.provenance): stop unless the files hold
   2,540,044 records, 49 columns and all nine attack families.
2. Splits: `time` = records sorted by stime, train the earliest 80% (cut at the
   80th percentile of stime), test the rest; `random` = stratified 80/20 by label,
   per seed. The time split is reported as confounded (and not used for H9) if a
   family has fewer than 50 records on either side.
3. For each split and seed 0-2: E5's row evidence and models (lr, rf, hgb, mlp,
   hgb_tuned on the full training set), saved to results/e7/<split>_seed<s>.npz.
4. Ledger line e7_unsw_time: provenance, split composition, macro-F1 per model and
   split, the random -> time drop, the original -> no_near drop on the random
   split, and the H9 verdict by its rule.

  python -m experiments.e7_unsw_time
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from experiments import data, unsw_raw
from experiments.e5_predictions import ALL_MODELS, fit_predict_any, row_evidence, tune_hgb
from experiments.e6_analysis import EPS, TAU, leaked_mask
from experiments.ledger import record
from leakcheck import Split
from leakcheck.scan import scan as audit

SEEDS = (0, 1, 2)
OUT = data.ROOT / "results" / "e7"
MIN_FAMILY = 50


def time_split(df):
    cut = float(np.quantile(df["stime"], 0.8))
    tr, te = df[df["stime"] <= cut], df[df["stime"] > cut]
    return tr.reset_index(drop=True), te.reset_index(drop=True), cut


def make_split(tr, te) -> Split:
    keep = unsw_raw.features(tr) + ["label"]
    return Split(tr[keep], te[keep], "label")


def composition(tr, te) -> dict:
    a, b = tr["attack_cat"].value_counts(), te["attack_cat"].value_counts()
    fams = {f: [int(a.get(f, 0)), int(b.get(f, 0))] for f in unsw_raw.FAMILIES + ["Normal"]}
    return {"train": len(tr), "test": len(te), "families": fams,
            "confounded": [f for f in unsw_raw.FAMILIES if min(fams[f]) < MIN_FAMILY]}


def run(name: str, split: Split, seed: int) -> dict:
    path = OUT / f"{name}_seed{seed}.npz"
    meta_path = OUT / f"{name}_seed{seed}.json"
    if meta_path.exists() and path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    t0 = time.time()
    arrays = {"y": np.asarray(split.y_test), **row_evidence(split)}
    tuned, cv = tune_hgb(split, seed)
    meta = {"split": name, "seed": seed, "hgb_tuned": {"params": {k.split("__", 1)[1]: v for k, v in tuned.items()},
                                                       "cv_macro_f1": cv}, "fit_seconds": {}}
    for m in ALL_MODELS:
        pred, secs = fit_predict_any(split, m, seed, tuned)
        arrays[f"pred_{m}"] = np.asarray(pred)
        meta["fit_seconds"][m] = round(secs, 1)
        print(f"   {name} seed {seed} {m}: {secs:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta["seconds"] = round(time.time() - t0, 1)
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


def scores(name: str, seed: int) -> dict:
    z = np.load(OUT / f"{name}_seed{seed}.npz")
    ev = {k: z[k] for k in ("exact", "dist", "sim")}
    clean = ~leaked_mask(ev, EPS, TAU)
    y = z["y"]
    out = {}
    for m in ALL_MODELS:
        p = z[f"pred_{m}"]
        out[m] = {"original": float(f1_score(y, p, average="macro")),
                  "no_near": float(f1_score(y[clean], p[clean], average="macro")) if clean.any() else None}
    out["_leaked_share"] = float(1 - clean.mean())
    return out


def main() -> None:
    df = unsw_raw.load()
    prov = unsw_raw.provenance(df)
    print(json.dumps(prov, indent=1), flush=True)
    if not prov["passes"]:
        record("e7_unsw_time", {"gate": "failed"}, {"provenance": prov})
        raise SystemExit("provenance check failed: E7 does not run (PLAN.md)")
    tr, te, cut = time_split(df)
    comp = composition(tr, te)
    print(json.dumps(comp, indent=1), flush=True)
    splits = {"time": make_split(tr, te)}
    res = {"provenance": prov, "time_cut_stime": cut, "time_composition": comp, "seeds": {},
           "audit_time": audit(splits["time"]).to_dict()}
    for seed in SEEDS:
        a, b = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
        splits_seed = {"time": splits["time"], "random": make_split(a.reset_index(drop=True), b.reset_index(drop=True))}
        if seed == 0:
            res["audit_random_seed0"] = audit(splits_seed["random"]).to_dict()
        for name, sp in splits_seed.items():
            run(name, sp, seed)
        res["seeds"][seed] = {name: scores(name, seed) for name in splits_seed}
    gaps = {}
    for m in ALL_MODELS:
        rand = np.mean([res["seeds"][s]["random"][m]["original"] for s in SEEDS])
        tim = np.mean([res["seeds"][s]["time"][m]["original"] for s in SEEDS])
        nn = np.mean([res["seeds"][s]["random"][m]["no_near"] for s in SEEDS])
        gaps[m] = {"random": float(rand), "time": float(tim), "random_minus_time": float(rand - tim),
                   "random_minus_no_near": float(rand - nn)}
    wins = sum(g["random_minus_time"] > g["random_minus_no_near"] for g in gaps.values())
    res["gaps"] = gaps
    res["h9"] = {"wins": wins, "of": len(gaps), "confounded": bool(comp["confounded"]),
                 "verdict": None if comp["confounded"] else ("supported" if wins >= 3 else "not supported")}
    record("e7_unsw_time", {"seeds": list(SEEDS), "models": list(ALL_MODELS), "split": "stime 80th percentile",
                            "eps": EPS, "tau": TAU}, res)
    print(json.dumps(res["h9"]), flush=True)


if __name__ == "__main__":
    main()
