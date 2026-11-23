"""E16 (descriptive, no hypothesis; FINDINGS section 24): why every model is perfect on Edge-IIoTset.

    python -m experiments.e16_edgeiiot_format

Section 24 guessed that five single features separate attack from normal because the captures
use different protocols. This measures it. Part 1 reads the CSV's raw text: for each of the five
features, the values written in normal rows and in attack rows, and how many they share. Part 2
reads every value that parses as a number as that number (so the text "0" and "0.0" are the same
value), then repeats leakcheck's single-feature check (L5) and E2's four baselines on the
same splits, seeds and views as E2, scored on the whole test set and without the leaked rows,
and compares exact copies and label conflicts (L1, L3) before and after, on the first seed.
"""
from __future__ import annotations

import csv
import io
import time
import zipfile

import numpy as np
import pandas as pd

from experiments import data
from experiments.e2_retrain import MODELS, fit_predict, original_split, scores, subsample
from experiments.e9_controls import SEEDS
from experiments.ledger import record
from leakcheck import Split, scan
from leakcheck.explain import leak_masks

FEATURES = ("dns.qry.name.len", "mqtt.conack.flags", "mqtt.msg", "mqtt.protoname", "mqtt.topic")
VIEWS = ("edgeiiot", "edgeiiot:no_identifiers")


def raw_values() -> dict:
    """The five features' values as text in the distributed CSV, by Attack_label."""
    (path,) = data.verify_local("edgeiiot")
    vals = {f: {"0": {}, "1": {}} for f in FEATURES}
    with zipfile.ZipFile(path) as zf, zf.open(data._zip_member(zf, "DNN-EdgeIIoT-dataset.csv")) as fh:
        rows = csv.reader(io.TextIOWrapper(fh, encoding="utf-8", newline=""))
        head = next(rows)
        at, cols = head.index("Attack_label"), {f: head.index(f) for f in FEATURES}
        for row in rows:
            for f, i in cols.items():
                d = vals[f][row[at]]
                d[row[i]] = d.get(row[i], 0) + 1
    out = {}
    for f, by in vals.items():
        shared = set(by["0"]) & set(by["1"])
        out[f] = {lab: {"rows": sum(c.values()), "distinct": len(c),
                        "top": sorted(c.items(), key=lambda kv: -kv[1])[:3]} for lab, c in by.items()}
        out[f]["shared_values"] = len(shared)
    return out


def as_numbers(s: pd.Series) -> pd.Series:
    """Every value that parses as a number becomes that number; a column that is all numbers is numeric."""
    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return s
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() == s.notna().sum():
        return num
    return s.where(num.isna(), num.map(repr)).astype(object)


def normalised(split: Split) -> Split:
    feats = [c for c in split.train.columns if c != split.label]
    tr, te = split.train.copy(), split.test.copy()
    for c in feats:
        both = as_numbers(pd.concat([tr[c], te[c]], ignore_index=True))
        tr[c], te[c] = both.iloc[:len(tr)].to_numpy(), both.iloc[len(tr):].to_numpy()
    return Split(tr, te, split.label)


def single_features(split: Split) -> list:
    f = scan(split, checks=("L5",)).get("L5.shortcut")
    return [(t["feature"], round(float(t["balanced_acc"]), 6)) for t in f.details["top"][:5]]


COPY_CODES = ("L1.cross", "L3.conflict", "L3.cross_contradict")


def copies(split: Split) -> dict:
    """Exact copies and label conflicts: rows that differ only in how a number is written count as different
    before reading numbers as numbers and as the same after."""
    rep = scan(split, checks=("L1", "L3"))
    return {c: float(rep.get(c).value) for c in COPY_CODES}


def run_view(name: str) -> dict:
    out = {"per_seed": []}
    for seed in SEEDS:
        t0 = time.time()
        raw = original_split(name, seed)
        if seed == SEEDS[0]:
            out["copies_raw"] = copies(raw)
        s = normalised(raw)
        del raw
        if seed == SEEDS[0]:
            out["copies_normalised"] = copies(s)
            out["single_features_normalised"] = single_features(s)
            print(f"   {name}: copies raw {out['copies_raw']} normalised {out['copies_normalised']}", flush=True)
        s = subsample(s, seed)
        leaked = leak_masks(s)["near"]
        y = s.y_test
        r = {"seed": seed, "n_train": int(len(s.train)), "n_test": int(len(y)), "n_leaked": int(leaked.sum()),
             "models": {}}
        for m in MODELS:
            pred, _ = fit_predict(s, m, seed)
            r["models"][m] = {"whole": scores(y, pred)["macro_f1"],
                              "without_leaked": scores(y[~leaked], pred[~leaked])["macro_f1"]}
        r["seconds"] = round(time.time() - t0)
        print(f"   {name} seed {seed}: " + ", ".join(f"{m} {v['whole']:.4f}/{v['without_leaked']:.4f}"
                                                   for m, v in r["models"].items()), flush=True)
        out["per_seed"].append(r)
    return out


def main() -> None:
    res = {"raw_values": raw_values()}
    for f, v in res["raw_values"].items():
        print(f"   {f}: shared values {v['shared_values']}", flush=True)
    for name in VIEWS:
        res[name] = run_view(name)
    record("e16_edgeiiot_format", {"features": list(FEATURES), "seeds": list(SEEDS), "models": list(MODELS),
                                   "views": list(VIEWS), "normalise": "numeric text read as numbers"},
           res, {"edgeiiot": data.dataset_hashes("edgeiiot")})


if __name__ == "__main__":
    main()
