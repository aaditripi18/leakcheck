"""Ready-made clean splits: for each benchmark, which test rows leak from training.

    python -m experiments.clean_splits make kdd99 nslkdd ...     writes splits/<case>.json
    python -m experiments.clean_splits write kdd99 --out DIR     writes DIR/train.csv, DIR/test.csv (leaked rows dropped)

The training set is left as published. A test row is dropped if it has an exact copy, or a
near copy (RMS distance <= 0.01 in leakcheck's scaled space; Jaccard >= 0.8 for text), anywhere
in the *full* training set: `leakcheck clean` with conflicts kept, so the training rows do not
change. The cleaned split is then re-checked on every row (`leakcheck.clean.verify`), and the
check is stored with the list.

Row numbers are 0-based positions in the test set. For an official split that is the test
file as published (pinned by SHA-256 in data/registry.json), in file order. For a random
split it is the test side of the split leakcheck builds (scikit-learn's train_test_split,
stratified, seed 0, 20%); `write` rebuilds both files from the pinned data, so the list and
the files always agree. Where nearest-neighbour search over every row is out of reach (tens
of millions of rows), only exact copies are listed, and `near` says so.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import data
from experiments.e1_audit import PREP
from leakcheck.clean import CleanOptions, clean_split, verify
from leakcheck.explain import NEAR_EPS, TEXT_TAU

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "splits"
# exact copies only where every-row nearest-neighbour search is out of reach
EXACT_ONLY = {"cseids2018", "cseids2018:no_identifiers", "ciciot2023", "ember2018"}
# random splits: which column the split was stratified on (the prep functions' choice)
OFFICIAL = {"kdd99", "nslkdd", "unsw_nb15", "botiot", "ember2018"}
# How each dataset's rows are numbered in `drop_source` (0-based), so the lists can be used
# without leakcheck: official splits count rows of the test file; random splits count rows of
# the whole dataset, read from the pinned files as described.
SOURCE = {
    "kdd99": "rows of the test set `corrected` (registry kdd99_test), in file order",
    "nslkdd": "rows of KDDTest+.txt, in file order",
    "unsw_nb15": "rows of the official testing set, in file order",
    "botiot": "rows of UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv, in file order",
    "ember2018": "rows of EMBER 2018's test files (test_features.jsonl), in file order, unlabelled samples excluded",
    "cicids2017": "rows of the eight MachineLearningCVE CSVs read in capture order, Monday to Friday "
                  "(experiments/data.py, cic_members)",
    "cseids2018": "rows of the ten CSVs read in capture order (02-14 ... 03-02-2018), with the header lines "
                  "repeated inside some files skipped",
    "ciciot2023": "rows of the 169 part-*.csv files read in registry order",
    "uci_tuandromd": "rows of TUANDROMD.csv without its one entirely empty row (row 2,533)",
}
SINGLE_FILE = "rows of the pinned file, in file order"
TEST_ROWS_MAX = 1_000_000  # above this, test_source is not written; rebuild it as the JSON says


def split_for(case: str):
    import experiments.e1_audit as e1
    e1.LAST_TEST_ROWS = None
    name, _, view = case.partition(":")
    if name == "ember2018" and not view:
        view = "numeric"  # the official time split, on leakcheck's numeric view (exact and near copies alike)
    prep = PREP[name]()
    views = {v: s for v, s, _ in prep.splits}
    s = views[view] if view else prep.splits[0][1]
    rows = e1.LAST_TEST_ROWS
    if name in OFFICIAL:
        rows = np.arange(len(s.test), dtype=np.int64)
    if rows is not None and len(rows) != len(s.test):
        rows = None
    return prep, s, rows


def gaps(case: str) -> dict | None:
    """The measured leakage gap for this case, from the ledger (E2, E13 or E14)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import report as r
    runs = r.latest()
    e = r._e14(runs, case) or r._e13(runs, case)
    if e is not None:
        return {"full_gap_macro_f1": e["full_gap"], "source": f"E13/E14, 4 models x 3 seeds (ledger e14:/e13_difficulty:{case})"}
    d = runs.get(f"e2_retrain:{case}")
    if d is None:
        return None
    return {"source": f"E2, 4 models x 5 seeds (ledger e2_retrain:{case})", "summary": d["results"].get("summary")}


def use_gpu_search() -> bool:
    """LEAKCHECK_NEAR=torch: near.nearest becomes leakcheck.checks.near_torch.nearest (same answers,
    tests/test_near_torch.py), for clean_split and its re-check alike, so the giants get every row."""
    import os
    if os.environ.get("LEAKCHECK_NEAR") != "torch":
        return False
    from leakcheck.checks import near, near_torch
    near.nearest = near_torch.nearest
    return True


def make(case: str) -> dict:
    t0 = time.time()
    gpu = use_gpu_search()
    prep, s, src = split_for(case)
    exact_only = case in EXACT_ONLY and not gpu
    opts = CleanOptions(near_eps=None if exact_only else NEAR_EPS,
                        text_tau=None if exact_only else (TEXT_TAU if s.text else None), conflicts="keep")
    res = clean_split(s, opts)
    drop = sorted({i for v in res.removed.values() for i in v["test"]})
    check = verify(Split_like(s, res), opts)
    out = {"case": case, "split": prep.split_policy,
           "files": {d: data.dataset_hashes(d) for d in prep.datasets},
           "label": s.label, "excluded_columns": prep.notes.get("excluded", []),
           "near": ("exact copies only: nearest-neighbour search over every row is out of reach at this size"
                    if exact_only else {"eps": NEAR_EPS, "text_jaccard": TEXT_TAU if s.text else None}),
           "n_train": len(s.train), "n_test": len(s.test), "n_drop": len(drop),
           "n_clean_test": len(s.test) - len(drop),
           "by_reason": {k: len(v["test"]) for k, v in res.removed.items()},
           "recheck": check, "measured_gap": gaps(case), "seconds": round(time.time() - t0),
           "search": "near_torch (GPU shortlist, exact float64 distances)" if gpu else "near.nearest (CPU)",
           "source_numbering": SOURCE.get(case.partition(":")[0], SINGLE_FILE) if src is not None else None}
    if case.partition(":")[0] not in OFFICIAL and src is not None:
        import experiments.e1_audit as e1
        out["split_rule"] = ("test rows = sklearn.model_selection.train_test_split(numpy.arange(n), test_size=0.2, "
                             f"random_state=0, stratify=column {e1.LAST_STRATIFY!r})[1], n = rows of the whole "
                             "dataset numbered as source_numbering says")
        out["test_source_written"] = len(src) <= TEST_ROWS_MAX
    OUT.mkdir(exist_ok=True)
    slug = case.replace(":", "__")
    (OUT / f"{slug}.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    with gzip.open(OUT / f"{slug}.drop.txt.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(map(str, drop)) + "\n")
    if src is not None:
        with gzip.open(OUT / f"{slug}.drop_source.txt.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(map(str, sorted(src[drop].tolist()))) + "\n")
        if case.partition(":")[0] not in OFFICIAL and len(src) <= TEST_ROWS_MAX:
            with gzip.open(OUT / f"{slug}.test_source.txt.gz", "wt", encoding="utf-8") as f:
                f.write("\n".join(map(str, sorted(src.tolist()))) + "\n")
    print(f"{case}: drop {len(drop)}/{len(s.test)} test rows, re-check "
          f"{'ok' if all(v['ok'] for v in check.values()) else check}, {out['seconds']} s", flush=True)
    return out


def Split_like(s, res):
    from leakcheck.split import Split
    return Split(res.train, res.test, s.label, time=s.time, group=s.group, ignore=list(s.ignore), text=s.text)


def write(case: str, out_dir: str) -> None:
    prep, s, _ = split_for(case)
    slug = case.replace(":", "__")
    with gzip.open(OUT / f"{slug}.drop.txt.gz", "rt", encoding="utf-8") as f:
        drop = np.array([int(x) for x in f.read().split()], dtype=np.int64)
    keep = np.ones(len(s.test), bool)
    keep[drop] = False
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    s.train.to_csv(d / "train.csv", index=False)
    s.test[keep].to_csv(d / "test.csv", index=False)
    print(f"{case}: wrote {d / 'train.csv'} ({len(s.train)} rows) and {d / 'test.csv'} ({keep.sum()} rows)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        for c in sys.argv[2:]:
            make(c)
    elif cmd == "write":
        write(sys.argv[2], sys.argv[sys.argv.index("--out") + 1])
    else:
        sys.exit(__doc__)
