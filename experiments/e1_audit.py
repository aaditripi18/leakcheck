"""E1: run every leakcheck check on every dataset, one ledger line per dataset.

Split policy (fixed before any results were seen):
  official split where the dataset ships one (KDD99, NSL-KDD, UNSW-NB15, EMBER);
  otherwise a stratified random 80/20 split, seed 0 (CIC-IDS2017, UCI Phishing,
  PhiUSIIL, SMS Spam) -- the most common protocol for these datasets.
Task: binary (attack/malicious/phishing/spam vs benign). Columns that encode the
label (attack category, attack name) and row identifiers are excluded; every
exclusion is listed in the config below and in the ledger line.

  python -m experiments.e1_audit                 # all datasets
  python -m experiments.e1_audit sms_spam unsw_nb15
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiments import data
from experiments.ledger import record
from leakcheck import Split, scan
from leakcheck.findings import Report

SEED = 0
TEST_SIZE = 0.2
AUDIT_CSV = data.ROOT / "results" / "audit.csv"


@dataclass
class Prepared:
    splits: list[tuple[str, Split, tuple[str, ...]]]  # (view name, split, checks)
    split_policy: str
    notes: dict = field(default_factory=dict)
    datasets: tuple[str, ...] = ()


def random_split(df: pd.DataFrame, stratify: pd.Series, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=stratify)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


ALL = ("L1", "L2", "L3", "L4", "L5", "L6")


def prep_kdd99(seed: int = SEED) -> Prepared:
    tr, te = data.load_kdd99("full"), data.load_kdd99("test")
    s = Split(tr.drop(columns=["label"]), te.drop(columns=["label"]), "is_attack")
    return Prepared([("all", s, ALL)], "official (kddcup.data vs corrected)", {"excluded": ["label (attack name)"]},
                    ("kdd99", "kdd99_test"))


def prep_nslkdd(seed: int = SEED) -> Prepared:
    tr, te = data.load_nslkdd("train"), data.load_nslkdd("test")
    s = Split(tr.drop(columns=["label"]), te.drop(columns=["label"]), "is_attack", ignore=["difficulty"])
    return Prepared([("all", s, ALL)], "official (KDDTrain+ vs KDDTest+)",
                    {"excluded": ["label (attack name)", "difficulty"]}, ("nslkdd",))


def prep_unsw(seed: int = SEED) -> Prepared:
    s = Split(data.load_unsw("train"), data.load_unsw("test"), "label", ignore=["id", "attack_cat"])
    return Prepared([("all", s, ALL)], "official (training-set vs testing-set)",
                    {"excluded": ["id (row number)", "attack_cat (attack name)"]}, ("unsw_nb15",))


def prep_cic(seed: int = SEED) -> Prepared:
    df = data.cached(data.load_cicids2017)
    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    tr, te = random_split(df, df["Label"], seed)
    s = Split(tr.drop(columns=["Label"]), te.drop(columns=["Label"]), "is_attack", time="capture_order")
    return Prepared([("all", s, ALL)], f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} by Label, seed {seed}",
                    {"excluded": ["Label (attack name)"], "time": "capture_order = index of the capture file (Mon..Fri)"},
                    ("cicids2017",))


def prep_uci_phishing(seed: int = SEED) -> Prepared:
    df = data.cached(data.load_uci_phishing)
    tr, te = random_split(df, df["Result"], seed)
    return Prepared([("all", Split(tr, te, "Result"), ALL)],
                    f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {seed}", {}, ("uci_phishing",))


def prep_phiusiil(seed: int = SEED) -> Prepared:
    df = data.cached(data.load_phiusiil)
    tr, te = random_split(df, df["label"], seed)
    s = Split(tr, te, "label", ignore=["FILENAME"], text="URL")
    return Prepared([("all", s, ALL)], f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {seed}",
                    {"excluded": ["FILENAME (row identifier)"], "text": "URL"}, ("phiusiil",))


def prep_sms(seed: int = SEED) -> Prepared:
    df = data.cached(data.load_sms_spam)
    tr, te = random_split(df, df["label"], seed)
    s = Split(tr, te, "label", text="message")
    return Prepared([("all", s, ALL)], f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {seed}",
                    {"text": "message"}, ("sms_spam",))


def big_split(df, label_col: str, target: str, seed: int, **kw) -> Split:
    """E1's stratified random split for tens of millions of rows: split by row index,
    build each side once and drop the label in place (no extra full copies)."""
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, random_state=seed,
                                      stratify=df[label_col])
    # same rows in the same order as random_split (train_test_split on the frame itself)
    tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    for side in (tr, te):
        del side[label_col]
    return Split(tr, te, target, **kw)


def views(s: Split) -> list:
    """The pre-registered view, plus no_identifiers when L5 flags any identifier."""
    out = [("all", s, ALL)]
    nid = without_identifiers(s)
    if nid is not s:
        out.append(("no_identifiers", nid, ALL))
    return out


def without_identifiers(s: Split) -> Split:
    """The same split without the columns leakcheck's L5 identifier check flags
    (phase 3 amendment: a post hoc second view of the current benchmarks)."""
    from leakcheck.checks.shortcut import check_identifiers

    drop = [c["feature"] for c in check_identifiers(s).details["columns"]]
    if not drop:
        return s
    return Split(s.train.drop(columns=drop), s.test.drop(columns=drop), s.label, time=s.time, group=s.group,
                 text=s.text, ignore=[c for c in s.ignore if c not in drop])


def prep_edgeiiot(seed: int = SEED) -> Prepared:
    df = data.cached(data.load_edgeiiot)
    tr, te = random_split(df, df["Attack_type"], seed)
    s = Split(tr.drop(columns=["Attack_type"]), te.drop(columns=["Attack_type"]), "Attack_label")
    return Prepared([("all", s, ALL), ("no_identifiers", without_identifiers(s), ALL)],
                    f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} by Attack_type, seed {seed}",
                    {"excluded": ["Attack_type (attack name)"],
                     "kept": "every distributed feature, identifiers included (frame.time, IPs, ports)"},
                    ("edgeiiot",))


def prep_cseids2018(seed: int = SEED) -> Prepared:
    df = data.load_cseids2018()  # not data.cached: 16M rows, no second copy
    df["is_attack"] = (df["Label"] != "Benign").astype("int8")
    s = big_split(df, "Label", "is_attack", seed, time="capture_order")
    del df
    return Prepared(views(s), f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} by Label, seed {seed}",
                    {"excluded": ["Label (attack name)"], "time": "capture_order = index of the capture day (10 files)",
                     "kept": "every distributed feature, identifiers included (Timestamp; Flow ID and IPs on 20-02)"},
                    ("cseids2018",))


def prep_ciciot2023(seed: int = SEED) -> Prepared:
    df = data.load_ciciot2023()  # not data.cached: 47M rows, no second copy
    df["is_attack"] = (df["label"] != "BenignTraffic").astype("int8")
    s = big_split(df, "label", "is_attack", seed)
    del df
    return Prepared(views(s), f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} by label, seed {seed}",
                    {"excluded": ["label (attack name)"]}, ("ciciot2023",))


def prep_ember(seed: int = SEED) -> Prepared:
    from experiments.ember import CATEGORICAL, columns, load_ember

    df = load_ember()
    df = df[df["label"] != -1]  # unlabelled training samples are not part of the supervised task
    df = df.assign(family=np.where(df["avclass"] != "", df["avclass"], "benign:" + df["sha256"]))
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    digest = Split(tr[["raw_digest", "label"]], te[["raw_digest", "label"]], "label")
    numeric_cols = columns() + list(CATEGORICAL)
    num = Split(tr[numeric_cols + ["label", "appeared", "family"]], te[numeric_cols + ["label", "appeared", "family"]],
                "label", time="appeared", group="family")
    return Prepared(
        [("raw_digest", digest, ("L1", "L3")), ("numeric", num, ("L2", "L4", "L5", "L6"))],
        "official temporal (train Jan-Oct 2018, test Nov-Dec 2018); unlabelled samples dropped",
        {"L1/L3": "on the digest of all raw features (= duplicates under any vectorisation)",
         "L2/L5": "on leakcheck's own ~680-column numeric view, not the official EMBER vector",
         "L6": "group = AVClass family for malware; each benign file is its own group"},
        ("ember2018",),
    )


def prep_ember_official(seed: int = SEED) -> Prepared:
    """E11: the official temporal split, on EMBER's own 2,381-dimension vectors."""
    from experiments.ember import load_ember
    from experiments.ember_official import aligned

    df = load_ember()
    df = df[df["label"] != -1]
    df = df.assign(family=np.where(df["avclass"] != "", df["avclass"], "benign:" + df["sha256"]))
    v = aligned(df.reset_index(drop=True))
    tr = v[v["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    te = v[v["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    s = Split(tr, te, "label", time="appeared", group="family")
    return Prepared([("official_vector", s, ALL)],
                    "official temporal (train Jan-Oct 2018, test Nov-Dec 2018); unlabelled samples dropped",
                    {"features": "EMBER's own feature_version 2 vectors (2,381 floats), made by EMBER's code outside "
                                 "this repository and verified against EMBER's published model (ROC AUC 0.996429)",
                     "L6": "group = AVClass family for malware; each benign file is its own group"},
                    ("ember2018",))


PREP = {
    "kdd99": prep_kdd99, "nslkdd": prep_nslkdd, "unsw_nb15": prep_unsw, "cicids2017": prep_cic,
    "ember2018": prep_ember, "uci_phishing": prep_uci_phishing, "phiusiil": prep_phiusiil, "sms_spam": prep_sms,
    "edgeiiot": prep_edgeiiot, "cseids2018": prep_cseids2018, "ciciot2023": prep_ciciot2023,
    "ember2018_official": prep_ember_official,
}


def summarize(rep: Report) -> dict:
    out = {}
    for f in rep.findings:
        d = {"value": f.value, "severity": f.severity}
        for k in ("n_hit", "n_near", "n_groups", "n_rows", "ci95", "vs_train_baseline", "identical_rate",
                  "per_class", "flagged", "columns", "after_test_median", "after_test_end", "shared_groups"):
            if k in f.details:
                d[k] = f.details[k]
        if f.code == "L5.shortcut":
            d["top"] = f.details["top"][:5]
        out[f.code] = d
    return out


def audit(name: str) -> dict:
    t0 = time.time()
    prep = PREP[name]()
    views = {}
    for view, split, checks in prep.splits:
        rep = scan(split, checks=checks, seed=SEED)
        views[view] = {"meta": rep.meta, "findings": summarize(rep)}
    results = {"split_policy": prep.split_policy, "notes": prep.notes, "views": views,
               "seconds": round(time.time() - t0, 1)}
    datasets = {d: data.dataset_hashes(d) for d in prep.datasets}
    record("e1_audit:" + name, {"seed": SEED, "test_size": TEST_SIZE}, results, datasets)
    return results


def flat_row(name: str, res: dict) -> dict:
    f = {}
    for v in res["views"].values():
        f.update(v["findings"])
    meta = next(iter(res["views"].values()))["meta"]

    def val(code, key="value"):
        return f[code][key] if code in f and key in f[code] else ""

    top = f.get("L5.shortcut", {}).get("top", [{}])
    return {
        "dataset": name, "split": res["split_policy"], "n_train": meta["n_train"], "n_test": meta["n_test"],
        "within_train": val("L1.within_train"), "within_test": val("L1.within_test"), "cross_exact": val("L1.cross"),
        "near": val("L2.near"), "near_vs_baseline": val("L2.near", "vs_train_baseline"), "near_text": val("L2.near_text"),
        "conflict": val("L3.conflict"), "ceiling": val("L3.ceiling"), "cross_contradict": val("L3.cross_contradict"),
        "train_after_test_start": val("L4.after_test_start"), "time_shortcut": val("L4.time_shortcut"),
        "best_single_feature": top[0].get("feature", "") if top else "", "best_single_feature_bacc": val("L5.shortcut"),
        "identifier_columns": val("L5.identifier"), "group_overlap": val("L6.group"), "seconds": res["seconds"],
    }


def main(names: list[str]) -> None:
    rows = []
    for name in names or list(PREP):
        print(f"== {name}", flush=True)
        res = audit(name)
        rows.append(flat_row(name, res))
        print(f"   done in {res['seconds']} s", flush=True)
    existing = {}
    if AUDIT_CSV.exists():
        existing = {r["dataset"]: r for r in csv.DictReader(open(AUDIT_CSV, encoding="utf-8"))}
    existing.update({r["dataset"]: r for r in rows})
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for name in PREP:
            if name in existing:
                w.writerow(existing[name])


if __name__ == "__main__":
    main(sys.argv[1:])
