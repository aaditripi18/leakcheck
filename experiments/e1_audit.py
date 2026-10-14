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


def random_split(df: pd.DataFrame, stratify: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED, stratify=stratify)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


ALL = ("L1", "L2", "L3", "L4", "L5", "L6")


def prep_kdd99() -> Prepared:
    tr, te = data.load_kdd99("full"), data.load_kdd99("test")
    s = Split(tr.drop(columns=["label"]), te.drop(columns=["label"]), "is_attack")
    return Prepared([("all", s, ALL)], "official (kddcup.data vs corrected)", {"excluded": ["label (attack name)"]},
                    ("kdd99", "kdd99_test"))


def prep_nslkdd() -> Prepared:
    tr, te = data.load_nslkdd("train"), data.load_nslkdd("test")
    s = Split(tr.drop(columns=["label"]), te.drop(columns=["label"]), "is_attack", ignore=["difficulty"])
    return Prepared([("all", s, ALL)], "official (KDDTrain+ vs KDDTest+)",
                    {"excluded": ["label (attack name)", "difficulty"]}, ("nslkdd",))


def prep_unsw() -> Prepared:
    s = Split(data.load_unsw("train"), data.load_unsw("test"), "label", ignore=["id", "attack_cat"])
    return Prepared([("all", s, ALL)], "official (training-set vs testing-set)",
                    {"excluded": ["id (row number)", "attack_cat (attack name)"]}, ("unsw_nb15",))


def prep_cic() -> Prepared:
    df = data.load_cicids2017()
    df["is_attack"] = (df["Label"] != "BENIGN").astype("int8")
    tr, te = random_split(df, df["Label"])
    s = Split(tr.drop(columns=["Label"]), te.drop(columns=["Label"]), "is_attack", time="capture_order")
    return Prepared([("all", s, ALL)], f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} by Label, seed {SEED}",
                    {"excluded": ["Label (attack name)"], "time": "capture_order = index of the capture file (Mon..Fri)"},
                    ("cicids2017",))


def prep_uci_phishing() -> Prepared:
    df = data.load_uci_phishing()
    tr, te = random_split(df, df["Result"])
    return Prepared([("all", Split(tr, te, "Result"), ALL)],
                    f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {SEED}", {}, ("uci_phishing",))


def prep_phiusiil() -> Prepared:
    df = data.load_phiusiil()
    tr, te = random_split(df, df["label"])
    s = Split(tr, te, "label", ignore=["FILENAME"], text="URL")
    return Prepared([("all", s, ALL)], f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {SEED}",
                    {"excluded": ["FILENAME (row identifier)"], "text": "URL"}, ("phiusiil",))


def prep_sms() -> Prepared:
    df = data.load_sms_spam()
    tr, te = random_split(df, df["label"])
    s = Split(tr, te, "label", text="message")
    return Prepared([("all", s, ALL)], f"stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, seed {SEED}",
                    {"text": "message"}, ("sms_spam",))


def prep_ember() -> Prepared:
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


PREP = {
    "kdd99": prep_kdd99, "nslkdd": prep_nslkdd, "unsw_nb15": prep_unsw, "cicids2017": prep_cic,
    "ember2018": prep_ember, "uci_phishing": prep_uci_phishing, "phiusiil": prep_phiusiil, "sms_spam": prep_sms,
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
