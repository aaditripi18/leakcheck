"""UNSW-NB15 raw records (UNSW-NB15_1..4.csv) for E7 (PLAN.md, phase 2).

The authors' four CSV files have no header; column names come from
NUSW-NB15_features.csv (stripped: the file spells one "ct_src_ ltm"). Known
inconsistencies in the raw files, normalised here and nowhere else:

  attack_cat  empty for normal traffic -> "Normal"; padded with spaces; and
              "Backdoors" / "Backdoor" both used -> "Backdoor"
  sport/dsport some values are hexadecimal ("0x000b") or "-": parsed as
              numbers where possible, else missing (they are not model features)
  ct_ftp_cmd  blank strings in some rows -> missing
  boundaries  each file ends with the record the next file begins with
              (2,540,047 lines; the published total is 2,540,044): the repeated
              first line of files 2-4 is dropped when it equals the previous
              file's last line, and only then
  BOM         UNSW-NB15_1.csv starts with a UTF-8 byte-order mark

Features follow the authors' own training/testing partition: everything except
srcip, sport, dstip, dsport, stime and ltime (addresses and timestamps), with
attack_cat excluded from the features and the binary Label as the label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments import data

DIR = data.ROOT / "data" / "raw" / "unsw_raw_official"
FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
NOT_FEATURES = ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "attack_cat"]
FAMILIES = ["Fuzzers", "Analysis", "Backdoor", "DoS", "Exploits", "Generic", "Reconnaissance", "Shellcode", "Worms"]


def names(directory: Path = DIR) -> list[str]:
    feats = pd.read_csv(directory / "NUSW-NB15_features.csv", encoding="latin-1")
    return [str(n).strip().replace(" ", "").lower() for n in feats["Name"]]


def _port(x: pd.Series) -> pd.Series:
    s = x.astype("string").str.strip()
    hexa = s.str.match(r"^0x[0-9a-fA-F]+$", na=False)
    out = pd.to_numeric(s.where(~hexa), errors="coerce")
    out[hexa] = s[hexa].map(lambda v: int(v, 16)).astype("float64")
    return out


def load(directory: Path = DIR) -> pd.DataFrame:
    cols = names(directory)
    parts = []
    for f in FILES:
        part = pd.read_csv(directory / f, header=None, names=cols, low_memory=False, encoding="utf-8-sig",
                           dtype=str, keep_default_na=False)
        if parts and part.iloc[0].equals(parts[-1].iloc[-1]):
            part = part.iloc[1:]  # the boundary record repeated from the previous file
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    text_cols = ("srcip", "dstip", "proto", "state", "service", "attack_cat", "sport", "dsport", "ct_ftp_cmd")
    df = raw.replace({"": None})
    for c in df.columns:
        if c not in text_cols:
            df[c] = pd.to_numeric(df[c], errors="raise")  # every other column must be numeric
    cat = df["attack_cat"].astype("string").str.strip()
    cat = cat.where(cat.notna() & (cat != ""), "Normal").replace({"Backdoors": "Backdoor"})
    df["attack_cat"] = cat
    for c in ("sport", "dsport"):
        df[c] = _port(df[c])
    df["ct_ftp_cmd"] = pd.to_numeric(df["ct_ftp_cmd"].astype("string").str.strip(), errors="coerce")
    for c in ("proto", "state", "service"):
        df[c] = df[c].astype("string").str.strip()
    df["label"] = df["label"].astype("int8")
    return df


def provenance(df: pd.DataFrame) -> dict:
    """The E7 gate: record count, feature count and the nine families."""
    fams = df["attack_cat"].value_counts().to_dict()
    return {
        "records": int(len(df)), "records_expected": 2_540_044,
        "columns": int(df.shape[1]), "columns_expected": 49,
        "families": {k: int(v) for k, v in fams.items()},
        "missing_families": [f for f in FAMILIES if f not in fams],
        "unexpected_categories": sorted(set(fams) - set(FAMILIES) - {"Normal"}),
        "label_vs_category_agree": bool(((df["attack_cat"] == "Normal") == (df["label"] == 0)).all()),
        "stime_range": [int(df["stime"].min()), int(df["stime"].max())],
        "passes": bool(len(df) == 2_540_044 and df.shape[1] == 49 and all(f in fams for f in FAMILIES)),
    }


def features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NOT_FEATURES + ["label"]]


if __name__ == "__main__":
    import json
    d = load()
    print(json.dumps(provenance(d), indent=1))
    _ = np
