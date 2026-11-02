"""E0-src (phase 3): check each current benchmark against an independent reference.

  python -m experiments.e0_sources_phase3 [edgeiiot cseids2018 ciciot2023]

References (read 2026-11-02; details in experiments/published_counts.py):
  Edge-IIoTset     the authors' own release; totals agree with later papers'
                   descriptions (about 2.2 million rows, 1.6 million normal).
  CSE-CIC-IDS2018  every CSV must equal the official S3 object byte for byte
                   (recomputed multipart ETag, experiments/verify_s3_etag.py);
                   label counts vs Sensors 22(19):7409, Table 3.
  CIC-IoT2023      label counts vs Neto et al., Sensors 23(13):5941, Table 3
                   (the paper of the dataset's authors).
A dataset passes only if every published count is matched exactly.
"""
from __future__ import annotations

import json
import sys

from experiments import data
from experiments.ledger import record
from experiments.published_counts import CICIOT2023, CSECICIDS2018


def compare(got: dict, want: dict) -> dict:
    diff = {k: {"published": want.get(k), "file": got.get(k)} for k in set(got) | set(want)
            if got.get(k) != want.get(k)}
    return {"counts": got, "total": sum(got.values()), "published_total": sum(want.values()),
            "differences": diff, "matches_published": not diff}


def check_edgeiiot() -> dict:
    df = data.cached(data.load_edgeiiot)
    counts = {str(k): int(v) for k, v in df["Attack_type"].value_counts().items()}
    return {"counts": counts, "total": int(len(df)), "normal": counts.get("Normal"),
            "attack_types": len(counts) - 1}


def check_cseids2018() -> dict:
    from experiments import verify_s3_etag
    etag_ok = verify_s3_etag.main() == 0
    df = data.cached(data.load_cseids2018)
    res = compare({str(k): int(v) for k, v in df["Label"].value_counts().items()}, CSECICIDS2018)
    res["identical_to_official_s3"] = etag_ok
    return res


def check_ciciot2023() -> dict:
    df = data.cached(data.load_ciciot2023)
    return compare({str(k): int(v) for k, v in df["label"].value_counts().items()}, CICIOT2023)


CHECKS = {"edgeiiot": check_edgeiiot, "cseids2018": check_cseids2018, "ciciot2023": check_ciciot2023}


def main(names: list[str]) -> None:
    for name in names or list(CHECKS):
        res = CHECKS[name]()
        record("e0_sources_phase3:" + name, {}, res, {name: data.dataset_hashes(name)})
        print(name, json.dumps({k: v for k, v in res.items() if k != "counts"}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
