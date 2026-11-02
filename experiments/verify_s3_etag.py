"""Prove local copies of the CSE-CIC-IDS2018 CSVs are the official files.

The official bucket (s3://cse-cic-ids2018, "Processed Traffic Data for ML
Algorithms") publishes every object's size and ETag. For a multipart upload the
ETag is md5(md5(part 1) + ... + md5(part N)) + "-N", so recomputing it from a
local file, with the part size the uploader used, proves the bytes are
identical. The files used here came from a Kaggle copy (much faster to fetch);
this check is what makes them the official files.

    python -m experiments.verify_s3_etag
"""
from __future__ import annotations

import hashlib
import re
import sys
import urllib.request

from experiments import data

LIST_URL = ("https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/?list-type=2&prefix="
            "Processed%20Traffic%20Data%20for%20ML%20Algorithms/")
MIB = 1 << 20
PART_SIZES = (8 * MIB, 16 * MIB, 5 * MIB, 32 * MIB, 64 * MIB)

# official object name -> the Kaggle copy's file name (solarmainframe/ids-intrusion-csv)
LOCAL = {
    "Wednesday-14-02-2018": "02-14-2018.csv", "Thursday-15-02-2018": "02-15-2018.csv",
    "Friday-16-02-2018": "02-16-2018.csv", "Thuesday-20-02-2018": "02-20-2018.csv",
    "Wednesday-21-02-2018": "02-21-2018.csv", "Thursday-22-02-2018": "02-22-2018.csv",
    "Friday-23-02-2018": "02-23-2018.csv", "Wednesday-28-02-2018": "02-28-2018.csv",
    "Thursday-01-03-2018": "03-01-2018.csv", "Friday-02-03-2018": "03-02-2018.csv",
}


def official() -> dict[str, tuple[int, str]]:
    with urllib.request.urlopen(LIST_URL, timeout=60) as r:
        xml = r.read().decode()
    out = {}
    for key, etag, size in re.findall(r"<Key>(.*?)</Key>.*?<ETag>(.*?)</ETag><Size>(\d+)</Size>", xml):
        if size != "0":
            out[key.rsplit("/", 1)[-1].split("_")[0]] = (int(size), etag.replace("&quot;", ""))
    return out


def multipart_etag(path, part_size: int) -> str:
    digests, n = [], 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(part_size), b""):
            digests.append(hashlib.md5(chunk).digest())
            n += 1
    if n == 1:
        return digests[0].hex()
    return hashlib.md5(b"".join(digests)).hexdigest() + f"-{n}"


def main() -> int:
    ok = True
    for day, (size, etag) in sorted(official().items()):
        path = data.RAW / "cseids2018" / LOCAL[day]
        if not path.exists() or path.stat().st_size != size:
            print(f"{day}: missing or wrong size")
            ok = False
            continue
        parts = int(etag.split("-")[1]) if "-" in etag else 1
        sizes = [p for p in PART_SIZES if -(-size // p) == parts] or list(PART_SIZES)
        match = next((p for p in sizes if multipart_etag(path, p) == etag), None)
        print(f"{day}: {'identical to the official object' if match else 'DIFFERENT'} "
              f"(ETag {etag}{f', {match // MIB} MiB parts' if match else ''}); sha256 {data.sha256_file(path)}")
        ok &= match is not None
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
