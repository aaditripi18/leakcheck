"""Dataset registry, verified downloads and loaders.

Every file in data/registry.json has a pinned SHA-256. `fetch` downloads to a
temporary name, hashes, and only moves the file into place if the hash
matches; a mismatch deletes the download and raises. data/manifest.json
records what is on disk (hash, size, source URL, time). Raw data lives in
data/raw/ and is never committed.

Archives are never extracted to disk: loaders read the members they need
straight out of the zip, by exact name.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
REGISTRY = DATA / "registry.json"
MANIFEST = DATA / "manifest.json"


class HashMismatch(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path = REGISTRY) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fetch(name: str, registry: dict | None = None, raw: Path = RAW, manifest: Path = MANIFEST,
          pin: bool = False) -> list[Path]:
    """Download every file of dataset `name`, verifying SHA-256.

    pin=True accepts an unpinned file (sha256 null) and returns its hash so it
    can be written into the registry. Pinned files must always match.
    """
    registry = registry if registry is not None else load_registry()
    entry = registry[name]
    man = _read_manifest(manifest)
    paths = []
    for f in entry["files"]:
        dest = raw / f["path"]
        expected = f.get("sha256")
        if expected is None and not pin:
            raise HashMismatch(f"{f['path']}: no pinned sha256 in registry (run with --pin once, then commit)")
        if dest.exists() and expected and sha256_file(dest) == expected:
            paths.append(dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        req = urllib.request.Request(f["url"], headers={"User-Agent": "leakcheck-fetch"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        got = sha256_file(tmp)
        if expected and got != expected:
            tmp.unlink()
            raise HashMismatch(f"{f['path']}: expected {expected}, got {got}")
        os.replace(tmp, dest)
        if expected is None:
            f["sha256"] = got
        man[f["path"]] = {
            "dataset": name,
            "sha256": got,
            "size": dest.stat().st_size,
            "url": f["url"],
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        paths.append(dest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def verify_local(name: str, registry: dict | None = None, raw: Path = RAW) -> list[Path]:
    """Paths of a dataset's files, after checking every file's hash."""
    registry = registry if registry is not None else load_registry()
    files = registry[name]["files"]
    for f in files:
        p = raw / f["path"]
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run python scripts/fetch.py {name}")
        if sha256_file(p) != f["sha256"]:
            raise HashMismatch(f"{p} does not match the registry hash")
    return [raw / f["path"] for f in files]


def dataset_hashes(name: str, registry: dict | None = None) -> dict[str, str]:
    registry = registry if registry is not None else load_registry()
    return {f["path"]: f["sha256"] for f in registry[name]["files"]}


# --- KDD Cup 99 and NSL-KDD -------------------------------------------------------

KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
KDD_CATEGORICAL = ["protocol_type", "service", "flag"]


def _finish_kdd(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["label"].astype("string").str.rstrip(".")  # KDD99 writes "normal."
    for c in KDD_CATEGORICAL:
        df[c] = df[c].astype("string")
    df["is_attack"] = (df["label"] != "normal").astype("int8")
    return df


def _read_kdd(buf, extra: list[str]) -> pd.DataFrame:
    names = KDD_FEATURES + ["label"] + extra
    dtypes = {c: "string" for c in KDD_CATEGORICAL + ["label"]}
    return _finish_kdd(pd.read_csv(buf, header=None, names=names, dtype=dtypes))


def _read_hf(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in sorted(paths)], ignore_index=True)
    df = df.rename(columns={"result": "label"})
    if list(df.columns) != KDD_FEATURES + ["label"]:
        raise ValueError(f"unexpected columns in {paths[0].parent}: {list(df.columns)}")
    return _finish_kdd(df)


def _zip_member(zf: zipfile.ZipFile, wanted: str) -> str:
    """The member named exactly `wanted` (no path components, so no traversal)."""
    hits = [n for n in zf.namelist() if n == wanted]
    if len(hits) != 1:
        raise KeyError(f"{wanted!r} not found exactly once in archive")
    return hits[0]


def load_kdd99(part: str) -> pd.DataFrame:
    """part: 'full' (original kddcup.data.gz, 4,898,431 rows), 'test' (labelled
    'corrected' test, 311,029 rows) or 'hf_train' (the Hugging Face copy of the
    full training set, only for validating that upload)."""
    if part == "full":
        (path,) = verify_local("kdd99")
        with gzip.open(path) as g:
            return _read_kdd(g, [])
    if part == "test":
        return _read_hf(verify_local("kdd99_test"))
    if part == "hf_train":
        return _read_hf(verify_local("kdd99_hf_train"))
    raise ValueError(part)


def load_nslkdd(part: str) -> pd.DataFrame:
    """part: 'train' (KDDTrain+, 125,973) or 'test' (KDDTest+, 22,544)."""
    member = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}[part]
    (path,) = verify_local("nslkdd")
    with zipfile.ZipFile(path) as zf, zf.open(_zip_member(zf, member)) as f:
        return _read_kdd(f, ["difficulty"])
