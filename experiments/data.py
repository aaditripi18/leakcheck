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
import http.client
import json
import os
import time
import urllib.error
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


_CACHE: dict = {}


def cached(loader, *args) -> pd.DataFrame:
    """Load once per process; hand out copies so callers may modify them."""
    key = (loader.__name__, args)
    if key not in _CACHE:
        _CACHE[key] = loader(*args)
    return _CACHE[key].copy()


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


RETRIES = 30


def download(url: str, tmp: Path, retries: int = RETRIES, wait: float = 2.0) -> None:
    """Stream `url` into `tmp`, resuming with HTTP Range after a dropped connection.

    CDNs sometimes close the connection mid-file without an error, so the
    file is only considered complete when its size reaches the total the
    server announced. The SHA-256 check after this is still the final word.
    """
    total = None
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        if total is not None and have >= total:
            return
        headers = {"User-Agent": "leakcheck-fetch"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                status = getattr(r, "status", None) or 200
                if status == 206:
                    rng = r.headers.get("Content-Range", "")
                    total = int(rng.rsplit("/", 1)[1]) if "/" in rng and not rng.endswith("*") else total
                    mode = "ab"
                else:  # 200: full body (server ignored Range, or first request)
                    length = r.headers.get("Content-Length")
                    total = int(length) if length else None
                    mode = "wb"
                with open(tmp, mode) as out:
                    while chunk := r.read(1 << 20):
                        out.write(chunk)
            if total is None or tmp.stat().st_size >= total:
                return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # range starts at the end: already complete
                return
            if e.code < 500:
                raise
        except (urllib.error.URLError, OSError, http.client.HTTPException):
            pass
        time.sleep(min(wait * 2**attempt, 60))
    raise RuntimeError(f"download of {url} incomplete after {retries} attempts")


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
        if entry.get("manual_download"):
            raise HashMismatch(f"{f['path']}: download it by hand from {f['url']} into {dest.parent} "
                               f"(the share serves a page, not the file); it must have SHA-256 {expected}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        download(f["url"], tmp)
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


# --- Week 4 datasets ----------------------------------------------------------------


def load_unsw(part: str) -> pd.DataFrame:
    """part: 'train' (UNSW_NB15_training-set.csv, 175,341) or 'test' (testing-set, 82,332)."""
    tr, te = verify_local("unsw_nb15")
    df = pd.read_csv(tr if part == "train" else te, encoding="utf-8-sig")
    for c in ("proto", "service", "state", "attack_cat"):
        df[c] = df[c].astype("string").str.strip()
    return df


# Capture order of the CIC-IDS2017 MachineLearningCSV files (Mon 3 Jul - Fri 7 Jul 2017).
CIC_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def cic_members(zf: zipfile.ZipFile) -> dict[str, str]:
    """File name -> archive member for the 8 CSVs (the zip nests them in a folder)."""
    out = {}
    for f in CIC_FILES:
        hits = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == f and ".." not in n]
        if len(hits) != 1:
            raise KeyError(f"{f!r} not found exactly once in archive")
        out[f] = hits[0]
    return out


def load_cicids2017() -> pd.DataFrame:
    """All 8 files concatenated, column names stripped, plus `capture_order` (0-7)."""
    (path,) = verify_local("cicids2017")
    frames = []
    with zipfile.ZipFile(path) as zf:
        for i, (f, m) in enumerate(cic_members(zf).items()):
            with zf.open(m) as fh:
                d = pd.read_csv(fh, encoding="latin-1", low_memory=False)
            d.columns = [c.strip() for c in d.columns]
            d["capture_order"] = i
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # "Web Attack <dash> XSS": the dash byte is not valid in any one encoding across copies
    df["Label"] = df["Label"].astype("string").str.replace(r"[^\x20-\x7e]+", " ", regex=True)
    df["Label"] = df["Label"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def load_uci_phishing() -> pd.DataFrame:
    (path,) = verify_local("uci_phishing")
    with zipfile.ZipFile(path) as zf:
        text = zf.read(_zip_member(zf, "Training Dataset.arff")).decode("utf-8")
    names, rows, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        if in_data:
            rows.append([int(v) for v in s.split(",")])
        elif s.lower().startswith("@attribute"):
            names.append(s.split()[1])
        elif s.lower().startswith("@data"):
            in_data = True
    return pd.DataFrame(rows, columns=names)


def load_phiusiil() -> pd.DataFrame:
    (path,) = verify_local("phiusiil")
    with zipfile.ZipFile(path) as zf, zf.open(_zip_member(zf, "PhiUSIIL_Phishing_URL_Dataset.csv")) as f:
        return pd.read_csv(f, encoding="utf-8-sig", low_memory=False)


def load_sms_spam() -> pd.DataFrame:
    (path,) = verify_local("sms_spam")
    with zipfile.ZipFile(path) as zf:
        text = zf.read(_zip_member(zf, "SMSSpamCollection")).decode("utf-8")
    rows = [line.split("\t", 1) for line in text.splitlines() if line]
    return pd.DataFrame(rows, columns=["label", "message"])
