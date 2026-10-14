"""EMBER 2018: read raw JSON features straight from the verified tarball.

Never extracts to disk: the tar.bz2 is streamed and only members named
`train_features_N.jsonl` / `test_features.jsonl` are read.

Two views per sample:
  raw_digest  SHA-1 of the canonical JSON of every raw feature (everything
              except sha256, md5, appeared, label, avclass). Two samples with
              equal digests have identical features under ANY vectorisation,
              including EMBER's official 2,381-dim one, so L1/L3 on the digest
              measure exact feature duplication without needing that code.
  numeric     a documented numeric view (~680 columns) built here from the raw
              fields, for L2 near-duplicates and L5 shortcuts. It is not the
              official EMBER vector (that code is AGPL-3.0 and is not used).

The parsed result is cached as parquet in data/processed/, keyed by the
tarball's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import DATA, dataset_hashes, verify_local

VIEW_VERSION = 1
MEMBER = re.compile(r"^(?:[\w.-]+/)?(train_features_\d+|test_features)\.jsonl$")
META = ("sha256", "md5", "appeared", "label", "avclass")
GENERAL = ("size", "vsize", "has_debug", "exports", "imports", "has_relocations", "has_resources",
           "has_signature", "has_tls", "symbols")
STRINGS = ("numstrings", "avlength", "printables", "entropy", "paths", "urls", "registry", "MZ")
OPTIONAL = ("major_image_version", "minor_image_version", "major_linker_version", "minor_linker_version",
            "major_operating_system_version", "minor_operating_system_version", "major_subsystem_version",
            "minor_subsystem_version", "sizeof_code", "sizeof_headers", "sizeof_heap_commit")
N_DATADIR = 15


def columns() -> list[str]:
    cols = [f"hist_{i}" for i in range(256)] + [f"bent_{i}" for i in range(256)]
    cols += [f"str_{k}" for k in STRINGS] + [f"str_pd_{i}" for i in range(96)]
    cols += [f"gen_{k}" for k in GENERAL]
    cols += ["hdr_coff_timestamp"] + [f"hdr_{k}" for k in OPTIONAL]
    cols += ["sec_n", "sec_n_zero_size", "sec_n_empty_name", "sec_entropy_mean", "sec_entropy_max",
             "sec_size_sum", "sec_vsize_sum", "imp_n_dll", "imp_n_func", "exp_n"]
    cols += [f"dd_{i}_{k}" for i in range(N_DATADIR) for k in ("size", "va")]
    return cols


CATEGORICAL = ("hdr_machine", "hdr_subsystem", "hdr_magic")


def _norm(v: list) -> list[float]:
    a = np.asarray(v, dtype=np.float64)
    s = a.sum()
    return (a / s).tolist() if s > 0 else a.tolist()


def numeric_row(r: dict) -> list[float]:
    st, ge, hd, se = r["strings"], r["general"], r["header"], r["section"]
    opt = hd.get("optional", {})
    pd_ = np.asarray(st["printabledist"], dtype=np.float64)
    row = _norm(r["histogram"]) + _norm(r["byteentropy"])
    row += [float(st[k]) for k in STRINGS]
    row += (pd_ / st["printables"]).tolist() if st["printables"] > 0 else pd_.tolist()
    row += [float(ge[k]) for k in GENERAL]
    row += [float(hd["coff"]["timestamp"])] + [float(opt.get(k, 0) or 0) for k in OPTIONAL]
    secs = se.get("sections", [])
    ent = [s["entropy"] for s in secs] or [0.0]
    row += [len(secs), sum(s["size"] == 0 for s in secs), sum(s["name"] == "" for s in secs),
            float(np.mean(ent)), float(np.max(ent)), float(sum(s["size"] for s in secs)),
            float(sum(s["vsize"] for s in secs)),
            len(r["imports"]), sum(len(v) for v in r["imports"].values()), len(r["exports"])]
    dds = r.get("datadirectories", [])[:N_DATADIR]
    for i in range(N_DATADIR):
        d = dds[i] if i < len(dds) else {"size": 0, "virtual_address": 0}
        row += [float(d["size"]), float(d["virtual_address"])]
    return row


def digest(r: dict) -> str:
    body = {k: v for k, v in r.items() if k not in META}
    return hashlib.sha1(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CHUNK = 50_000


def parse(tar_path: Path, limit: int | None = None) -> pd.DataFrame:
    """Rows are buffered as Python lists in chunks and packed into float32
    arrays (EMBER's own vectors are float32): 1M x ~680 Python floats would
    otherwise need ~20 GB."""
    cols = columns()
    meta, num, cat, chunks = [], [], [], []
    with tarfile.open(tar_path, mode="r|bz2") as tf:
        for m in tf:
            hit = MEMBER.match(m.name)
            if not (m.isfile() and hit) or ".." in m.name:
                continue
            split = "test" if hit.group(1) == "test_features" else "train"
            fh = tf.extractfile(m)
            for line in fh:
                r = json.loads(line)
                meta.append((split, r["sha256"], r["appeared"], int(r["label"]), r.get("avclass") or "", digest(r)))
                num.append(numeric_row(r))
                if len(num) == CHUNK:
                    chunks.append(np.asarray(num, dtype=np.float32))
                    num = []
                coff, opt = r["header"]["coff"], r["header"].get("optional", {})
                cat.append((coff.get("machine", ""), opt.get("subsystem", ""), opt.get("magic", "")))
                if limit and len(meta) >= limit:
                    break
    if num or not chunks:
        chunks.append(np.asarray(num, dtype=np.float32).reshape(-1, len(cols)))
    df = pd.DataFrame(np.vstack(chunks), columns=cols)
    df[list(CATEGORICAL)] = pd.DataFrame(cat, dtype="string")
    m = pd.DataFrame(meta, columns=["split", "sha256", "appeared", "label", "avclass", "raw_digest"])
    return pd.concat([m, df], axis=1)


def load_ember(refresh: bool = False) -> pd.DataFrame:
    (tar,) = verify_local("ember2018")
    sha = dataset_hashes("ember2018")["ember2018/ember_dataset_2018_2.tar.bz2"]
    cache = DATA / "processed" / f"ember2018_view{VIEW_VERSION}_{sha[:12]}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = parse(tar)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
