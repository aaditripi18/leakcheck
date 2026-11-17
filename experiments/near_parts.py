"""The near search of the largest clean splits, in parts and keyed by row content.

A test row's nearest-neighbour distance depends only on the row and the training side: the scaling
is fitted on the training rows (near.embed), and every row is searched on its own. So the search
can be split into parts run as separate jobs, rows with the same content are searched once, and the
re-check of the cleaned split (same training side, a subset of the test rows) reads the distances
already found instead of searching again.

Distances are stored per training side (its row count and the SHA-256 of its row hashes) and per
test-row hash (Split.test_hash: every feature column, canonicalised), in .npz files in a cache
directory. Two rows with the same hash have the same features, so the same distance.

    LEAKCHECK_NEAR_CACHE=<dir>   read and write the cache
    LEAKCHECK_NEAR_PART=i/n      compute part i of n of the first search's missing rows, save it, and
                                 stop the process (exit 0); without it, the search reads the cache
                                 and computes only what is missing
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np


class PartDone(SystemExit):
    """A part was computed and saved; the process ends here (exit status 0)."""

    def __init__(self):
        super().__init__(0)


def train_key(split) -> str:
    h = hashlib.sha256(np.ascontiguousarray(split.train_hash).tobytes()).hexdigest()[:20]
    return f"{len(split.train)}-{h}"


def load(cache: Path, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hashes sorted, distances, nearest training rows) of every file stored for this training side."""
    hs, ds, ns = [], [], []
    for f in sorted(cache.glob(f"{key}--*.npz")):
        with np.load(f) as z:
            if str(z["key"]) != key:
                raise ValueError(f"{f.name}: stored for another training side")
            hs.append(z["hash"]), ds.append(z["dist"]), ns.append(z["nn"])
    if not hs:
        return np.empty(0, np.uint64), np.empty(0), np.empty(0, np.int64)
    h, d, n = np.concatenate(hs), np.concatenate(ds), np.concatenate(ns)
    h, first = np.unique(h, return_index=True)
    return h, d[first], n[first]


def _lookup(known: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.searchsorted(known, h)
    pos = np.minimum(pos, max(len(known) - 1, 0))
    found = (known[pos] == h) if len(known) else np.zeros(len(h), bool)
    return pos, found


def install(inner, cache: Path, part: tuple[int, int] | None = None):
    """A near.nearest with the same answers as `inner`, reading and writing `cache`."""
    cache.mkdir(parents=True, exist_ok=True)
    stats = {"searches": 0, "rows": 0, "unique_rows": 0, "from_cache": 0, "computed": 0}

    def nearest(split, test_idx=None):
        idx = np.arange(len(split.test)) if test_idx is None else np.asarray(test_idx, dtype=np.int64)
        h = np.asarray(split.test_hash)[idx]
        key = train_key(split)
        uniq, first = np.unique(h, return_index=True)
        kh, _, _ = load(cache, key)
        _, found = _lookup(kh, uniq)
        if part is not None:
            # the parts divide the distinct rows themselves, whatever the cache already holds
            sel = np.arange(part[0], len(uniq), part[1])
            need = sel[~found[sel]]
        else:
            need = np.flatnonzero(~found)
        if len(need):
            # a row with a training row's exact content is at distance 0 from it (identical features,
            # so the same categorical bucket and the same scaled vector): no search needed
            th = np.asarray(split.train_hash)
            tu, tfirst = np.unique(th, return_index=True)
            tpos, copy = _lookup(tu, uniq[need])
            d = np.zeros(len(need))
            nn = np.where(copy, tfirst[tpos], -1).astype(np.int64)
            if (~copy).any():
                d[~copy], nn[~copy] = inner(split, idx[first[need[~copy]]])
            stats["exact_copies"] = stats.get("exact_copies", 0) + int(copy.sum())
            tag = f"p{part[0]}of{part[1]}" if part is not None else "rest"
            np.savez(cache / f"{key}--{tag}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.npz",
                     key=key, hash=uniq[need], dist=d, nn=nn)
        print(f"near_parts: {len(idx)} rows, {len(uniq)} distinct, {int(found.sum())} from the cache, "
              f"{len(need)} computed ({stats.get('exact_copies', 0)} exact copies so far)"
              + (f" (part {part[0]} of {part[1]})" if part else ""), flush=True)
        if part is not None:
            raise PartDone()
        kh, kd, kn = load(cache, key)
        pos, ok = _lookup(kh, h)
        if not ok.all():
            raise RuntimeError(f"near_parts: {int((~ok).sum())} rows missing after the search")
        stats["searches"] += 1
        stats["rows"] += len(idx)
        stats["unique_rows"] += len(uniq)
        stats["from_cache"] += int(found.sum())
        stats["computed"] += len(need)
        return kd[pos], kn[pos]

    nearest.stats = stats
    return nearest


def from_env(inner):
    """install() as the environment asks, or `inner` unchanged."""
    cache = os.environ.get("LEAKCHECK_NEAR_CACHE")
    if not cache:
        return inner
    spec = os.environ.get("LEAKCHECK_NEAR_PART")
    part = tuple(int(x) for x in spec.split("/")) if spec else None
    if part is not None and not (len(part) == 2 and 0 <= part[0] < part[1]):
        raise ValueError(f"LEAKCHECK_NEAR_PART={spec!r}: expected i/n with 0 <= i < n")
    return install(inner, Path(cache), part)
