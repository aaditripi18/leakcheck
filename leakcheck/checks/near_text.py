"""L2 text near-duplicates: MinHash + LSH over character shingles.

LSH only proposes candidate pairs. Every pair that is reported has its exact
Jaccard similarity computed from the shingle sets, so a reported near-duplicate
is never an estimate. What LSH can do is miss pairs (recall), which E0 measures.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from ..findings import Finding, grade, wilson

SHINGLE = 5
NUM_PERM = 128
BANDS, ROWS = 32, 4  # candidate threshold ~ (1/32)^(1/4) = 0.42
TAU = 0.8
BUCKET_CAP = 100  # training members kept per LSH bucket
SAMPLE = 20_000
N_EXAMPLES = 5
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def normalize(s: str) -> str:
    return " ".join(str(s).lower().split())


def shingles(s: str, k: int = SHINGLE) -> frozenset[str]:
    s = normalize(s)
    if len(s) <= k:
        return frozenset([s])
    return frozenset(s[i : i + k] for i in range(len(s) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _mix(x: np.ndarray) -> np.ndarray:
    # splitmix64 finaliser; uint64 arithmetic wraps, which is what we want.
    x = x ^ (x >> np.uint64(30))
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x = x ^ (x >> np.uint64(27))
    x = x * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


_SEEDS = _mix(np.arange(1, NUM_PERM + 1, dtype=np.uint64))


def signature(sh: frozenset[str]) -> np.ndarray:
    h = np.fromiter((zlib.crc32(x.encode("utf-8")) for x in sh), dtype=np.uint64, count=len(sh))
    with np.errstate(over="ignore"):
        return _mix(h[None, :] ^ _SEEDS[:, None]).min(axis=1)


def _band_keys(sig: np.ndarray) -> np.ndarray:
    """(n_docs, BANDS) bucket key per band."""
    b = sig.reshape(len(sig), BANDS, ROWS)
    with np.errstate(over="ignore"):
        k = np.zeros(b.shape[:2], dtype=np.uint64)
        for r in range(ROWS):
            k = _mix(k ^ b[:, :, r])
    return k


def nearest_text(train_text, test_text, test_idx=None, tau: float = TAU):
    """For each chosen test doc: best exact Jaccard to a training doc among LSH candidates.

    Returns (similarity, train_index); -1 / 0.0 where no candidate was found.
    """
    test_text = pd.Series(test_text).fillna("").reset_index(drop=True)
    if test_idx is None:
        test_idx = np.arange(len(test_text))
    tr_sh = [shingles(s) for s in pd.Series(train_text).fillna("")]
    te_sh = [shingles(test_text.iloc[i]) for i in test_idx]
    sim = np.zeros(len(test_idx))
    best = np.full(len(test_idx), -1, dtype=np.int64)
    if not tr_sh or not te_sh:
        return sim, best

    sig_tr = np.stack([signature(s) for s in tr_sh])
    sig_te = np.stack([signature(s) for s in te_sh])
    ktr, kte = _band_keys(sig_tr), _band_keys(sig_te)

    pairs = []
    for b in range(BANDS):
        tr = pd.DataFrame({"key": ktr[:, b], "j": np.arange(len(ktr))})
        tr = tr.groupby("key", sort=False).head(BUCKET_CAP)
        te = pd.DataFrame({"key": kte[:, b], "i": np.arange(len(kte))})
        m = te.merge(tr, on="key")
        i, j = m["i"].to_numpy(), m["j"].to_numpy()
        est = (sig_te[i] == sig_tr[j]).mean(axis=1)
        keep = est >= tau - 0.15  # generous: exact Jaccard decides below
        pairs.append(pd.DataFrame({"i": i[keep], "j": j[keep], "est": est[keep]}))
    cand = pd.concat(pairs).drop_duplicates(["i", "j"])
    if cand.empty:
        return sim, best

    cand = cand.sort_values("est", ascending=False).groupby("i").head(5)
    for ii, jj in zip(cand["i"].to_numpy(), cand["j"].to_numpy()):
        s = jaccard(te_sh[ii], tr_sh[jj])
        if s > sim[ii]:
            sim[ii], best[ii] = s, jj
    return sim, best


def check_near_text(split, tau: float = TAU, sample: int = SAMPLE, seed: int = 0) -> Finding | None:
    if split.text is None:
        return None
    n_te = len(split.test)
    idx = np.arange(n_te) if n_te <= sample else np.sort(np.random.default_rng(seed).choice(n_te, sample, replace=False))
    sim, nn = nearest_text(split.train[split.text], split.test[split.text], idx, tau)

    tr_norm = set(split.train[split.text].fillna("").map(normalize))
    te_norm = split.test[split.text].fillna("").map(normalize).iloc[idx]
    same = te_norm.isin(tr_norm).to_numpy()
    near = (sim >= tau) & ~same
    k, n = int(near.sum()), len(idx)
    rate = k / n if n else 0.0
    ex = np.flatnonzero(near)[:N_EXAMPLES]
    return Finding(
        code="L2.near_text",
        severity=grade(rate, high=0.05, medium=0.01),
        value=rate,
        summary=f"{rate:.2%} of test {split.text!r} values are near-copies (Jaccard >= {tau}) of a training value; "
        f"{same.mean():.2%} are identical after normalisation",
        details={
            "tau": tau,
            "n_checked": n,
            "n_near": k,
            "identical_rate": float(same.mean()) if n else 0.0,
            "sampled": n < n_te,
            "ci95": wilson(k, n),
            "shingle": SHINGLE,
            "num_perm": NUM_PERM,
            "bands": BANDS,
        },
        examples=[{"test": int(idx[i]), "train": int(nn[i]), "jaccard": float(sim[i])} for i in ex],
    )
