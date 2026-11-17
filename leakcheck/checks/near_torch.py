"""The same nearest-neighbour search as `near.nearest`, on a GPU when one is available.

For tens of millions of rows the exact search costs ~10^15 operations: days on a CPU, minutes
to hours on a GPU. The result must not depend on the device, so the GPU only shortlists, and
the distance to each shortlisted training row is then recomputed exactly as `near.nearest`
computes it; the closest of those is returned.

Shortlisting runs in two passes. The first is float32, which a GPU runs many times faster than
float64 (32x on the T4), and is certified rather than trusted: each training row is ranked by a
lower bound on its distance. Squared distances come from one matrix product per block,
|q|^2 + |t|^2 - 2 q.t (TF32 disabled), the form a GPU runs at full speed; an earlier version took
coordinate differences (torch.cdist without the product), whose kernel ran slower on the T4 than
the float64 product, and CSE-CIC-IDS2018 without identifier columns (one bucket, 1.9M x 13M
rows) did not finish in 12 hours. The product's float32 error is at most (d + 6) u (|q|^2 + |t|^2)
(u the unit roundoff, d the number of columns: the dot product's error bound plus four roundings);
the bound subtracts twice that, then the rounding of the centred coordinates to float32, which
grows with the rows' norms (robust scaling can leave features near 3e5, as in RT-IoT2022). A test
row's answer is accepted only when it is an exact copy (distance 0) or when the lower bound of
every training row left off the shortlist exceeds the distance found. The rows where that cannot
be shown go through the second pass, float64 squared distances (data centred first, to keep the
subtraction well conditioned) as before.

Repeated training rows are searched once. Without identifier columns many training rows are
copies of one another (CSE-CIC-IDS2018), and a test row whose nearest row has more than K copies
cannot be certified: the K+1-th bound is that same row again. So within each bucket, training rows
with the same row hash are compared, and a row whose embedded vector equals a kept row of its hash
is left out (rows whose hash matches but whose vectors differ, equal only after the hash's
rounding, each stay). Distances are unchanged; the nearest row returned is the first of its copies.
tests/test_near_torch.py and
experiments/clean_splits.py (`validate`) check that the result equals `near.nearest`.

Requires PyTorch (not a leakcheck dependency): pip install torch.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import near

K = 8  # shortlist per test row
U32 = 2.0 ** -24  # float32 unit roundoff
PROGRESS_MIN = 1e10  # buckets of at least this many (training x test) row pairs log their progress


def _norms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(x, dtype=np.float64), axis=1))


def _say(msg: str) -> None:
    print(f"near_torch: {msg}", flush=True)


def _shortlist(torch, device, xm, centre, queries, k, rows, block, fast, progress=0):
    """For every query row, the k training rows (positions in xm) with the smallest score.
    fast: the score is a float32 lower bound on the distance, and the (k+1)-th smallest bound
    (the least any row left off the list can be) is returned too; else float64 squared distances."""
    dtype = torch.float32 if fast else torch.float64
    d = xm.shape[1]
    kk = min(k, len(xm))
    keep = min(kk + 1, len(xm)) if fast else kk
    c = 2.0 * (d + 8) * U32  # twice the product's error bound, (d + 6) u (|q|^2 + |t|^2), with room
    best = {}
    t0, nb = time.time(), -(-len(xm) // block)
    for bi, b in enumerate(range(0, len(xm), block)):
        if progress and bi and bi % progress == 0:
            el = time.time() - t0
            _say(f"{'float32' if fast else 'float64'} pass, {len(queries)} test rows: training block {bi}/{nb} "
                 f"({b} of {len(xm)} rows), {el:.0f} s, about {el * (nb - bi) / bi:.0f} s to go")
        tc = xm[b:b + block] - centre
        t = torch.as_tensor(tc, dtype=dtype, device=device)
        if fast:
            tb = _norms(tc)  # float64
            tb2 = torch.as_tensor((1 - c) * tb ** 2, dtype=dtype, device=device)
            tb = torch.as_tensor(tb, dtype=dtype, device=device)
        else:
            tn = (t * t).sum(1)
        kb = min(keep, t.shape[0])
        for r in range(0, len(queries), rows):
            qc = queries[r:r + rows] - centre
            q = torch.as_tensor(qc, dtype=dtype, device=device)
            if fast:
                # (1 - c)(|q|^2 + |t|^2) - 2 q.t <= the squared distance minus the product's error;
                # then the rounding of the centred coordinates to float32, relative to their norms
                qa = _norms(qc)
                s = torch.addmm(tb2[None, :], q, t.T, alpha=-2.0)
                s += torch.as_tensor((1 - c) * qa ** 2, dtype=dtype, device=device)[:, None]
                s.clamp_(min=0).sqrt_()
                qa = torch.as_tensor(qa, dtype=dtype, device=device)
                s -= 4 * U32 * (qa[:, None] + tb[None, :])
            else:
                s = (q * q).sum(1, keepdim=True) + tn[None, :] - 2.0 * (q @ t.T)
            v, i = torch.topk(s, kb, dim=1, largest=False)
            i = i + b
            if r in best:
                bv, bi = (x.to(device) for x in best[r])
                v, j = torch.topk(torch.cat([bv, v], 1), min(keep, bv.shape[1] + v.shape[1]), dim=1, largest=False)
                i = torch.gather(torch.cat([bi, i], 1), 1, j)
            best[r] = (v.cpu(), i.cpu())
            del s, q
        del t
    for r in range(0, len(queries), rows):
        v, i = best.pop(r)
        v, i = v.double().numpy(), i.numpy()
        floor = v[:, kk] if v.shape[1] > kk else np.full(len(v), np.inf)
        yield r, i[:, :kk], floor


def _distinct(xtr: np.ndarray, members: np.ndarray, th: np.ndarray, chunk: int = 1_000_000):
    """(members kept, their matrix): the bucket's training rows with repeated vectors left out.
    A row is left out only when its vector is identical to a kept row with the same hash; rows
    with the same hash but different vectors are compared again among themselves, so each distinct
    vector keeps exactly one row, its first. When the bucket is the whole training side, xtr is
    compacted in place (no second copy of a matrix that can be 7 GB), and the matrix returned is a
    view of its front."""
    keep = np.zeros(len(members), dtype=bool)
    cand = np.arange(len(members))  # positions in members not yet kept or left out
    while len(cand):
        _, first, inv = np.unique(th[members[cand]], return_index=True, return_inverse=True)
        rep = cand[first[inv]]  # the first candidate with the same hash
        keep[cand[first]] = True
        other = rep != cand
        cand, rep = cand[other], rep[other]
        differ = np.zeros(len(cand), dtype=bool)
        for a in range(0, len(cand), chunk):
            c, r = cand[a:a + chunk], rep[a:a + chunk]
            differ[a:a + chunk] = ~(xtr[members[c]] == xtr[members[r]]).all(axis=1)
        cand = cand[differ]  # same hash, another vector: the next round keeps one of each
    whole = len(members) == len(xtr)  # the indices of a groupby are sorted: members is arange
    if keep.all():
        return members, (xtr if whole else xtr[members])
    kept = members[keep]
    if not whole:
        return kept, xtr[kept]
    dst = 0
    for a in range(0, len(xtr), chunk):  # rows only move towards the front
        blk = xtr[a:a + chunk][keep[a:a + chunk]]
        xtr[dst:dst + len(blk)] = blk
        dst += len(blk)
    return kept, xtr[:dst]


def nearest(split, test_idx: np.ndarray | None = None, device: str | None = None, k: int = K,
            rows: int = 1024, block: int = 250_000, fast: bool = True,
            distinct: bool = True, progress: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """progress: a line every this many training blocks (rows done, elapsed time, time to go), so a
    slow search shows in the log while it runs; 0 for none."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False  # the bound assumes float32 products
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if test_idx is None:
        test_idx = np.arange(len(split.test))
    xtr, xte = near.embed(split)
    ktr, kte = near._cat_keys(split)
    dist = np.full(len(test_idx), np.inf)
    nn = np.full(len(test_idx), -1, dtype=np.int64)
    if xtr.shape[1] == 0 or len(split.train) == 0:
        return dist, nn
    # recomputing a distance in the matrices' own precision moves it by at most this share
    slack = 2.0 * (xtr.shape[1] + 4) * (U32 if xtr.dtype == np.float32 else 2.0 ** -53)
    nearest.redone = 0  # rows the float32 pass could not certify (read by experiments and tests)
    nearest.searched = 0  # training rows searched, after repeated rows are left out
    th = np.asarray(split.train_hash) if distinct else None

    tr_groups = pd.Series(np.arange(len(ktr))).groupby(ktr).indices
    sub_keys = kte[test_idx]
    for key, pos in pd.Series(np.arange(len(sub_keys))).groupby(sub_keys).indices.items():
        members = tr_groups.get(key)
        if members is None:
            continue
        if distinct:
            members, xm = _distinct(xtr, members, th)
        else:
            xm = xtr if len(members) == len(xtr) else xtr[members]  # one bucket: no copy
        nearest.searched += len(members)
        big = progress and len(members) * len(pos) >= PROGRESS_MIN  # buckets worth watching
        if big:
            _say(f"bucket of {len(members)} training rows ({len(xm)} after repeats are left out) x {len(pos)} test rows")
        t_bucket = time.time()
        centre = xm.mean(axis=0)

        def settle(p, cand):
            # exact distances to the shortlist, the formula near.nearest uses
            d = np.sqrt(np.sum((xte[test_idx[p]][:, None, :] - xm[cand]) ** 2, axis=2))
            j = np.argmin(d, axis=1)
            dist[p] = d[np.arange(len(p)), j]
            nn[p] = members[cand[np.arange(len(p)), j]]

        unsure = []
        for r, cand, floor in _shortlist(torch, device, xm, centre, xte[test_idx[pos]], k, rows, block, fast,
                                         progress if big else 0):
            p = pos[r:r + rows]
            settle(p, cand)
            if fast:
                # certified: every row off the list has a squared distance above the one found,
                # with room for the rounding of the recomputation on both sides
                # (a distance of 0, an exact copy, cannot be beaten, however many copies tie)
                ok = (dist[p] == 0) | (floor * (1 - slack) > dist[p].astype(np.float64) * (1 + slack))
                unsure.append(p[~ok])
        if fast and unsure:
            p = np.concatenate(unsure)
            nearest.redone += len(p)
            if big:
                _say(f"{len(p)} of {len(pos)} test rows not certified by float32: float64 pass")
            for r, cand, _ in _shortlist(torch, device, xm, centre, xte[test_idx[p]], k, rows, block, False,
                                         progress if big else 0):
                settle(p[r:r + rows], cand)
        if big:
            _say(f"bucket done in {time.time() - t_bucket:.0f} s")
    return dist, nn
