"""The same nearest-neighbour search as `near.nearest`, on a GPU when one is available.

For tens of millions of rows the exact search costs ~10^15 operations: days on a CPU, minutes
to hours on a GPU. The result must not depend on the device, so the GPU only shortlists, and
the distance to each shortlisted training row is then recomputed exactly as `near.nearest`
computes it; the closest of those is returned.

Shortlisting runs in two passes. The first is float32, which a GPU runs many times faster than
float64 (32x on the T4), and is certified rather than trusted: each training row is ranked by a
lower bound on its distance. The distance is computed from coordinate differences, so float32's
error is relative to the distance itself; the bound subtracts that and the rounding of the
coordinates to float32, which grows with the rows' norms (robust scaling can leave features near
3e5, as in RT-IoT2022). A test row's answer is accepted only when it is an exact copy (distance 0)
or when the lower bound of every training row left off the shortlist exceeds the distance found. The rows where that cannot be shown go
through the second pass, float64 squared distances (data centred first, to keep the
subtraction well conditioned) as before. tests/test_near_torch.py and
experiments/clean_splits.py (`validate`) check that the result equals `near.nearest`.

Requires PyTorch (not a leakcheck dependency): pip install torch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import near

K = 8  # shortlist per test row
U32 = 2.0 ** -24  # float32 unit roundoff


def _norms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(x, dtype=np.float64), axis=1))


def _shortlist(torch, device, xm, centre, queries, k, rows, block, fast):
    """For every query row, the k training rows (positions in xm) with the smallest score.
    fast: the score is a float32 lower bound on the distance, and the (k+1)-th smallest bound
    (the least any row left off the list can be) is returned too; else float64 squared distances."""
    dtype = torch.float32 if fast else torch.float64
    d = xm.shape[1]
    kk = min(k, len(xm))
    keep = min(kk + 1, len(xm)) if fast else kk
    c = 2.0 * (d + 4) * U32  # twice the standard bound for d-term sums and the 3-term combination
    best = {}
    for b in range(0, len(xm), block):
        tc = xm[b:b + block] - centre
        t = torch.as_tensor(tc, dtype=dtype, device=device)
        tn = (t * t).sum(1)
        tb = torch.as_tensor(_norms(tc), dtype=dtype, device=device) if fast else None
        kb = min(keep, t.shape[0])
        for r in range(0, len(queries), rows):
            qc = queries[r:r + rows] - centre
            q = torch.as_tensor(qc, dtype=dtype, device=device)
            if fast:
                # differences first (no cancellation): float32 error is relative to the distance,
                # plus the rounding of the centred coordinates to float32, relative to their norms
                s = torch.cdist(q, t, compute_mode="donot_use_mm_for_euclid_dist")
                qa = torch.as_tensor(_norms(qc), dtype=dtype, device=device)
                s = s * (1 - c) - 4 * U32 * (qa[:, None] + tb[None, :])
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
        del t, tn
    for r in range(0, len(queries), rows):
        v, i = best.pop(r)
        v, i = v.double().numpy(), i.numpy()
        floor = v[:, kk] if v.shape[1] > kk else np.full(len(v), np.inf)
        yield r, i[:, :kk], floor


def nearest(split, test_idx: np.ndarray | None = None, device: str | None = None, k: int = K,
            rows: int = 1024, block: int = 250_000, fast: bool = True) -> tuple[np.ndarray, np.ndarray]:
    import torch

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

    tr_groups = pd.Series(np.arange(len(ktr))).groupby(ktr).indices
    sub_keys = kte[test_idx]
    for key, pos in pd.Series(np.arange(len(sub_keys))).groupby(sub_keys).indices.items():
        members = tr_groups.get(key)
        if members is None:
            continue
        xm = xtr if len(members) == len(xtr) else xtr[members]  # one bucket: no copy
        centre = xm.mean(axis=0)

        def settle(p, cand):
            # exact distances to the shortlist, the formula near.nearest uses
            d = np.sqrt(np.sum((xte[test_idx[p]][:, None, :] - xm[cand]) ** 2, axis=2))
            j = np.argmin(d, axis=1)
            dist[p] = d[np.arange(len(p)), j]
            nn[p] = members[cand[np.arange(len(p)), j]]

        unsure = []
        for r, cand, floor in _shortlist(torch, device, xm, centre, xte[test_idx[pos]], k, rows, block, fast):
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
            for r, cand, _ in _shortlist(torch, device, xm, centre, xte[test_idx[p]], k, rows, block, False):
                settle(p[r:r + rows], cand)
    return dist, nn
