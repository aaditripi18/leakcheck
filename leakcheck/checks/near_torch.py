"""The same nearest-neighbour search as `near.nearest`, on a GPU when one is available.

For tens of millions of rows the exact search costs ~10^15 operations: days on a CPU, minutes
to hours on a GPU. The result must not depend on the device, so the GPU only shortlists:
float64 squared distances (data centred first, to keep the subtraction well conditioned;
float32 is not enough: robust scaling can leave features near 3e5, where float32 loses every
digit of a 0.01 distance, as RT-IoT2022 showed) select the K closest training rows per test row, and the distance to each shortlisted row
is then recomputed in float64 exactly as `near.nearest` computes it. The closest of those is
returned. tests/test_near_torch.py and experiments/clean_splits.py (`validate`) check that
the result equals `near.nearest` on real benchmarks.

Requires PyTorch (not a leakcheck dependency): pip install torch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import near

K = 8  # shortlist per test row


def nearest(split, test_idx: np.ndarray | None = None, device: str | None = None, k: int = K,
            rows: int = 1024, block: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
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

    tr_groups = pd.Series(np.arange(len(ktr))).groupby(ktr).indices
    sub_keys = kte[test_idx]
    for key, pos in pd.Series(np.arange(len(sub_keys))).groupby(sub_keys).indices.items():
        members = tr_groups.get(key)
        if members is None:
            continue
        xm = xtr if len(members) == len(xtr) else xtr[members]  # one bucket: no copy
        centre = xm.mean(axis=0)
        kk = min(k, len(members))
        blocks = []
        for b in range(0, len(members), block):
            t = torch.as_tensor(xm[b:b + block] - centre, dtype=torch.float64, device=device)
            blocks.append((b, t, (t * t).sum(1)))
        for r in range(0, len(pos), rows):
            q = torch.as_tensor(xte[test_idx[pos[r:r + rows]]] - centre, dtype=torch.float64, device=device)
            qn = (q * q).sum(1, keepdim=True)
            best_d = best_i = None
            for b, t, tn in blocks:
                d2 = qn + tn[None, :] - 2.0 * (q @ t.T)
                kb = min(kk, t.shape[0])
                v, i = torch.topk(d2, kb, dim=1, largest=False)
                i = i + b
                if best_d is None:
                    best_d, best_i = v, i
                else:
                    best_d, j = torch.topk(torch.cat([best_d, v], 1), kk, dim=1, largest=False)
                    best_i = torch.gather(torch.cat([best_i, i], 1), 1, j)
            cand = best_i.cpu().numpy()
            # exact float64 distances to the shortlist, the formula near.nearest uses
            p = pos[r:r + rows]
            d = np.sqrt(np.sum((xte[test_idx[p]][:, None, :] - xm[cand]) ** 2, axis=2))
            j = np.argmin(d, axis=1)
            dist[p] = d[np.arange(len(p)), j]
            nn[p] = members[cand[np.arange(len(p)), j]]
    return dist, nn
