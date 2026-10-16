"""E6: threshold sweeps and bootstrap intervals from E5's saved predictions.

No model is trained here. For each dataset, seed and threshold, a test row is
"leaked" if it has an exact copy, a training row within RMS distance eps, or a
training document with Jaccard >= tau. The macro-F1 gap is original - clean
(clean = rows not leaked). Intervals are a paired bootstrap over test rows:
1,000 resamples, drawn as multinomial counts over the cells
(label, prediction(s), leaked), from which macro-F1 is recomputed exactly
(per class 2TP / (2TP + FP + FN), over classes present, as scikit-learn does).

  python -m experiments.e6_analysis
"""

from __future__ import annotations

import csv
import json

import numpy as np

from experiments import data
from experiments.e5_predictions import ALL_MODELS, OUT, SEEDS
from experiments.ledger import record

EPS_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
TAU_GRID = (0.6, 0.7, 0.8, 0.9, 0.95)
EPS, TAU = 0.01, 0.8  # E2's defaults
B = 1000
CSV = data.ROOT / "results" / "e6_sweep.csv"
ORDER = ["kdd99", "nslkdd", "unsw_nb15", "cicids2017", "ember2018", "uci_phishing", "phiusiil", "sms_spam"]


def macro_f1_from_counts(cm: np.ndarray) -> np.ndarray:
    """cm[..., true, pred] -> macro-F1 over classes present, for each leading index."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fp = cm.sum(axis=-2) - tp
    fn = cm.sum(axis=-1) - tp
    denom = 2 * tp + fp + fn
    present = denom > 0
    f1 = np.where(present, 2 * tp / np.where(present, denom, 1), 0.0)
    return f1.sum(axis=-1) / np.maximum(present.sum(axis=-1), 1)


def leaked_mask(ev: dict, eps: float, tau: float) -> np.ndarray:
    m = ev["exact"].copy()
    m |= np.nan_to_num(ev["dist"], nan=np.inf) <= eps
    m |= np.nan_to_num(ev["sim"], nan=-1.0) >= tau
    return m


class Cells:
    """Rows grouped by (label, pred_a[, pred_b], leaked), for fast exact bootstraps."""

    def __init__(self, y, preds: list[np.ndarray], leaked: np.ndarray):
        labels = np.unique(np.concatenate([y, *preds]))
        self.k = len(labels)
        idx = [np.searchsorted(labels, v) for v in (y, *preds)]
        self.shape = (self.k,) * len(idx) + (2,)
        flat = np.ravel_multi_index((*idx, leaked.astype(int)), self.shape)
        self.counts = np.bincount(flat, minlength=int(np.prod(self.shape))).astype(np.float64)
        self.n = len(y)

    def resample(self, rng, b: int) -> np.ndarray:
        """b bootstrap draws of the cell counts, shape (b, *cells)."""
        return rng.multinomial(self.n, self.counts / self.n, size=b).reshape((b, *self.shape)).astype(np.float64)


def gaps(cnt: np.ndarray, which: int, n_preds: int) -> np.ndarray:
    """original - clean macro-F1 for prediction `which` (1-based) from counts shaped
    (*leading, label, pred_1, ..., pred_n, leaked)."""
    lead = cnt.ndim - (n_preds + 2)
    others = tuple(lead + j for j in range(1, n_preds + 1) if j != which)
    full = cnt.sum(axis=-1)
    clean = cnt[..., 0]
    if others:
        full, clean = full.sum(axis=others), clean.sum(axis=others)
    return macro_f1_from_counts(full) - macro_f1_from_counts(clean)


def load(name: str, seed: int):
    meta = json.loads((OUT / f"{name}_seed{seed}.json").read_text(encoding="utf-8"))
    z = np.load(OUT / meta["npz"], allow_pickle=True)  # written by e5_predictions (labels may be strings)
    return meta, {k: z[k] for k in z.files}


def analyse(name: str) -> tuple[list[dict], dict]:
    rows, summary = [], {"seeds": {}}
    rng = np.random.default_rng(0)
    for seed in SEEDS:
        meta, z = load(name, seed)
        ev = {k: z[k] for k in ("exact", "dist", "sim")}
        y = z["y"]
        settings = [("eps", e, e, TAU) for e in EPS_GRID] + [("tau", t, EPS, t) for t in TAU_GRID if t != TAU]
        for kind, val, eps, tau in settings:
            leaked = leaked_mask(ev, eps, tau)
            for m in ALL_MODELS:
                c = Cells(y, [z[f"pred_{m}"]], leaked)
                point = float(gaps(c.counts.reshape(c.shape), 1, 1))
                lo, hi = np.percentile(gaps(c.resample(rng, B), 1, 1), [2.5, 97.5])
                rows.append({"dataset": name, "seed": seed, "sweep": kind, "value": val, "model": m,
                             "leaked_share": float(leaked.mean()), "gap": point, "lo": float(lo), "hi": float(hi)})
        # H7: LR's gap minus hgb_tuned's, paired (same resampled rows for both)
        leaked = leaked_mask(ev, EPS, TAU)
        c = Cells(y, [z["pred_lr"], z["pred_hgb_tuned"]], leaked)
        full = c.counts.reshape(c.shape)
        point = float(gaps(full, 1, 2) - gaps(full, 2, 2))
        draws = c.resample(rng, B)
        lo, hi = np.percentile(gaps(draws, 1, 2) - gaps(draws, 2, 2), [2.5, 97.5])
        summary["seeds"][seed] = {"lr_minus_hgb_tuned": {"diff": point, "lo": float(lo), "hi": float(hi)},
                                  "hgb_tuned": meta.get("hgb_tuned"), "n_train": meta["n_train"], "n_test": meta["n_test"]}
    return rows, summary


def main() -> None:
    names = [n for n in ORDER if all((OUT / f"{n}_seed{s}.json").exists() for s in SEEDS)]
    all_rows, results = [], {}
    for n in names:
        rows, summary = analyse(n)
        all_rows += rows
        results[n] = summary
        print(f"{n}: {len(rows)} rows", flush=True)
    with open(CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    cell = {}
    for r in all_rows:
        cell.setdefault((r["dataset"], r["sweep"], r["value"], r["model"]), []).append(r)
    results["_cells"] = {f"{d}|{s}|{v}|{m}": {"gap_mean": float(np.mean([x["gap"] for x in rs])),
                                              "lo": [x["lo"] for x in rs], "hi": [x["hi"] for x in rs],
                                              "leaked_share_mean": float(np.mean([x["leaked_share"] for x in rs]))}
                         for (d, s, v, m), rs in cell.items()}
    record("e6_analysis", {"eps_grid": EPS_GRID, "tau_grid": TAU_GRID, "default": [EPS, TAU], "bootstrap": B,
                           "seeds": list(SEEDS), "models": list(ALL_MODELS), "source": "results/e5/*.npz"}, results)


if __name__ == "__main__":
    main()
