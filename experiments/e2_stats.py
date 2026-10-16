"""E2 uncertainty: how sure are the E2 gaps? (secondary analysis, no retraining)

Added after the E2 verdicts were recorded, so it changes none of them; it says
which of the measured gaps the data can tell apart from zero.

Read from the per-seed caches in results/e2/ (the same files the e2_retrain
ledger lines summarise; this script first checks it reproduces those summaries
exactly) and gives, per dataset and model:

  seeds     the macro-F1 gap original - no_near over the 5 seeds: mean and a
            95% t interval (4 degrees of freedom). For datasets with an official
            split the test set is fixed, so this is model and subsample
            variation only.
  rows      for seed 0, the accuracy on leaked test rows against the accuracy on
            clean (no_near) rows: two disjoint sets of rows, so a Newcombe
            (hybrid Wilson) 95% interval for the difference and a two-sided
            two-proportion z-test. The accuracy gap original - no_near equals
            the leaked share times this difference, so its interval is the
            same interval scaled by the share.
  H4 paired the LR gap minus the HistGB gap, paired by seed: mean, 95% t interval.

  python -m experiments.e2_stats
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
from scipy import stats

from experiments import data
from experiments.ledger import record
from experiments.models import MODELS

OUT = data.ROOT / "results" / "e2"
STATS_CSV = data.ROOT / "results" / "e2_stats.csv"
LEDGER = data.ROOT / "results" / "ledger.jsonl"
ORDER = ["kdd99", "nslkdd", "unsw_nb15", "cicids2017", "ember2018", "uci_phishing", "phiusiil", "sms_spam"]
Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid score interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return float(2 * stats.norm.sf(abs((k1 / n1 - k2 / n2) / se)))


def t_interval(xs) -> tuple[float, float, float]:
    xs = np.asarray(xs, float)
    m = float(xs.mean())
    if len(xs) < 2 or xs.std(ddof=1) == 0:
        return m, m, m
    h = float(stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / math.sqrt(len(xs)))
    return m, m - h, m + h


def _ledger_gaps() -> dict:
    out = {}
    for line in open(LEDGER, encoding="utf-8"):
        d = json.loads(line)
        if d["experiment"].startswith("e2_retrain:"):
            out[d["experiment"].split(":", 1)[1]] = d["results"]["macro_f1_gap"]  # last line wins
    return out


def load(name: str) -> list[dict]:
    return [json.loads((OUT / f"{name}_seed{s}.json").read_text(encoding="utf-8")) for s in range(5)]


def analyse(name: str, per_seed: list[dict], recorded: dict) -> dict:
    res = {"models": {}, "leak_share_seed0": per_seed[0]["leak_rate"]["near"]}
    for m in MODELS:
        gaps = [r["models"][m]["original"]["macro_f1"] - r["models"][m]["no_near"]["macro_f1"]
                if r["models"][m]["no_near"]["macro_f1"] is not None else np.nan for r in per_seed]
        if not np.any(np.isnan(gaps)):
            # the per-seed caches must reproduce the recorded E2 summary
            assert abs(np.mean(gaps) - recorded[m]["orig_minus_no_near"]) < 1e-12, (name, m)
        g, glo, ghi = t_interval([x for x in gaps if not np.isnan(x)])
        s0 = per_seed[0]["models"][m]
        lk, cl = s0["leaked"], s0["no_near"]
        row = {"f1_gap_mean": g, "f1_gap_lo": glo, "f1_gap_hi": ghi, "n_leaked": lk["n"], "n_clean": cl["n"]}
        if lk["n"] and cl["n"]:
            k1, k2 = round(lk["accuracy"] * lk["n"]), round(cl["accuracy"] * cl["n"])
            d, lo, hi = newcombe(k1, lk["n"], k2, cl["n"])
            share = lk["n"] / (lk["n"] + cl["n"])
            row.update({"acc_leaked": lk["accuracy"], "acc_clean": cl["accuracy"], "acc_diff": d,
                        "acc_diff_lo": lo, "acc_diff_hi": hi, "p_two_prop": two_prop_p(k1, lk["n"], k2, cl["n"]),
                        "acc_gap": share * d, "acc_gap_lo": share * lo, "acc_gap_hi": share * hi})
        res["models"][m] = row
    lr = [r["models"]["lr"]["original"]["macro_f1"] - r["models"]["lr"]["no_near"]["macro_f1"] for r in per_seed
          if r["models"]["lr"]["no_near"]["macro_f1"] is not None]
    hgb = [r["models"]["hgb"]["original"]["macro_f1"] - r["models"]["hgb"]["no_near"]["macro_f1"] for r in per_seed
           if r["models"]["hgb"]["no_near"]["macro_f1"] is not None]
    if len(lr) == len(hgb) == 5:
        d, lo, hi = t_interval(np.array(lr) - np.array(hgb))
        res["h4_paired"] = {"lr_minus_hgb": d, "lo": lo, "hi": hi}
    return res


def main() -> None:
    recorded = _ledger_gaps()
    rows, results = [], {}
    for name in ORDER:
        r = analyse(name, load(name), recorded[name])
        results[name] = r
        for m, v in r["models"].items():
            rows.append({"dataset": name, "model": m, **{k: (round(x, 6) if isinstance(x, float) else x)
                                                        for k, x in v.items()}})
        h = r.get("h4_paired")
        print(f"{name:13s} " + "  ".join(
            f"{m} {v['f1_gap_mean'] * 100:+.1f} [{v['f1_gap_lo'] * 100:+.1f},{v['f1_gap_hi'] * 100:+.1f}]"
            for m, v in r["models"].items()) + (f"  | LR-HGB {h['lr_minus_hgb'] * 100:+.1f} "
                                                f"[{h['lo'] * 100:+.1f},{h['hi'] * 100:+.1f}]" if h else ""))
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "model"), k))
    with open(STATS_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    record("e2_stats", {"source": "results/e2/*_seed{0..4}.json", "interval": "95%",
                        "seed_interval": "t, df=4", "row_interval": "Newcombe hybrid score, seed 0",
                        "post_hoc": True}, results)


if __name__ == "__main__":
    main()
