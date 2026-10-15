# leakcheck

Security ML benchmarks report 99% accuracy. leakcheck measures how much of
that comes from the test set overlapping the training set: exact and
near-duplicates, contradictory labels, future data in training, and single
columns that give the label away. It also removes the leakage, and proves the
removal worked by re-scanning its own output.

## Results

8 benchmarks, 9.5M rows, 4 untuned baselines × 5 seeds. Every number below is
generated from `results/ledger.jsonl` into [docs/RESULTS.md](docs/RESULTS.md);
the narrative is in [docs/FINDINGS.md](docs/FINDINGS.md). The decision rules
for each hypothesis were committed before the runs they govern.

| Dataset | Test rows exactly copying a training row | Macro-F1 lost when leaked rows are removed (LR → HistGB) |
|---|---|---|
| KDD Cup 99 | **68.4%** | **17.5 → 15.0 points** |
| NSL-KDD | 3.0% | 11.9 → 8.8 |
| SMS Spam Collection | 11.0% | 3.1 → 2.5 |
| EMBER 2018 (random split) | 0.01% | 2.4 → 1.3 |
| UNSW-NB15 (official split) | 10.4% | 2.0 → 1.0 |
| UCI Phishing Websites | **64.4%** | 1.5 → 0.8 |
| CIC-IDS2017 | 14.1% | ≈ 0 |
| PhiUSIIL | 0.0% | 0 (every model at 0.9999+) |

| | Hypothesis | Verdict |
|---|---|---|
| H1 | At least half of the benchmarks have ≥ 5% of test rows copying a training row | **supported** (5 of 8) |
| H2 | Removing the copies costs ≥ 5 macro-F1 points on the two most-copied benchmarks | **not supported**: KDD99 loses 15–17.5, UCI Phishing ≤ 1.7 |
| H3 | A time-aware split costs more than deduplication | **supported** on EMBER, by a small margin (0.3–0.9 points) |
| H4 | Weaker models lean harder on leaked rows | **supported** (LR gap > HistGB gap on 6 of 6 eligible datasets) |
| H5 | Some benchmark has a single feature at ≥ 95% balanced accuracy | **supported** (PhiUSIIL `URLSimilarityIndex`, 99.6%) |

What the numbers say:

- **The copy rate alone does not predict inflation.** UCI Phishing has as many
  copies as KDD99 but loses about a point: with 30 features of 2–3 values
  each, a copy is a common pattern that a model would get right anyway. KDD99's
  copies are memorised exceptions, and nearly three quarters of its gap (72%)
  comes from *near*-copies.
- **Validated against the literature.** leakcheck reproduces the KDD99
  redundancy tables of Tavallaee et al. (2009) exactly for the training set,
  and to within 2 records for the test set. On synthetic data it recovers
  planted leakage with zero error in 720 configurations.
- **It does not flag everything.** EMBER's official temporal split comes back
  clean: 0.009% exact copies and no training file newer than the test period.
- **Some benchmarks are too easy to rank models.** PhiUSIIL stays at 0.9995+
  macro-F1 even after removing the four features that each reach 95% alone.
- **A random split can hide a collapse.** CIC-IDS2017 scores 0.97–0.999 on a
  random split and 0.46–0.57 when tested on the later capture files, whose
  attack types never appear in training.

## Use it

```
pip install -e .

python -m leakcheck scan --train train.csv --test test.csv --label label \
    [--time ts] [--group host] [--text url] [--ignore id] [--json report.json]

python -m leakcheck clean --train train.csv --test test.csv --label label --out cleaned/ \
    [--near-eps 0.01] [--text-tau 0.8] [--drop-group-overlap] [--dedup-within] \
    [--conflicts drop|majority|keep]
```

`scan` exits 0 when clean or minor, 1 on medium findings, 2 on high findings, 3
on bad input.

| Check | What it reports |
|---|---|
| L1 exact | rows repeated within train / within test; test rows identical to a training row (per class) |
| L2 near (tabular) | test rows within an RMS distance of a training row, after signed-log + robust (median/IQR) scaling; categorical values must match exactly; exact nearest neighbour; 95% Wilson interval when sampled; compared with the same curve for held-out training rows |
| L2 near (text) | `--text` column: MinHash + LSH over 5-character shingles, every reported pair verified by exact Jaccard |
| L3 conflict | identical features with different labels; the test-accuracy ceiling this forces; test rows whose training copies all disagree |
| L4 temporal | training rows newer than the earliest / median / latest test row (TESSERACT constraint C1); per-class time ranges; timestamp as a one-feature classifier |
| L5 shortcut | best one-feature classifier (balanced test accuracy); identifier-like columns |
| L6 group | test rows whose host / family / session also appears in training |

**`clean` proves its own output.** It resolves conflicts, optionally removes
within-split repeats, then drops test rows that leak from the final training
set. It writes `train.csv`, `test.csv` and `clean_log.json`, then re-reads the
written files and re-checks every row. If anything it was asked to remove is
still there, it exits with code 4. The log records SHA-256 of inputs and
outputs, the options, and the input row number of every removed row with the
reason.

Speed on a laptop CPU: L1 + L3 + L5 on 1M train / 300k test rows in 31 s; L2
tabular with 20k sampled test rows in 59 s; L2 text on 160k URLs in 16 s.

## Website and in-browser scanner

`site/` is a static site: the project page (`index.html`) and a scanner
(`scan.html`) where anyone can drop a train/test CSV pair and get the same
report, without installing anything.

- **The scanner cannot send your files anywhere.** Both pages carry a
  Content-Security-Policy with `connect-src 'none'`, and the scan runs in a
  worker started from a `blob:` URL, so it inherits that policy. Verified in
  Chromium: `fetch` and `WebSocket` are refused from the page and from the
  worker. Example datasets ship as scripts, so even they need no request.
- **Same answers as Python.** `site/engine/` is a JavaScript port of every
  check, including scikit-learn's one-feature tree. `tests/test_js_parity.py`
  runs both engines on the same files and requires identical counts, and
  single-feature scores within 0.005.
- **No hand-typed numbers.** `python scripts/build_site.py` generates both
  pages from `site/data.json`, which `scripts/report.py` generates from the
  ledger. `tests/test_site.py` checks this, along with the CSP.

```
python scripts/report.py
python scripts/build_site.py --examples    # pages, example datasets, CSS (needs Node for Tailwind)
python -m http.server 8765 --directory site
```

Deploy the `site/` folder to any static host. `vercel.json` and `_headers`
send the same policy as an HTTP header. Design from Stitch; content and
wiring are generated.

## Limitations

- **A copy is not always a bug.** Real traffic repeats. The finding is that a
  benchmark which rewards memorised repeats does not measure detection of new
  attacks; both scores are reported.
- **The cleaned test set is harder by construction**: it is the part unlike
  the training data. The gap measures how much of a headline score was earned
  on copies, not that a model is worse in every sense.
- **Thresholds are choices.** Near-duplicate distance (0.01 RMS in scaled
  space) and Jaccard (0.8) are fixed defaults; L2 also reports a curve and a
  training-set baseline. The identifier heuristic is advisory, and has one
  known false positive (KDD99 `dst_host_same_src_port_rate`).
- **Baselines are untuned and training sets are capped at 300k rows**, so the
  scores are typical, not state of the art.
- **EMBER** is scanned on leakcheck's own ~680-column numeric view, not the
  official 2,381-dimension vector (EMBER's code is AGPL and is not used); exact
  duplicates use a digest of every raw feature instead.
- **CIC-IDS2017's chronological split is confounded** with unseen attack types,
  so H3 was decided on EMBER.
- Open items before publication are listed at the end of
  [docs/FINDINGS.md](docs/FINDINGS.md).

## Reproduce

Everything runs on a laptop CPU; no GPU is needed. Downloads are verified
against pinned SHA-256 hashes and resume after dropped connections. Where each
dataset comes from, and how each copy was checked against an independent
reference, is in [data/README.md](data/README.md).

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest                       # 91 tests

python scripts/fetch.py                              # ~2.1 GB, hash-verified
python -m experiments.e0_sources && python -m experiments.e0_sources_week4
python -m experiments.e0_synthetic && python -m experiments.e0_kdd
python -m experiments.e1_audit                       # ~45 min, EMBER is 30 of them
python -m experiments.e2_retrain uci_phishing sms_spam unsw_nb15 kdd99 cicids2017 nslkdd phiusiil ember2018
python -m experiments.e4_ablation phiusiil cicids2017
python scripts/report.py                             # docs/RESULTS.md + site/data.json
```

Every run appends a line to `results/ledger.jsonl` with the git commit, dataset
hashes, configuration and numbers. `docs/RESULTS.md` and `site/data.json` are
generated from it and never edited by hand. Per-seed raw outputs are in
`results/e2/` and `results/e4/`. [PLAN.md](PLAN.md) holds the hypotheses and
every decision rule, with the date each was fixed.

Tests use synthetic splits (`leakcheck/synth.py`) where every planted
duplicate, conflict and shortcut is counted, and the checks must recover the
counts exactly.
