# leakcheck

**Live site and in-browser scanner: https://leakcheck-nu.vercel.app**

Security ML benchmarks report 99% accuracy. leakcheck measures how much of
that comes from the test set overlapping the training set: exact and
near-duplicates, contradictory labels, future data in training, and single
columns that give the label away. It also removes the leakage, and proves the
removal worked by re-scanning its own output.

The problem is not confined to old benchmarks. In the datasets current papers
use, **40% of CSE-CIC-IDS2018's test rows and 44% of CIC-IoT2023's are exact
copies of training rows**; in IDS2018 (and in Edge-IIoTset, 16%) the copies only
show once timestamp, address and port columns, which make every row unique, are
set aside.

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
| H6 | Tuning and all the training data do not remove the gap | **supported**: tuned HistGB on all 4.9M KDD99 rows loses 22.2 points, NSL-KDD 8.6; every seed's interval above 5 |
| H7 | LR's gap exceeds tuned HistGB's, interval above zero | **supported** on 4 of 4 large benchmarks, 3 of 3 seeds each |
| H8 | Whether a gap clears one point does not depend on the threshold | **not supported**: same in 32 of 40 cells; every change is a gap near one point |
| H9 | On network traffic, a time split costs more than removing copies | **not supported**: raw UNSW-NB15, time split 0.2–0.5 points vs near-copies 4.3–6.5 |

H6–H9 are phase 2 (PLAN.md): full training sets, a tuned model, every test
prediction saved, a paired bootstrap over test rows, a threshold sweep, and the
raw UNSW-NB15 records.

**Phase 3: the benchmarks in use now** (2018-2023; [docs/RESULTS.md](docs/RESULTS.md)).
Each was checked against its publisher before use: IDS2018's ten CSVs are
byte-identical to the official AWS objects; IDS2018's and CIC-IoT2023's label
counts equal the published tables (16,232,943 and 46,686,579 rows).

| Dataset | Test rows | Exact copies of train, as distributed | ... without identifier columns | Best single feature |
|---|--:|--:|--:|---|
| Edge-IIoTset (2022) | 443,841 | 0.04% | **16.2%** (38.4% near) | `dns.qry.name.len` 100% |
| CSE-CIC-IDS2018 | 3,246,589 | 4.9% | **40.3%** (54.9% near; 13.1% contradicted) | `Dst Port` 90.9% |
| CIC-IoT2023 | 9,337,316 | **44.3%** (32.8% near) | (no identifier columns) | `Weight` 99.1% |

The "without identifiers" view drops the columns leakcheck's own identifier
check flags; it was added after the first of these runs and is reported as post
hoc. The pre-registered view is the first column.

**The caveats, measured** (phase 3b): each statement this README used to make
as a caveat now has an experiment.

| Caveat | Measurement |
|---|---|
| "A copy is not always a bug" | Retrained with every training copy removed, models still get **97% of KDD99's copied test rows right** (98% NSL-KDD, 92-93% UCI Phishing); 0-1% are memorised (EMBER 6-19%). The copies are easy rows, not memorised exceptions (E9) |
| "The cleaned test set is harder by construction" | Removing a random subset of the same size and class mix costs **-6 points on KDD99 and 0.5-0.7 on NSL-KDD**, against 15-17 and 8.5-11.9 for removing the copies: the gap comes from which rows go, mostly (E10) |
| "CIC-IDS2017's time split is confounded with unseen attacks" | Split by time **within each attack type**, it costs 0.5-1.5 points: the collapse to 0.46-0.57 on the file-wise time split was the unseen attack types (E12) |

| | Hypothesis (phase 3) | Verdict |
|---|---|---|
| H10 | At least 2 of the 3 current benchmarks have >= 5% exact copies, as distributed | **not supported**: 1 of 3 (IDS2018 4.91%, just under) |
| H12 | Some current benchmark has a single feature at >= 95% | **supported**: Edge-IIoTset 100%, CIC-IoT2023 99.1% |
| H13 | KDD99 and NSL-KDD memorise more copied rows than UCI Phishing | **not supported**: 0.0% and 0.3% vs 3.3% |
| H14 | Wherever copies cost >= 1 point, random removal costs < 0.5 | **not supported**: 9 of 23 cells (NSL-KDD, UNSW-NB15, UCI Phishing LR at 1.5) |
| H16 | With every attack type seen in training, CIC-IDS2017's time split costs <= 2 points | **supported**: 0.5-1.5 |

What the numbers say:

- **Copies inflate scores by being easy, not by being memorised.** A model
  trained without any copy still gets 97% of KDD99's copied test rows right
  (E9). What the copies do is fill the test set with repeated, easy traffic:
  remove them and what is left is the hard part, where KDD99 loses 15-17
  macro-F1 points and NSL-KDD 8.5-11.9. Nearly three quarters of KDD99's gap
  (72%) comes from *near*-copies. (This README used to call KDD99's copies
  "memorised exceptions"; E9 measured it and that was wrong.)
- **Today's benchmarks hide their copies behind identifiers.** A timestamp or a
  flow ID makes every row unique, so an exact-duplicate check sees nothing;
  without those columns, 40% of IDS2018's test set and 16% of Edge-IIoTset's
  copy a training row. CIC-IoT2023 ships no identifiers and 44% of its test rows
  are copies as distributed.
- **Validated against the literature.** leakcheck reproduces the KDD99
  redundancy tables of Tavallaee et al. (2009) exactly for the training set,
  and to within 2 records for the test set. On synthetic data it recovers
  planted leakage with zero error in 720 configurations.
- **It does not flag everything.** EMBER's official temporal split comes back
  clean: 0.009% exact copies and no training file newer than the test period.
- **Some benchmarks are too easy to rank models.** PhiUSIIL stays at 0.9995+
  macro-F1 even after removing the four features that each reach 95% alone.
- **On network traffic the copies matter and the clock does not.** In the raw
  UNSW-NB15 records (every attack family before and after the cut), splitting by
  capture time costs under half a point; removing near-copies costs 4–6. Even the
  time split leaves a third of its test rows with a copy in training, because the
  traffic repeats. `sttl` (source TTL) alone reaches 98–99% balanced accuracy.
- **A random split can hide a collapse.** CIC-IDS2017 scores 0.97–0.999 on a
  random split and 0.46–0.57 when tested on the later capture files, whose
  attack types never appear in training.

## Install

```
pip install "git+https://github.com/aaditripi18/leakcheck"
docker build -t leakcheck https://github.com/aaditripi18/leakcheck.git
docker run --rm -v "$PWD:/data" leakcheck scan --train /data/train.csv --test /data/test.csv --label label
```

Python 3.10–3.13 on Linux, Windows and macOS (CI); the research harness in
`experiments/` needs Python 3.11+ (scikit-learn 1.8+). The supported Python API,
exit statuses and every check code are in [docs/API.md](docs/API.md); changes are
in [CHANGELOG.md](CHANGELOG.md). Distributed from GitHub only.

## How it relates to existing tools

[Deepchecks](https://github.com/deepchecks/deepchecks)' train/test validation
already flags samples shared by train and test, conflicting labels, date-based
leakage and features or identifiers that predict the label, and
[cleanlab](https://github.com/cleanlab/cleanlab) finds near-duplicates and label
issues. If you use those, you have much of the detection. What leakcheck adds:

- **The number the duplicates are worth.** The audit retrains standard models on
  the cleaned split and reports the gap, with intervals; on these benchmarks the
  copy rate alone does not predict it (KDD Cup 99 and UCI Phishing copy about as
  much; one loses 17.5 macro-F1 points, the other 1.5).
- **Near-copies judged against a baseline**: how close test rows sit to training
  rows compared with how close training rows sit to each other, plus the
  accuracy ceiling that conflicting labels impose.
- **Group overlap** (the same host, family or session on both sides).
- **`clean` that proves itself**: the cleaned files are read back and re-scanned,
  with a log of every removed row.
- **A browser scanner that cannot upload** (`connect-src 'none'`), with the same
  counts as Python.

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
- **Sized for up to about half a million rows.** Measured with the engine's
  Node entry point: 50,000 rows in 12 s (0.5 GB), 200,000 in 56 s (1.4 GB),
  500,000 in 122 s (2.8 GB), 20 numeric columns. A browser tab gives out
  beyond that; larger datasets (KDD Cup 99 has 4.9 million training rows) go
  through the command-line tool or its Docker image, where the near-copy search
  uses NumPy instead of plain JavaScript.
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

Deployed on Vercel from `site/` (`cd site && vercel deploy --prod` after
`python scripts/build_site.py`). Any static host works. `vercel.json` and `_headers`
send the same policy as an HTTP header. Design from Stitch; content and
wiring are generated.

## Limitations

- **A copy is not always a bug.** Real traffic repeats, and E9 shows the copies
  are mostly rows any model gets right. The finding is about the benchmark: a
  test set that is mostly repeats measures little about new attacks. Both scores
  are reported.
- **The cleaned test set is harder by construction.** E10 measures how much: a
  random removal of the same size costs up to 1.5 points (UCI Phishing, LR),
  against 15-17 on KDD99. Where the two are close, the README says so.
- **Thresholds are choices.** Near-duplicate distance (0.01 RMS in scaled
  space) and Jaccard (0.8) are fixed defaults; L2 also reports a curve and a
  training-set baseline. The identifier heuristic is advisory, and has one
  known false positive (KDD99 `dst_host_same_src_port_rate`).
- **E2's baselines are untuned and capped at 300k training rows.** Phase 2 (E5,
  E6) repeats every classic benchmark with full training sets and a tuned
  gradient-boosting model: the gaps hold (H6, H7). On the two largest current
  benchmarks, E5 scores a 200k-row test sample and trains RF/MLP on 2M rows
  (PLAN.md amendment: the full nearest-neighbour evidence would be ~10^14
  distance computations).
- **EMBER** is scanned on leakcheck's own ~680-column numeric view, not the
  official 2,381-dimension vector (EMBER's code is AGPL and is not used); exact
  duplicates use a digest of every raw feature instead.
- **CIC-IDS2017's chronological split is confounded** with unseen attack types,
  so H3 was decided on EMBER; E12 then separates the two (time within each
  attack type costs 0.5-1.5 points).
- **Two current benchmarks come from copies, not their official download pages**
  (which sit behind registration forms or served about 60 KB/s): IDS2018's copy is
  proven byte-identical to the official bucket; CIC-IoT2023's matches all 34
  published label counts, which a different file could also do.
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
.venv/Scripts/python -m pytest                       # 121 tests

python scripts/fetch.py                              # ~2.1 GB, hash-verified
python -m experiments.e0_sources && python -m experiments.e0_sources_week4
python -m experiments.e0_synthetic && python -m experiments.e0_kdd
python -m experiments.e1_audit                       # ~45 min, EMBER is 30 of them
python -m experiments.e2_retrain uci_phishing sms_spam unsw_nb15 kdd99 cicids2017 nslkdd phiusiil ember2018
python -m experiments.e4_ablation phiusiil cicids2017

# phase 3: current benchmarks (see data/README.md for the three downloads)
python -m experiments.verify_s3_etag                 # IDS2018 copies == official S3 objects
python -m experiments.e0_sources_phase3              # label counts == published tables
python -m experiments.e1_audit edgeiiot cseids2018 ciciot2023
python -m experiments.e2_retrain edgeiiot edgeiiot:no_identifiers cseids2018 cseids2018:no_identifiers ciciot2023
python -m experiments.e9_controls kdd99 nslkdd unsw_nb15 cicids2017 uci_phishing phiusiil sms_spam ember2018 edgeiiot
python -m experiments.e12_cic_within
python -m experiments.e5_predictions edgeiiot cseids2018 ciciot2023 && python -m experiments.e6_analysis
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

## Licence

Code: MIT, see [LICENSE](LICENSE). The datasets are not redistributed, except
the small example samples in `site/examples/`. Those come from UCI datasets
under CC BY 4.0, with the source named in each file.
