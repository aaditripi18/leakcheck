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

### What the numbers say

- **Whether copies are memorised depends on the benchmark, and E13 can tell which.**
  Retrain without every training copy of the leaked test rows and score that
  copy-free model on the same split. On **KDD99 and NSL-KDD** it keeps the whole
  gap (16.2 and 10.1 points): the copied rows are easy traffic any model gets
  right, and removing them leaves the hard part (E9: 97% of KDD99's copied rows
  are right without their copies). On **CSE-CIC-IDS2018** (identifiers set aside)
  and **EMBER's official vectors** the gap disappears and reverses (+11.7 → -17.8
  and +1.5 → -0.2 points): the copy-free model is *worse* on the leaked rows than
  on the rest, so the score was coming from the copies themselves. Nearly three
  quarters of KDD99's gap (72%) comes from *near*-copies.
- **`leakcheck explain` gives a first look at whether copies are easy or
  memorised, without training a model.** It removes each leaked row's training
  copies and asks whether the nearest remaining row still has the right label.
  On the eight dev benchmarks that statistic separates E13's easy cases (KDD99,
  NSL-KDD, UNSW-NB15) from the memorised ones (EMBER, UCI Phishing, SMS Spam,
  IDS2018) completely. On fresh data, with every prediction committed before a
  model was trained, it was right on 7 of 10 cases with a gap to explain: every
  large gap right (TUANDROMD's 21 points, RT-IoT2022's 15), every miss at a
  cut-off. H19b is not supported (5 of 8 in the second round, against 90%).
- **Leaked is not the same as inflated.** On TON_IoT, BoT-IoT and X-IIoTID, 16
  cases leak 10-100% of their test rows, yet in 14 of them the leaked rows
  inflate macro-F1 by less than a point, and in two they are the hard rows (E14).
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

```
python -m leakcheck explain --train train.csv --test test.csv --label label [--json explain.json]
```

`explain` answers whether the leaked rows are easy or memorised, without
training a model: it removes their training copies, takes each row's nearest
remaining neighbour as a vote, and compares the vote accuracy on the leaked
rows with the rest (S). Above the threshold fixed on eight benchmarks whose
answer is known (s\* = +0.072), the leaked rows look easy; below it, memorised.

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

### Clean benchmark splits, ready to use

For 39 benchmark cases studied here, [splits/README.md](splits/README.md) lists
which test rows copy the training set, by row number in the original data, so
they can be dropped without running leakcheck. Every cleaned split was
re-checked on every row. KDD99 loses 83% of its test set this way, CIC-IDS2017
78%, NSL-KDD 33%. `python -m experiments.clean_splits write kdd99 --out DIR`
rebuilds the clean files from the pinned data.

### In CI

```yaml
- uses: aaditripi18/leakcheck@main
  with:
    train: data/train.csv
    test: data/test.csv
    label: label
    fail-on: high          # or medium, or never (report only)
```

The action writes the report to the run's summary page, adds `explain` when
rows leak, and fails the step at the chosen severity. The repository's own CI
runs it on a planted leak (must fail) and a clean split (must pass).

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

### The numbers

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
| "A copy is not always a bug" | Retrained with every training copy removed, models still get **97% of KDD99's copied test rows right** (98% NSL-KDD, 92-93% UCI Phishing); 0-1% are memorised (EMBER 6-19%). On the classic benchmarks the copies are easy rows (E9); on the current ones they are not (E13, next row) |
| "The cleaned test set is harder by construction" | Removing a random subset of the same size and class mix costs **-6 points on KDD99 and 0.5-0.7 on NSL-KDD**, against 15-17 and 8.5-11.9 for removing the copies: the gap comes from which rows go, mostly (E10) |
| "Duplicated rows may just be easy rows" | A model trained without the copies keeps the gap on KDD99/NSL-KDD (easy rows) but loses it on IDS2018 and EMBER, where it is worse on the leaked rows than on the rest (memorised copies) (E13) |
| "EMBER is scanned on leakcheck's own view" | On EMBER's official 2,381-dimension vectors (EMBER's code, run outside the repository; its published model reproduces AUC 0.996429): the same 0.009% exact copies and gaps within 0.4 points (E11) |
| "The near-copy threshold is a choice" | Every gap recomputed at eps 0.001-0.1 and Jaccard 0.6-0.95: whether a gap clears one point is the same at eps 0.003, 0.01 and 0.03 in 32 of 40 cells, and every change is a gap close to one point (E6, H8) |
| "The baselines are untuned and capped at 300k rows" | Every classic benchmark again with all its training data and a tuned gradient-boosting model, every prediction saved, a paired bootstrap: tuned HistGB still loses 22.2 points on KDD99 and 8.6 on NSL-KDD, intervals above 5 in every seed (E5/E6, H6, H7) |
| "CIC-IDS2017's time split is confounded with unseen attacks" | Split by time **within each attack type**, it costs 0.5-1.5 points: the collapse to 0.46-0.57 on the file-wise time split was the unseen attack types (E12) |

| | Hypothesis (phase 3) | Verdict |
|---|---|---|
| H10 | At least 2 of the 3 current benchmarks have >= 5% exact copies, as distributed | **not supported**: 1 of 3 (IDS2018 4.91%, just under) |
| H11 | On every current benchmark with >= 5% exact copies, removing leaked rows costs some model >= 1 point (interval above 0) | **not supported**: only CIC-IoT2023 qualifies; its largest gap is 0.98 points (LR, interval above 0 in 3 of 3 seeds) |
| H12 | Some current benchmark has a single feature at >= 95% | **supported**: Edge-IIoTset 100%, CIC-IoT2023 99.1% |
| H13 | KDD99 and NSL-KDD memorise more copied rows than UCI Phishing | **not supported**: 0.0% and 0.3% vs 3.3% |
| H14 | Wherever copies cost >= 1 point, random removal costs < 0.5 | **not supported**: 9 of 23 cells (NSL-KDD, UNSW-NB15, UCI Phishing LR at 1.5) |
| H16 | With every attack type seen in training, CIC-IDS2017's time split costs <= 2 points | **supported**: 0.5-1.5 |
| H15 | EMBER's conclusions hold on the official 2,381-dimension vectors (within 1 point) | **supported**: same exact-copy rate; gaps differ by at most 0.36 points |
| H17 | The leakage gap is difficulty, not memory (copy-free model keeps >= 80% of it) | **not supported**: 0 of 12 cells on IDS2018 (no identifiers) and on EMBER official |
| H18 | Duplicated rows are easy (copy-free model more accurate on them than on the rest) | **not supported**: 12/12 on CIC-IoT2023, 6/12 on EMBER, 0/12 on IDS2018 |

H17 and H18 were added after H13 failed (PLAN.md amendment 2026-11-05) and were
judged once on four datasets their quantities had never been computed on.

| | Hypothesis (phase 4, amendment 2026-11-09) | Verdict |
|---|---|---|
| H19 | S, with a threshold fixed on eight dev cases, predicts the E14 outcome of >= 90% of eligible fresh cases (>= 4 eligible) | **not judged**: right on 2 of 2 eligible; 17 of 19 fresh cases had a gap below one point |
| H20 | On near-perfect fresh benchmarks, leakage hides in the rare classes | **not judged**: no fresh case had macro-F1 >= 0.98 and a leaked rare class |

| H19b | The same S and s*, on a second fresh round of five UCI security benchmarks (8 cases, amendment 2026-11-12) | **not supported**: right on 5 of 8 eligible; the three misses sit at a cut-off (S 0.005 below s*; copy-free models keeping 75-78% of the gap against 80%) |
| H21 | EMBER 2018's 50,000 pre-2018 training files (all benign, found while setting up E15) do not change its results | **supported**: no model moves by half a point (+0.24, -0.06, +0.17, +0.00) |

Each round's predictions were committed and pushed before any of its models
was trained (`44f56fb` for phase 4, `9151e80` for phase 5); phase 4's E14 ran
from its commit on Kaggle, phase 5's on a laptop.

## Data provenance and method notes

- **Where the data came from.** Every dataset is pinned by SHA-256 in
  `data/`. CSE-CIC-IDS2018's ten CSVs are byte-identical to the official AWS
  bucket (multipart ETags). CIC-IoT2023's official page sits behind a
  registration form, so it was taken from a public copy; that copy matches all
  34 label counts the publishers list (46,686,579 rows). Edge-IIoTset is the
  authors' own release.
- **The two largest current benchmarks** (37M and 13M training rows) are scored
  on a stratified 200,000-row test sample, with RF and MLP trained on 2M rows;
  the nearest-neighbour evidence over every row would be ~10^14 distance
  computations (PLAN.md amendment 2026-11-04). H11's rule does not depend on
  RF or MLP.
- **The identifier check is advisory**: it names columns that look like
  timestamps, addresses or ports. On KDD99 it also names
  `dst_host_same_src_port_rate`, a rate, not an identifier.
- **Verdicts are not reworded.** A hypothesis that the data did not support is
  reported as not supported, with its numbers, next to the ones that were.
  Open questions about third-party data (how PhiUSIIL defines
  `URLSimilarityIndex`; where EMBER 2018's 50,000 pre-2018 training files come
  from) are listed at the end of [docs/FINDINGS.md](docs/FINDINGS.md).

## Reproduce

Everything runs on a laptop CPU; no GPU is needed. Downloads are verified
against pinned SHA-256 hashes and resume after dropped connections. Where each
dataset comes from, and how each copy was checked against an independent
reference, is in [data/README.md](data/README.md).

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest                       # 133 tests

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

# phase 4: easy or memorised (one process per case keeps memory flat)
python -m experiments.e14_explain stat kdd99         # ... and each of the 14 E13 datasets
python scripts/h19_threshold.py                      # s* from the eight dev cases, written into leakcheck/explain.py
python -m experiments.e14_explain predict ton_network    # ... each of the 19 fresh cases; commit before the next line
python scripts/kaggle_e14.py push-data && python scripts/kaggle_e14.py run ton_network ...   # E14 on Kaggle CPU

# phase 5: the second fresh round (UCI downloads are pinned in data/registry.json)
python -m experiments.e14_explain predict uci_spambase   # ... each of the 8 cases; commit before the next line
python -m experiments.e14_explain full uci_spambase      # ... each of the 8 cases
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
