# leakcheck — plan

## The problem

Security ML papers routinely report 97–99.9% accuracy on intrusion-detection,
malware and phishing benchmarks. Many of those benchmarks put identical or
near-identical samples in both the training and the test split, mix future
attacks into training, or contain a single column (a timestamp, a port, a flow
ID) that gives the label away. The model then scores well by recognising samples
it has already seen, which says little about how it handles new attacks.

Arp et al. ("Dos and Don'ts of Machine Learning in Computer Security", USENIX
Security 2022) describe these pitfalls. Individual datasets have been audited
(KDD99 → NSL-KDD, Tavallaee et al. 2009; CIC-IDS2017, Engelen et al. 2021;
temporal bias in malware, Pendlebury et al. "TESSERACT" 2019; NIDS "design
smells", Flood et al. 2024 — **verify every citation before it goes on the
project page**). What is missing is a **single, reusable tool** that measures
all leakage types the same way across domains, and then reports **how much of
the published accuracy survives once the leakage is removed**.

## What we build

`leakcheck`: a Python package and CLI that runs on any train/test split.

```
python -m leakcheck scan --train train.csv --test test.csv --label label \
       [--time ts] [--group host] [--ignore id] [--json report.json]
```

| Check | What it finds | Method |
|---|---|---|
| L1 exact | identical rows within train, within test, and **across** splits | canonical row hash (floats rounded to fixed significant digits, NaN canonicalised, label column excluded) |
| L2 near | test rows that are near-copies of training rows | tabular: nearest neighbour in robust-scaled feature space, calibrated against train→train distances; text/URL: MinHash + LSH Jaccard |
| L3 conflict | identical features carrying different labels (puts a ceiling on achievable accuracy) | grouping by row hash |
| L4 temporal | training samples newer than test samples; time-aware re-split | timestamp column, TESSERACT-style split |
| L5 shortcut | a single feature that predicts the label on its own; identifier-like columns | one-feature tree trained on train, scored on test; column-name and cardinality heuristics |
| L6 group | the same host / family / session in both splits | group column overlap |

Every check returns a `Finding` with a number, a severity and example row
indices, so you can confirm any claim by hand.

**The claim leakcheck enforces on its own output:** after `leakcheck clean`,
re-running `leakcheck scan` on the cleaned split must report zero L1
cross-split duplicates and zero L3 conflicts. This round-trip is part of the
test suite and runs again for every dataset in the experiments.

## Hypotheses (pre-registered — written before running anything on real data)

- **H1** At least half of the audited benchmarks have ≥ 5% of test rows that
  exactly duplicate a training row.
- **H2** On the most affected benchmarks, removing cross-split duplicates and
  near-duplicates lowers macro-F1 of the standard baselines by ≥ 5 points.
- **H3** A time-aware split costs more accuracy than deduplication on datasets
  that have timestamps (malware in particular).
- **H4** The accuracy lost to leakage shrinks as model strength grows: the gap
  for logistic regression is larger than the gap for gradient boosting. This
  mirrors the "leakage gap collapses with model strength" finding from the
  portfolio that motivated this project. It may well fail in this domain, and
  that would be reported.
- **H5** At least one widely used benchmark has a single feature reaching
  ≥ 0.95 balanced accuracy on its own.

Each hypothesis is reported as supported or not supported, with the numbers.
None are reworded after the fact.

## Datasets (candidates; confirm licence and download for each)

| Domain | Dataset | Why |
|---|---|---|
| IDS | KDD Cup 99 → NSL-KDD | **validation**: known heavy duplication; leakcheck must rediscover it |
| IDS | UNSW-NB15 (official train/test) | widely used, official split |
| IDS | CIC-IDS2017 / CSE-CIC-IDS2018 | most cited; known labelling problems |
| IDS | CIC-IoT-2023 or TON_IoT | newer IoT benchmark |
| Malware | EMBER 2018 (features) | has timestamps → L4 |
| Malware | BODMAS | timestamps + families → L4, L6 |
| Phishing | UCI Phishing Websites | tiny feature space → expect many duplicates |
| Phishing | PhiUSIIL (2024) | recent, high reported accuracy |
| Phishing / URL | ISCX-URL2016 | URL text → MinHash path of L2 |
| Spam | SMS Spam Collection | text, known duplicate messages |

Where a dataset has no official split, use the split most papers use (check
the three most cited papers on it) and record it in `data/README.md`.

## Experiments

- **E0 validation.** (a) Synthetic: inject known exact and near-duplicate
  rates, conflicts and a planted shortcut, then check leakcheck recovers each
  rate within ±1%. (b) KDD99 vs NSL-KDD: duplicate counts should match those
  published by Tavallaee et al.
- **E1 audit.** Every L-check on every dataset → `results/audit.csv`.
- **E2 retrain.** Baselines: logistic regression, random forest, gradient
  boosting (HistGradientBoosting), MLP. Five seeds each, trained on
  original / deduplicated / near-deduplicated / time-split versions.
  Output: macro-F1 and accuracy per condition → `results/retrain.csv`.
- **E3 model strength.** Gap (original − cleaned) against model family → H4.
- **E4 shortcut ablation.** Retrain without the flagged L5 features.

Everything runs on a laptop CPU. No GPU and no Kaggle quota.

## Reproducibility discipline (same as Proof-of-Forgetting and Stowaway)

- Dataset files are verified by SHA-256 in `data/manifest.json`. Raw data is
  never committed.
- Every run writes a line to `results/ledger.jsonl`: git commit, dataset hash,
  seed, config, numbers.
- `docs/RESULTS.md` is generated by `scripts/report.py` and never edited by hand.
- `docs/FINDINGS.md` is the narrative and cites ledger rows.

## Timeline (8 weeks)

| Week | Deliverable |
|---|---|
| 1 | Package skeleton, L1 + L3 + L5, `Finding` / report / CLI, synthetic tests |
| 2 | L2 (tabular kNN + MinHash), L4, L6, `leakcheck clean` + round-trip test |
| 3 | Data downloaders + manifest; E0 validation (synthetic + KDD99/NSL-KDD) |
| 4 | E1 audit on all datasets |
| 5 | E2 retrain harness + ledger; first half of the datasets |
| 6 | E2 second half; E3, E4 |
| 7 | `report.py`, FINDINGS.md, hypotheses verdicts |
| 8 | Project page: headline table, method, limitations, reproduce-it section; optional browser demo (drop a CSV, scan locally, CSP `connect-src 'none'`) |

## Honest limitations to state up front

- An exact duplicate across splits is not always a bug: real traffic does
  repeat. The finding is that a benchmark which rewards memorising repeats does
  not measure generalisation to new attacks. Both numbers are reported.
- Near-duplicate thresholds are a choice. Results are reported for several
  thresholds, not one tuned value.
- Retraining reproduces the *standard* baselines, not every paper's exact model.

## E2 operationalisation (fixed 2026-10-15, before any E2 run)

The hypotheses above leave some choices open. These are fixed in code
(`experiments/e2_retrain.py`, `experiments/models.py`) before any E2 result
exists:

- **Seeds** 0–4. Random-split datasets are re-split per seed; official splits
  stay fixed. Training sets over 300,000 rows are stratified-subsampled per
  seed, and "leaked" always means "has a copy in the training rows the model
  actually saw".
- **Models**: untuned standard baselines (LR, RF-100, HistGB-200, MLP 128-64),
  inputs as described in `models.py`.
- **Metric**: macro-F1 on the binary task, mean over 5 seeds. "Gap" = original
  minus cleaned, in percentage points.
- **H2** is supported if, on **both** of the two most affected datasets by
  E1's exact cross-split rate (KDD99 68.4%, UCI Phishing 64.4%), the
  original → no_near macro-F1 gap is ≥ 5 points for at least 3 of the 4
  models.
- **H3** is supported on a dataset if the original → time gap exceeds the
  original → no_near gap for at least 3 of the 4 models. Tested on CIC-IDS2017
  (time = capture file; train = files up to the boundary closest to 80% of
  rows) and, in week 6, on EMBER (official temporal split vs a random split).
- **H4** is supported if the LR gap exceeds the HistGB gap (original → no_near)
  on a majority of the datasets where either gap is at least 1 point.
- Week 5 runs the five leakiest datasets (KDD99, UCI Phishing, CIC-IDS2017,
  SMS Spam, UNSW-NB15). NSL-KDD, PhiUSIIL and EMBER follow in week 6. The
  final H4 verdict waits for all eight.

## Week 6 operationalisation (fixed 2026-10-15, before any week-6 run)

- **E2 on NSL-KDD and PhiUSIIL**: same harness and rules as week 5.
- **E2 on EMBER, and its H3 test**: models see leakcheck's numeric view (about
  680 columns, not the official EMBER vector). "original" = a stratified random
  split of all 800k labelled files, with the official sizes (600k / 200k)
  re-drawn per seed; leaked rows are measured on the numeric view. "time" = the
  official temporal split (train up to Oct 2018, test Nov–Dec 2018). Training
  sets are capped at 300k as before. H3's rule is unchanged: the time gap must
  exceed the no_near gap for at least 3 of 4 models. **EMBER is the decisive H3
  test**, because CIC-IDS2017's chronological split is confounded with unseen
  attack types (FINDINGS §13). H3 is supported overall only if it holds on
  EMBER.
- **E4 shortcut ablation**: drop sets are read from each dataset's E1 ledger
  line (`experiments/e4_ablation.py`). minus_top = E1's best single feature;
  minus_flagged = all features at ≥ 0.95 plus name-matched identifier columns,
  minus the known false positive. Datasets: **PhiUSIIL** (URLSimilarityIndex;
  flagged: URLSimilarityIndex, LineOfCode, NoOfExternalRef, NoOfSelfRef) and
  **CIC-IDS2017** (Destination Port). Same 4 models, 5 seeds, macro-F1 on the
  full test set. A shortcut "carries" the score if removing it costs ≥ 5 points
  for at least 3 of 4 models.
