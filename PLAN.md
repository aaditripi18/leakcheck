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

## Phase 2 (written 2026-10-16, before any of its runs)

E2 used four untuned baselines on training sets capped at 300k rows, scored with
one near-copy threshold, and kept only aggregate scores. Phase 2 removes those
limits.

**E5. Predictions, uncapped and tuned.** For every dataset and seeds 0-2: the four
E2 baselines trained on the *full* training set, plus `hgb_tuned` (randomised
search, 12 configurations of learning rate, depth, leaf count, L2 and iterations,
3-fold CV macro-F1 on a stratified 200k-row subsample of the training set, then
refit on all of it). Every test row's prediction is saved with its exact-copy
flag, nearest-neighbour RMS distance and best text Jaccard, so every later number
is computed from saved predictions without retraining.

**E6. Analysis of E5.** Near-copy threshold swept over eps in {0.001, 0.003, 0.01,
0.03, 0.1} and text Jaccard in {0.6, 0.7, 0.8, 0.9, 0.95}; macro-F1 gap
(original - clean) per model with a paired bootstrap over test rows (1,000
resamples, 95%); LR - hgb_tuned difference by the same bootstrap.

Hypotheses and decision rules:

- **H6** Tuning and full training data do not remove the leakage gap: on KDD99 and
  NSL-KDD, `hgb_tuned` loses >= 5 macro-F1 points (eps 0.01, mean of 3 seeds) with
  a bootstrap interval above 5 on at least 2 of 3 seeds.
- **H7** H4 survives a strong model and proper intervals: LR's gap exceeds
  `hgb_tuned`'s with the bootstrap interval of the difference above zero on at
  least 3 of the 4 large benchmarks (KDD99, NSL-KDD, UNSW-NB15, EMBER random split),
  in at least 2 of 3 seeds each.
- **H8** The conclusions do not hinge on the threshold: for each dataset and model,
  whether the gap (mean of seeds) is >= 1 point is the same at eps 0.003, 0.01 and
  0.03. Supported if this holds for at least 90% of dataset x model cells.

### E7. UNSW-NB15 raw records: a chronological split without unseen attack types (written before the raw files are read)

CIC-IDS2017's chronological split is confounded: its later capture day holds
attack types that never occur earlier (FINDINGS 13), so H3 was decided on EMBER.
The raw UNSW-NB15 records (UNSW-NB15_1..4.csv from the authors' share, with
capture timestamps) can give a network-intrusion time split in which the attack
families occur on both sides.

- **Provenance check first.** The four files must hold 2,540,044 records, the
  49 features listed in NUSW-NB15_features.csv, and the nine attack families named
  in the official description; otherwise E7 does not run.
- **Split.** Records sorted by `stime`; train = the earliest 80%, test = the rest
  (the cut at the 80th percentile of `stime`). If any of the nine families has
  fewer than 50 records on either side, the split is reported as confounded and
  E7 is not used for H9.
- **Runs.** leakcheck audit (E1 checks) on that split and on a stratified random
  80/20 split of the same records; E5-style models (lr, rf, hgb, mlp, hgb_tuned;
  3 seeds; full training set) on both splits; macro-F1 gap random -> chronological
  and original -> no_near on the random split.
- **H9** A chronological split costs more accuracy than removing near-copies on
  network-intrusion data too: the random -> chronological macro-F1 drop exceeds
  the original -> no_near drop for at least 3 of the 5 models (mean of 3 seeds).

## Phase 3 (written 2026-11-02, before any of its data is downloaded)

The eight audited benchmarks are the classic ones (KDD Cup 99 from 1999,
CIC-IDS2017). Current intrusion-detection papers mostly report on newer
datasets. Phase 3 asks whether the same problems are in the benchmarks people
use now, with the same checks, splits and models as phases 1 and 2.

**E8. Current benchmarks.** Three datasets, each from its maintainers' own
release, checksummed and validated against published counts before use:

- **CSE-CIC-IDS2018** (CIC and the Communications Security Establishment): the
  "Processed Traffic Data for ML Algorithms" CSVs in the public AWS bucket
  `s3://cse-cic-ids2018` (registry.opendata.aws/cse-cic-ids2018).
- **Edge-IIoTset** (Ferrag et al., IEEE Access 2022): `DNN-EdgeIIoT-dataset.csv`
  from the authors' Kaggle dataset `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`.
- **CIC-IoT2023** (Neto et al., Sensors 2023): the CSVs of the UNB release. If
  the only download is behind a registration form, a mirror is used only if it
  matches the published per-class counts; otherwise the dataset is dropped and
  the reason recorded.

None ships an official train/test split, so each gets the CIC-IDS2017 rule:
stratified random 80/20 by attack label, seed 0. Flow identifiers (IPs, ports,
timestamps, flow IDs) are kept as features exactly as distributed, because that
is how the published ML files are used; L5 reports whether any of them is a
shortcut. E1 (audit), E2 (four untuned baselines, 300k cap) and E5/E6 (full
training sets, `hgb_tuned`, three seeds, paired bootstrap) run unchanged.

- **H10** At least 2 of the 3 current benchmarks have >= 5% of test rows that
  exactly copy a training row (H1's rule).
- **H11** On every current benchmark with >= 5% exact copies, removing leaked
  test rows (exact and near, eps 0.01) costs at least one model >= 1 macro-F1
  point, with the paired bootstrap interval above zero in >= 2 of 3 seeds.
- **H12** At least one current benchmark has a single feature reaching >= 0.95
  balanced accuracy on its own (H5's rule).

Verdicts are reported whatever they are, next to H1, H2 and H5 for the classic
benchmarks.

**Amendment (2026-11-02, after the Edge-IIoTset E1 run, before the other two
datasets are read).** Edge-IIoTset keeps per-packet identifiers (`frame.time`,
IP addresses, ports), and `frame.time` alone makes almost every row unique, so
an exact-copy rate measured with them in says little about whether the same
traffic sits on both sides. E1 therefore also audits a second view of each
current benchmark, **`no_identifiers`**: the same split with the columns
leakcheck's own L5 identifier check flags removed (the rule is the tool's, not
chosen per dataset). This view is post hoc and is reported as such; H10-H12 are
judged on the pre-registered view as distributed, and the second view is
reported beside it.

## Phase 3b (written 2026-11-02, before any of its runs): turning the caveats into measurements

Four statements the README makes as caveats each get an experiment.

**E9. Are the copies memorised, or would they be predicted anyway?** For every
dataset and model (the four E2 baselines, seeds 0-2, E2's training cap): train
once on the training set and once on the training set with every row that
exactly or nearly (eps 0.01) copies a test row removed; score both on the
leaked test rows only. A leaked row is *memorised* if the full model gets it
right and the deduplicated model does not. Reported per dataset: the share of
leaked rows that are memorised, and the share predicted correctly either way.
- **H13** On KDD Cup 99 and NSL-KDD, more leaked rows are memorised than on UCI
  Phishing (the benchmark whose copies cost under 2 points).

**E10. A random-removal control for the gap.** For every dataset, model and seed
of E2: remove from the test set a random subset of rows the same size and class
mix as the leaked rows (seed 0-2) and score what is left. The leakage gap is
reported next to this control gap.
- **H14** On every dataset with a leakage gap >= 1 point, the random-removal gap
  is < 0.5 points (the gap comes from which rows were removed, not from removing
  rows).

**E11. EMBER on the official feature vector.** EMBER's own code (AGPL) is run
unmodified in a separate environment, outside this repository, to produce the
official 2,381-dimension vectors; nothing of it is copied here. L1-L5 and E2
then run on those vectors.
- **H15** The EMBER conclusions do not depend on the view: exact-copy rate and
  E2 gap on the official vectors are within 1 point of leakcheck's own view.

**E12. CIC-IDS2017 without the unseen-attack confound.** A chronological split
*within each label*: for every label, the earliest 80% of its flows (by capture
file, then row order) train and the latest 20% test, so every test attack type
also appears in training. E2 models score this split and the random split.
- **H16** Once every attack type is seen in training, the time split costs no
  more than 2 macro-F1 points relative to the random split (i.e. CIC-IDS2017's
  collapse in E2 was the unseen attack types, not time).

**Note (2026-11-03, after the first E9 run).** The first E9 implementation
removed one nearest training copy per leaked test row per round, for at most 20
rounds, re-fitting the scaling each round. On KDD Cup 99 that left 148k leaked
rows with a training copy within eps (also 66 NSL-KDD, 2,261 UNSW-NB15, 10-18
PhiUSIIL), so it did not do what the procedure says ("every row that exactly or
nearly copies a test row"). It now removes every training row within eps of a
leaked row in one radius search, in the scaled space in which E2 called the row
leaked, and checks that none is left. E9/E10 are rerun on every dataset with this
implementation; the first run is reported as superseded. Its E10 numbers do not
depend on the change.

**Amendment (2026-11-04, before E5 runs on any current benchmark).** E5 as
specified does not fit two of the current benchmarks on this machine. Its
per-row evidence finds every test row's nearest training row: for CIC-IoT2023
that is 9.3M test rows against 37M training rows, and for CSE-CIC-IDS2018 3.2M
against 13M (on the order of 10^14-10^15 distance computations), and a random
forest or MLP on 13-37M rows exceeds the memory and time available. For these
two only:
- every model is scored on, and per-row evidence is computed for, a stratified
  random sample of 200,000 test rows (by label, seed = the run's seed);
- lr, hgb and hgb_tuned train on the full training set; rf and mlp train on a
  stratified 2,000,000-row sample of it (seed = the run's seed).
Edge-IIoTset (1.8M training rows) runs exactly as specified. H11 is judged on
these runs; its rule ("some model") does not depend on rf or mlp.

**Note on E11 (2026-11-04, before any E11 run).** EMBER's code (elastic/ember @
d97a0b5) was installed in a separate environment outside the repository. Its files
are unmodified, but it does not run against current scikit-learn: it passes one
string per sample to `FeatureHasher(input_type="string")`, which 2019 versions
hashed character by character and current versions reject. The driver
(`_ember_official/vectorize.py`) restores the 2019 behaviour before importing
EMBER (a string sample becomes the list of its characters). The vectors are
accepted only because EMBER's own published 2018 model scores ROC AUC 0.996429 on
their test set, the figure EMBER's notebook reports (0.9964289468); the four files
are pinned by SHA-256 in `experiments/ember_official.py`.

## Amendment 2026-11-05: are duplicated test rows simply easy? (E13, H17-H18)

Written before any E13 number exists. Motivation: E9 (H13, not supported) found that on
KDD Cup 99 most leaked test rows are classified correctly *whether or not* their training
copies are present ("correct either way"). That suggests a different explanation for
the leakage gap: duplicated rows are easy rows (floods of near-identical attack traffic),
so removing them from the test set makes the test harder, not just less leaked. E9 never
measured the copy-free model on the *other* test rows, so it could not test this.

**E13. Difficulty of the leaked rows.** Same splits, seeds (0-2), models and copy
removal as E9. For the full model and the copy-free model (trained without every
training row within eps of a leaked test row): macro-F1 on the whole test set and on the
test set without its leaked rows, and accuracy on the leaked rows and on the rest.

- **Dev** (method checked, may be inspected): the phase-3b E9 datasets (kdd99, nslkdd,
  uci_phishing, edgeiiot and the rest of E9's list).
- **Fresh** (judged once; E13's quantities have never been computed on them):
  cseids2018, cseids2018:no_identifiers, ciciot2023, ember2018_official.

Hypotheses (12 cells per dataset = 4 models x 3 seeds):

- **H17 (the gap is difficulty, not memory)** On every fresh dataset whose full-model
  leakage gap is >= 1 point, the copy-free model's gap (macro-F1 whole test minus
  without leaked rows) is at least 80% of the full model's gap in >= 10 of 12 cells.
- **H18 (duplicated rows are easy)** On every fresh dataset with >= 1% leaked test rows,
  the copy-free model's accuracy on the leaked rows is higher than on the remaining
  test rows in >= 10 of 12 cells.

A fresh dataset that does not meet a hypothesis's entry condition is reported, not
counted. Both verdicts are reported as supported or not supported, as all others.
