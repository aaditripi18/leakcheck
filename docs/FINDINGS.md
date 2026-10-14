# Findings (preliminary: E0 + E1)

Every number here is in [RESULTS.md](RESULTS.md), which is generated from
`results/ledger.jsonl`. E0 ran at commit `0f452ad`, E1 at `6780929`. This file
is the narrative. Only H1 and H5 can be decided from E1 alone. H2–H4 need the
retraining experiments (E2–E3).

## 1. leakcheck recovers planted leakage exactly

Over 720 synthetic configurations (5 seeds × grids of cross-split copies,
near-copies, label conflicts and within-train repeats), every check returned
the planted count with **zero error**. The sampled L2 estimate's 95% interval
covered the true rate in 97% of 200 runs. MinHash text matching found 100% of
planted near-copies in every Jaccard bin from 0.80 up, with no false positives.
False positives cannot occur by design, because every reported pair is
re-verified by exact Jaccard.

## 2. It reproduces the published KDD99 redundancy figures

Against Tables I and II of Tavallaee et al. (2009), which introduced NSL-KDD:

- **Training set: all six figures match exactly**: 4,898,431 records,
  1,074,992 distinct (78.05% redundant); attacks 262,178 distinct, normal
  812,814.
- **Test set: attack figures match exactly** (250,436 records, 29,378
  distinct). Normal differs by exactly two records: the file has 60,593 normal
  records (47,913 distinct), and the paper reports 60,591 (47,911). The paper's
  test total, 311,027, is also two short of the file's 311,029. The most likely
  explanation is that the authors' copy of the test file had two fewer distinct
  normal records. This is a two-row discrepancy in a 2009 table, worth a
  footnote and nothing more.

## 3. What the paper did not measure: overlap *between* KDD99 train and test

Tavallaee et al. count repeats within each set. leakcheck also measures
overlap across sets, using features only on the binary attack/normal task:

- **68.36% of KDD99 test rows (212,612) are feature-identical to a training
  row.** For attack rows it is **81.83%**; for normal rows, 12.69%.
- **No classifier can exceed 98.17% test accuracy** on these features, because
  identical feature vectors carry both labels.
- **1.05% of test rows (3,279) match training rows that all carry the opposite
  label**, so a model that memorises training data is guaranteed to get them
  wrong.

## 4. NSL-KDD removed repeats within each split, but not across splits

**2.95% of KDDTest+ rows (664) are feature-identical to a KDDTrain+ row**
(4.36% of attack rows, 1.08% of normal rows). That is far below KDD99's 68%,
but not zero. Re-checking the paper's wording on whether cross-set repeats were
meant to be removed is a to-do before stating this publicly.

## 5. Calibration note: L2 near-duplicate severity needs its baseline

On NSL-KDD, 30.13% of test rows lie within RMS distance 0.01 of a training row,
which the absolute threshold rates **high**. But **50.45%** of held-out
*training* rows are that close to other training rows. So the test set is
*further* from training than a random split would be (×0.60 of baseline).
NSL-KDD's test set was deliberately built from harder records and new attacks,
which fits that.

This was found during E0, before any E1 run. leakcheck now reports the ratio
(`vs_train_baseline`) next to the absolute rate. The severity rule is
**unchanged** so that no threshold is tuned on data. E1 reports must quote L2
with its ratio, and a near-duplicate claim needs both a high absolute rate and
a ratio above 1.

# E1: the audit

Eight benchmarks, binary task. Official splits are used for KDD99, NSL-KDD,
UNSW-NB15 and EMBER; the others use a stratified random 80/20 split with seed 0.
The split policy was fixed in code before any E1 result existed.

## 6. H1 supported: exact train/test copies are the norm, not the exception

Five of seven datasets audited so far (EMBER pending) have at least 5% of test
rows exactly duplicating a training row:

| Dataset | Split | Exact copies of a training row |
|---|---|---|
| KDD99 | official | 68.36% (attack rows 81.83%) |
| UCI Phishing Websites | random | **64.41%**, with 42.61% of training rows repeated within training |
| CIC-IDS2017 | random | 14.09% (attack rows 28.71%) |
| SMS Spam Collection | random | 11.03% |
| UNSW-NB15 | **official** | 10.37% (attack rows 15.59%) |
| NSL-KDD | official | 2.95% |
| PhiUSIIL | random | 0.00% |

A random split cannot add duplication; it only shows the duplication already in
the dataset. The UCI Phishing figure follows from its feature space: 30
features with 2–3 values each, which 11,055 sites must share. UNSW-NB15 matters
more, because its overlap sits in the **official** split that published results
are measured on.

## 7. Official splits that sit closer than random

L2 compares test rows' distance to training with training rows' distance to
each other (×1 = what a random split would give). On the official splits,
KDD99 (×1.37) and UNSW-NB15 (×1.35) put test rows *closer* to training than a
random split would, on top of their exact copies. NSL-KDD (×0.60) goes the
other way, as designed. The random-split CIC-IDS2017 sits at ×0.98, as
expected. On PhiUSIIL, 3.98% of test URLs are near-copies (Jaccard ≥ 0.8) of a
training URL, with no exact copies.

## 8. H5 supported: one feature can do the whole job

**PhiUSIIL: `URLSimilarityIndex` alone reaches 99.60% balanced test accuracy**
(a 32-leaf tree on that one column). Four more single features exceed 95%
(`LineOfCode` 96.17%, `NoOfExternalRef` 96.12%, `NoOfSelfRef` 95.36%). Going by its
title, the PhiUSIIL paper proposes a "similarity index" as the core of its
framework. Before near-perfect accuracy on this dataset is credited to a model,
check whether the model only re-learns that one column. (To verify: how the
paper defines URLSimilarityIndex.) Elsewhere, the best single features are high but below the threshold:
KDD99 `src_bytes` 93.80%, UCI Phishing `SSLfinal_State` 90.33%, CIC-IDS2017
`Destination Port` 90.17%. Destination Port is a context shortcut rather than
behaviour: in CIC-IDS2017 each attack scenario targets a fixed service, so the
port identifies the scenario.

## 9. Accuracy ceilings

Identical features carrying both labels cap test accuracy at 98.17% on KDD99
and 99.32% on UCI Phishing. In those two datasets, 1.05% and 1.04% of test
rows have training copies that all carry the *opposite* label.

## 10. Changes made during E1 setup (before any recorded E1 run)

- **L5 identifier heuristic tightened** after a smoke run on SMS, UCI Phishing
  and UNSW-NB15. It flagged the SMS text column, the binary flag
  `having_IP_Address` and UNSW's `ct_dst_*` counters. Now: no `src`/`dst`/`flow`
  name tokens, at least 50 distinct values for a name match, and the text column
  is exempt. It still flags KDD's `dst_host_same_src_port_rate`, a rate rather
  than a port. That is a known false positive, left as is so the heuristic is not
  tuned on audit results. The identifier check is advisory and feeds no
  hypothesis.
- **CIC-IDS2017 labels:** the "Web Attack – XSS" dash is a non-ASCII byte;
  the loader replaces it with a space. Counts then match the published table
  exactly.

## 11. EMBER 2018

Pending (download in progress).
