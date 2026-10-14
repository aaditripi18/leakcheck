# Findings (preliminary, E0 only)

Every number here is in [RESULTS.md](RESULTS.md), which is generated from
`results/ledger.jsonl` at commit `0f452ad`. This file is the narrative. E1
(the audit across all datasets) has not started; nothing below is a
hypothesis verdict yet.

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
