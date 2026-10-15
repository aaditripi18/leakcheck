# Findings (preliminary: E0, E1, E2 on 5 of 8 datasets)

Every number here is in [RESULTS.md](RESULTS.md), which is generated from
`results/ledger.jsonl`. E0 ran at commit `0f452ad`, E1 at `6780929`. This file
is the narrative. E2 (week 5) ran at `42caa56` on the five leakiest datasets,
with H2–H4 decision rules committed in PLAN.md before any E2 run.

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

Five of eight datasets have at least 5% of test rows exactly duplicating a
training row:

| Dataset | Split | Exact copies of a training row |
|---|---|---|
| KDD99 | official | 68.36% (attack rows 81.83%) |
| UCI Phishing Websites | random | **64.41%**, with 42.61% of training rows repeated within training |
| CIC-IDS2017 | random | 14.09% (attack rows 28.71%) |
| SMS Spam Collection | random | 11.03% |
| UNSW-NB15 | **official** | 10.37% (attack rows 15.59%) |
| NSL-KDD | official | 2.95% |
| PhiUSIIL | random | 0.00% |
| EMBER 2018 | official (temporal) | 0.009% |

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

## 11. EMBER 2018: the negative control comes back clean

EMBER's official split is temporal: train up to Oct 2018, test Nov–Dec 2018,
600k labelled training files and 200k test files. leakcheck finds what a
well-built split should look like:

- **18 of 200,000 test files (0.009%)** have raw features identical to a
  training file, measured on the digest of every raw feature, so under any
  vectorisation. There are no label conflicts.
- **0%** of training files are newer than the earliest test file, and the
  timestamp alone is at chance (50.00%).
- Near-copies: 19.95% of test rows, but **×0.79** of the training baseline, so
  test sits further from training than training sits from itself.
- No single feature passes 72% (best: max section entropy, 71.69%).

That matters for the method: the audit does not flag everything. Two facts
worth stating alongside EMBER results:

- **Family overlap (L6): 49.48% of test rows, which is essentially every malware
  test file, belong to an AVClass family also present in training** (748 shared
  families). That is not a defect, since families persist. But it means EMBER's
  test set measures "new samples of known families", not unseen families.
- **50,000 training files (8.3%) first appeared before 2018**, back to Dec
  2006. EMBER's README says 2018 samples were chosen to be harder; the older
  ones are a detail to check against the EMBER 2018 description before citing.

H1 with all eight datasets: **5 of 8** have ≥ 5% exact cross-split copies, so H1
is supported.

# E2: retraining the baselines (week 5: KDD99, UCI Phishing, CIC-IDS2017, SMS Spam, UNSW-NB15)

Four untuned baselines (LR, RF, HistGB, MLP), 5 seeds, macro-F1. Each model is
trained once and scored on the full test set and on the test rows that have no
exact or near copy in its training data. Training sets above 300k rows are
subsampled (KDD99, CIC-IDS2017), so "leaked" means a copy in the rows the model
actually saw.

## 12. H2 not supported: copies inflate scores only when they are memorised exceptions

| Dataset | Leaked test rows (exact / exact or near) | Macro-F1 gap, original → no_near (LR / RF / HGB / MLP) |
|---|---|---|
| **KDD99** | 55.9% / 78.0% | **+17.51 / +16.27 / +15.00 / +15.95** |
| SMS Spam | 11.4% / 14.0% | +3.07 / +3.95 / +2.49 / +1.82 |
| UNSW-NB15 | 10.4% / 17.9% | +2.01 / +1.16 / +1.02 / +1.51 |
| UCI Phishing | 65.4% / 65.4% | +1.51 / +1.05 / +0.81 / +1.67 |
| CIC-IDS2017 | 7.1% / 66.9% | −0.25 / −0.14 / −0.08 / −0.20 |

H2 required a ≥ 5-point gap for 3 of 4 models on **both** of the two most
affected datasets. KDD99 passes with all four models (15–17.5 points); UCI
Phishing fails with all four (≤ 1.7 points), so **H2 is not supported**.

The contrast is the finding. UCI Phishing has as many copies as KDD99, but its
30 features take only 2–3 values each, so a duplicate is a *common pattern*
that any model that generalises also gets right. KDD99's copies are a few
traffic patterns repeated hundreds of thousands of times, and its clean
remainder is dominated by the rare attacks a copy-trained model never learned.
**The exact-copy rate alone does not predict inflation.** On KDD99, two thirds
of the gap comes from *near*-copies (+4.2–4.8 points from exact copies alone).

Caveat on reading the gap: the no_near subset is, by construction, the part of
the test set unlike the training data, so it is harder than the full set. The
gap measures how much of the headline score is earned on copies rather than on
new traffic. That is the quantity a deployment cares about, but it is not
"the model is worse than reported" in every sense.

Memorisation is visible directly: on SMS Spam, RF, HistGB and MLP score
**exactly 1.0000 macro-F1 on the leaked rows in all 5 seeds**. On KDD99, all
four models score an identical 0.9238 on the leaked rows: on exact copies they
all predict the training label, so they all make the same mistakes, on the rows
whose training copies carry the opposite label (L3.cross_contradict).

## 13. H3 supported by its rule on CIC-IDS2017, but confounded

Retrained on a chronological split (train = capture files up to the boundary
closest to 80% of rows, Monday to Friday morning; test = Friday afternoon),
macro-F1 falls from 0.97–0.999 to **0.46–0.57**: gaps of 40.5–53.7 points,
against −0.1 to −0.25 for removing near-copies. 4 of 4 models, so H3's rule is
met.

But in CIC-IDS2017 each capture day runs different attacks, and Friday
afternoon holds PortScan and DDoS, which appear nowhere earlier in the week.
This "time split" is really an **unseen-attack-type split**: balanced accuracy
around 0.5–0.64 means the models mostly call the new attacks benign. That is a
real and important result: random splits of CIC-IDS2017 hide that the
detectors do not transfer to attack types absent from training. But it is not
evidence about gradual drift. **EMBER's temporal split (week 6) is the cleaner
H3 test**, so H3 is held as provisional.

## 14. H4 supported so far: weaker models lean harder on leaked rows

On every dataset where the LR or HistGB gap is at least 1 point (KDD99,
UNSW-NB15, UCI Phishing, SMS Spam), LR's gap exceeds HistGB's: 17.51 vs 15.00,
2.01 vs 1.02, 1.51 vs 0.81 and 3.07 vs 2.49. The direction is consistent but the
differences are modest. CIC-IDS2017 is excluded by the rule (both gaps below
1 point). Final verdict once NSL-KDD, PhiUSIIL and EMBER are in.

## 15. What did not move

- **CIC-IDS2017 random split**: removing 67% of test rows as near-copies leaves
  macro-F1 unchanged, even slightly higher. Near-copies here are dense benign
  traffic, not memorised attacks, which is why the L2 ratio (×0.98) said a
  random split looks like this anyway.
- **UNSW-NB15 official split**: 10.4% exact copies cost 1–2 points. Real, but
  small next to KDD99.
