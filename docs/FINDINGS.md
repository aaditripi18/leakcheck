# Findings (E0, E1, E2 on all 8 datasets, E4)

Every number here is in [RESULTS.md](RESULTS.md), which is generated from
`results/ledger.jsonl`. E0 ran at commit `0f452ad`, E1 at `6780929`. This file
is the narrative. E2 ran at `42caa56` (week 5: five leakiest datasets),
`8821fdf` (NSL-KDD) and `ad29867` (PhiUSIIL, EMBER, E4). All H2–H4 decision
rules were committed in PLAN.md before the runs they govern.

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
but not zero.

This matches what the paper says it did. Section V: "we first removed all the
redundant records in **both** train and test sets", meaning within each set.
KDDTrain+ and KDDTest+ were then sampled separately from the distinct KDD99
train and test records (Tables III and IV). The paper never mentions overlap
*between* the two sets. So this is a gap the construction does not address, not
a broken promise: NSL-KDD fixed within-set redundancy and left cross-set
copies in place.

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
**The exact-copy rate alone does not predict inflation.** On KDD99, nearly three
quarters of the gap (72%, 71.7–72.6% across the four models) comes from
*near*-copies; exact copies alone account for 4.2–4.8 points.

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

# Week 6: the remaining datasets, H3 on EMBER, and the shortcut ablation

## 16. NSL-KDD: clean on paper, still a 8–12 point gap

NSL-KDD removed KDD99's within-split repeats and has only 2.95% exact
cross-split copies. But a third of KDDTest+ (33.07%) has an exact or near copy
in KDDTrain+, and on those rows all four models score 0.98 macro-F1, against
0.63–0.73 on the rest. Removing them costs **+8.36 to +11.86 points**. The
overall L2 ratio (×0.60, FINDINGS §5) was correct about the test set *as a
whole*, but the near-copy third still carries a large share of the headline
score.

## 17. H3 supported, on its decisive test, by a small margin

On EMBER, the cleaner test, every model loses more to the official temporal
split than to removing near-copies from a random split:

| Model | Random split | no_near gap | Temporal split | time gap |
|---|---|---|---|---|
| LR | 0.8671 | +2.39 | 0.8406 | **+2.65** |
| RF | 0.9579 | +1.47 | 0.9378 | **+2.00** |
| HistGB | 0.9536 | +1.30 | 0.9349 | **+1.87** |
| MLP | 0.9562 | +1.38 | 0.9335 | **+2.27** |

4 of 4 models, so **H3 is supported** under the pre-registered rule (which
required EMBER). The margins are small: 0.26–0.89 points, against seed
standard deviations of 0.0005–0.0023, so they exceed the noise but are not
large. The honest summary: on EMBER, two months of drift cost about 2–2.7
points, slightly more than removing near-copies does. On CIC-IDS2017 the
"time" drop (40–54 points) is really an unseen-attack effect (§13).

EMBER's random split has almost no exact copies (0.01%) but 23.6% near-copies
in leakcheck's numeric view, and on those the forest scores 0.9991. A random
split of EMBER would overstate a model by about 1.3–2.4 points.

## 18. H4 supported: weaker models lean harder on leaked rows

All eight datasets are in. On every dataset where the LR or HistGB gap reaches
1 point (6 of 8: KDD99, NSL-KDD, UNSW-NB15, EMBER, UCI Phishing, SMS Spam),
LR's gap exceeds HistGB's. **6 of 6, H4 supported.** CIC-IDS2017 and PhiUSIIL
are excluded by the rule (both gaps below 1 point).

**How sure is that? (E2 uncertainty, added after the verdict, changes none.)**
The H4 rule compares two means and asks only which is bigger. Paired by seed,
LR's gap minus HistGB's has a 95% interval above zero on 4 of the 6:

| Dataset | LR gap − HistGB gap, macro-F1 points [95%, t over 5 seeds] |
|---|---|
| NSL-KDD | +3.05 [+2.67, +3.43] |
| KDD99 | +2.51 [+1.76, +3.27] |
| EMBER (random split) | +1.09 [+1.05, +1.13] |
| UNSW-NB15 | +0.99 [+0.95, +1.03] |
| UCI Phishing | +0.70 [−0.47, +1.88] |
| SMS Spam | +0.58 [−0.26, +1.43] |

So the claim that survives is narrower than "6 of 6": on the four large
benchmarks the weaker model's extra inflation is measurable; on the two small
ones (2,211 and 1,115 test rows) it cannot be told from seed noise. The same
holds for the gaps themselves: UCI Phishing's HistGB gap (+0.81) has an
interval of −0.1 to +1.7 points, and on seed 0 its leaked rows score 96.7%
against 97.0% on clean rows (Newcombe interval −1.7 to +1.4). The KDD99, NSL-KDD,
UNSW-NB15 and EMBER gaps sit far outside their intervals. The row-level
interval treats test rows as independent, which clusters of near-copies are
not, so it is if anything too narrow. Full table: RESULTS.md, "E2 uncertainty".

## 19. PhiUSIIL: there is nothing for leakage to inflate

PhiUSIIL has no exact copies and 4.47% near-copies (mostly by URL text). Every
model scores **0.9999–1.0000 macro-F1** on the full set, on the cleaned set and
on the leaked rows, so every gap is +0.00.

## 20. E4: removing the shortcut does not remove the separability

| Dataset | Dropped | Macro-F1 change (LR / RF / HGB / MLP) | Best single feature afterwards |
|---|---|---|---|
| PhiUSIIL | URLSimilarityIndex | −0.01 / −0.01 / −0.01 / −0.00 | LineOfCode 96.17% |
| PhiUSIIL | + LineOfCode, NoOfExternalRef, NoOfSelfRef | −0.03 / −0.03 / −0.02 / −0.02 | NoOfImage 93.98% |
| CIC-IDS2017 | Destination Port | −1.91 / −0.04 / −0.07 / −0.17 | Average Packet Size 86.38% |

No shortcut "carries" the score under the pre-registered rule (≥ 5 points for
3 of 4 models). What this says about PhiUSIIL is stronger than the shortcut
finding itself: **the classes are separable by many independent features**.
Remove the four features that each reach ≥ 95% on their own, and 0.9995+
macro-F1 remains. A test that easy cannot rank models: every baseline is at
the ceiling. A plausible cause is that legitimate and phishing pages were
collected from different sources and differ in page size, link counts and
images. That is a hypothesis for the dataset's documentation to confirm. It is
not shown here.

For CIC-IDS2017, Destination Port is a context shortcut that the strong models
do not need. Only LR loses (1.91 points); the tree models find the same
information in packet-size statistics.

## 22. E5/E6: the gaps survive tuning, full data and proper intervals (H6, H7 supported; H8 not)

E2 used untuned models on training sets capped at 300k rows and kept only
aggregate scores. E5 retrained every baseline on the **full** training sets
(KDD99: 4,898,431 rows; CIC-IDS2017: 2,264,594) plus a tuned HistGB (randomised
search, 12 configurations, 3-fold CV), 3 seeds, saving every test prediction; E6
computed the gaps from those predictions with a paired bootstrap over test rows
and swept the near-copy threshold.

**H6 supported.** Tuned HistGB still loses **22.16 macro-F1 points on KDD99** and
**8.63 on NSL-KDD** when the leaked rows are removed, with every seed's 95%
interval above 5 points. The KDD99 gap is larger than in E2 (15.0 for untuned
HistGB on 300k rows): with all 4.9 million training rows, 83.1% of test rows have
an exact or near copy in training, against 78% with the cap. More data, more
copies.

**H7 supported.** LR's gap exceeds tuned HistGB's on all four large benchmarks,
with the paired bootstrap interval of the difference above zero in 3 of 3 seeds
each: KDD99 +3.43, NSL-KDD +3.23, EMBER +1.50, UNSW-NB15 +0.98 points. This is
the claim of section 18, now with a tuned model and per-row intervals.

**H8 not supported (32 of 40 cells, rule: 90%).** Whether a gap is at least one
point is the same at eps 0.003, 0.01 and 0.03 for 32 of 40 dataset x model cells.
All 8 changes are gaps near one point (UNSW-NB15 and EMBER tree models, CIC-IDS2017
LR); the large gaps hold at every threshold (KDD99 tuned HistGB: +16.3 at 0.003,
+36.0 at 0.03). The rule was a poor way to state robustness, since a one-point
cut-off flips for any gap that sits near one point, and the honest summary is the
sweep itself (RESULTS, E5/E6): the size of every gap depends on how near "near"
is, and the ordering of the datasets does not.

Small benchmarks stay inconclusive: UCI Phishing's gaps (0.9-2.0 points) all have
intervals reaching below zero, and PhiUSIIL's are exactly zero (every model is at
the ceiling, section 19).

## 21. E7: on network traffic, copies matter and the clock does not (H9 not supported)

H3 was decided on EMBER because CIC-IDS2017's later days hold attack types never
seen earlier. E7 looked for a network-intrusion time split without that
confound: the raw UNSW-NB15 records (UNSW-NB15_1..4.csv from the authors' share).
They hold 2,540,047 lines, but each file ends with the record the next one begins
with; without those 3 repeats there are exactly the published 2,540,044 records,
2,218,761 normal, and the nine attack families at their published counts. Cut at
the 80th percentile of capture time, every family occurs on both sides (Worms:
122 before, 52 after).

| Model | Random split | Chronological | Cost of the time split | Cost of removing near-copies |
|---|--:|--:|--:|--:|
| LR | 97.85 | 97.70 | +0.15 | +6.46 |
| RF | 98.77 | 98.29 | +0.48 | +4.26 |
| HistGB | 98.60 | 98.27 | +0.33 | +4.43 |
| MLP | 98.39 | 98.09 | +0.31 | +5.03 |
| HistGB, tuned | 98.63 | 98.24 | +0.39 | +4.38 |

(Macro-F1, mean of 3 seeds, full training sets of about 2 million rows.)

**H9 is not supported: 0 of 5 models.** On UNSW-NB15 the clock costs less than
half a point; the copies are what flatter the score. 41% of the random split's
test rows have an exact or near copy in training, and the chronological split
does not remove them: 33% of its test rows still do (12.0% exact), because the
same traffic recurs across the capture days. A time split is therefore not a fix
for this benchmark; deduplication is. Together with EMBER (H3 supported) this
says the leak that matters depends on the data: malware drifts over time,
replayed network traffic repeats.

The usual caveat applies with force here: the no_near rows are by construction
the ones unlike training, so part of that 4-6 point gap is difficulty, not
leakage. What the number does say is how much of the headline score is earned on
traffic the model has, in effect, already seen.

**One feature does the job.** On both splits `sttl` (the source's IP TTL) alone
reaches 98-99% balanced accuracy, with `ct_state_ttl`, derived from it, close
behind (L5). That points to the testbed: the attacking and the normal hosts sent
packets with different initial TTLs. It is a measured property of this capture,
not evidence that TTL detects attacks elsewhere; it is the same kind of shortcut
as PhiUSIIL's (section 8).

## Open items before any of this is published

- How the PhiUSIIL paper defines URLSimilarityIndex. Neither the UCI page nor
  the Mendeley entry describes it; it is in the paywalled paper
  (doi:10.1016/j.cose.2023.103545).
- ~~Whether Tavallaee et al. meant NSL-KDD's de-duplication to cover train/test
  overlap.~~ Resolved: the paper removes redundancy within each set only (§4).
- The origin of EMBER 2018's 50,000 pre-2018 training files (§11).
- **Pipeline fix found in week 6:** scikit-learn's RandomForest takes NaN only
  in dense input. With a text column, the input is sparse, so missing values
  are now a sentinel for RF on text datasets. This affected only PhiUSIIL, the
  first dataset with both text and missing values, and it was fixed before its
  recorded run (`ad29867`).
