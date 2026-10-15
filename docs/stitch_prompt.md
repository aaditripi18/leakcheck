Design a single-page, desktop-first (fully responsive down to 375px phones) project website for "leakcheck", an open-source research tool that audits security machine-learning datasets for train/test leakage. Audience: ML and security researchers, PhD admissions committees, reviewers. Tone: precise, evidence-first, like a well-made research report, not a startup landing page. No marketing language, no stock illustrations, no testimonials, no pricing.

VISUAL STYLE
- Light theme by default with an optional dark theme toggle. Off-white background, near-black text, one accent colour (amber #D97706) used only for "leak" highlights and key numbers. Green for "supported", grey for "not supported" badges.
- Fonts: Inter for text, JetBrains Mono for numbers, dataset names, feature names and code.
- Generous whitespace, max content width about 1100px, thin 1px borders, small radius (6px), no heavy shadows, no gradients.
- Every number in the page uses tabular monospaced figures. Tables are clean, zebra-free, right-aligned numbers.
- Output: plain HTML + Tailwind CSS, static, no JavaScript frameworks, no analytics, no external scripts except Google Fonts. Use semantic HTML (header, main, section, table with th scope) and accessible contrast.
- Put a data-key attribute on every element that shows a number (e.g. data-key="audit.kdd99.exact_copies"), because the numbers will later be filled from a JSON file.

SECTIONS, IN ORDER

1. Top bar: "leakcheck" wordmark in mono at left; links at right: Results, Method, Reproduce, GitHub (placeholder href "#").

2. Hero
- Eyebrow: "Security ML · dataset audit"
- Headline: "How much of a 99% score is the test set copying the training set?"
- Subhead: "leakcheck finds exact and near-duplicate rows, contradictory labels, future data and single-column shortcuts across train/test splits, then retrains the standard baselines without them."
- Big highlighted stat: "68.4%" with caption "of KDD Cup 99's test rows are exact copies of training rows. Removing leaked rows costs up to 17.5 macro-F1 points."
- Two buttons: "Read the findings" (primary), "Reproduce it" (secondary, outline).

3. Three stat cards in a row
- "8 benchmarks · 9.5M rows audited"
- "5 of 8 have ≥ 5% of test rows copying training rows"
- "0 errors recovering planted leakage across 720 synthetic configurations"

4. "The problem" — short paragraph: Papers report 97–99.9% accuracy on intrusion, malware, phishing and spam benchmarks. Many of those benchmarks put identical or near-identical rows on both sides of the split, mix future data into training, or contain a single column that gives the label away. The model then scores by recognising rows it has already seen.

5. "What it checks" — 6 small cards in a 3×2 grid, each with a mono code chip:
- L1 Exact copies — test rows identical to a training row, after canonical hashing.
- L2 Near copies — tiny-distance neighbours (tabular) or Jaccard ≥ 0.8 (text), compared with how close training rows sit to each other.
- L3 Label conflicts — identical features with different labels, and the accuracy ceiling they impose.
- L4 Time leakage — training rows newer than test rows; the timestamp as a one-feature classifier.
- L5 Shortcuts — one feature that predicts the label alone; identifier-like columns.
- L6 Group overlap — the same host, family or session on both sides of the split.
Under the grid, one line: "`leakcheck clean` removes the leakage, then re-reads its own output and re-checks every row. If anything remains, it fails."

6. "The audit" — a full-width table, title "Eight benchmarks, one audit". Columns: Dataset | Domain | Split | Test rows | Exact copies of train | Near copies (× train baseline) | Best single feature. Rows:
- KDD Cup 99 | Network intrusion | official | 311,029 | 68.36% | 14.80% (×1.37) | src_bytes 93.80%
- NSL-KDD | Network intrusion | official | 22,544 | 2.95% | 30.13% (×0.60) | src_bytes 83.01%
- UNSW-NB15 | Network intrusion | official | 82,332 | 10.37% | 7.46% (×1.35) | smean 79.13%
- CIC-IDS2017 | Network intrusion | random 80/20 | 566,149 | 14.09% | 62.67% (×0.98) | Destination Port 90.17%
- EMBER 2018 | Malware | official (temporal) | 200,000 | 0.01% | 19.95% (×0.79) | sec_entropy_max 71.69%
- UCI Phishing Websites | Phishing | random 80/20 | 2,211 | 64.41% | 0.00% | SSLfinal_State 90.33%
- PhiUSIIL | Phishing URLs | random 80/20 | 47,159 | 0.00% | 0.31% (×1.32) | URLSimilarityIndex 99.60%
- SMS Spam Collection | Spam | random 80/20 | 1,115 | 11.03% | — | —
Highlight cells ≥ 5% exact copies and ≥ 95% single feature in amber. Small footnote: "× train baseline > 1 means test rows sit closer to training than training rows sit to each other."

7. "What the leakage is worth" — a horizontal grouped bar chart built with plain HTML/CSS bars (no chart library). One row per dataset, two bars per row: Logistic regression (amber) and Gradient boosting (dark grey), value = macro-F1 points lost when leaked test rows are removed. Data (LR, HistGB):
- KDD Cup 99: 17.51, 15.00
- NSL-KDD: 11.86, 8.81
- SMS Spam Collection: 3.07, 2.49
- EMBER 2018 (random split): 2.39, 1.30
- UNSW-NB15: 2.01, 1.02
- UCI Phishing Websites: 1.51, 0.81
- CIC-IDS2017: −0.25, −0.08
- PhiUSIIL: 0.00, 0.00
Axis 0–20 points, value labels at the bar ends. Caption: "4 untuned baselines × 5 seeds; the other two models (random forest, MLP) sit between these."

8. "Five hypotheses, decided by rules fixed before the runs" — 5 cards, each with ID chip, statement, verdict badge, one-line evidence:
- H1 · At least half the benchmarks have ≥ 5% of test rows copying training rows · SUPPORTED · 5 of 8.
- H2 · Removing the copies costs ≥ 5 macro-F1 points on the two most-copied benchmarks · NOT SUPPORTED · KDD99 loses 15–17.5; UCI Phishing ≤ 1.7.
- H3 · A time-aware split costs more than deduplication · SUPPORTED · EMBER, by a small margin (0.3–0.9 points).
- H4 · Weaker models lean harder on leaked rows · SUPPORTED · 6 of 6 eligible datasets.
- H5 · Some benchmark has a single feature at ≥ 95% balanced accuracy · SUPPORTED · PhiUSIIL URLSimilarityIndex 99.6%.

9. "What we learned" — 4 finding cards in a 2×2 grid, each a bold one-line title and 2–3 sentence body:
- "Copies are not the same as inflation." UCI Phishing has as many copies as KDD99 (64% vs 68%) but loses about a point, because with 30 features of 2–3 values a copy is just a common pattern. KDD99's copies are memorised exceptions; two thirds of its gap comes from near-copies.
- "The negative control comes back clean." EMBER's temporal split: 0.009% exact copies, no training file newer than the test period, timestamp at chance. The audit does not flag everything.
- "Too easy to rank models." PhiUSIIL stays at 0.9995+ macro-F1 even after removing the four features that each reach 95% on their own.
- "A random split can hide a collapse." CIC-IDS2017 scores 0.97–0.999 on a random split and 0.46–0.57 when tested on later capture files whose attack types never appear in training.

10. "Checked against the literature" — two side-by-side panels:
- Left: a compact table comparing leakcheck with Tavallaee et al. (2009) on KDD99 redundancy. Rows: Train total 4,898,431 / distinct 1,074,992 — exact match; Train attacks 3,925,650 / 262,178 — exact; Train normal 972,781 / 812,814 — exact; Test attacks 250,436 / 29,378 — exact; Test normal 60,593 vs 60,591 / 47,913 vs 47,911 — off by 2 records. Use a small green check for exact rows and an amber "±2" chip.
- Right: synthetic validation list: "720 planted configurations, 0 counting error", "95% intervals cover the truth in 97% of runs", "MinHash text recall 100% at Jaccard ≥ 0.8, 0 false positives".

11. "Method and reproducibility" — a 2-column layout. Left: 4 short bullet points: "Every dataset download is verified against a pinned SHA-256, and third-party copies are checked against independent references." "Every run appends a line to an append-only ledger with the git commit, dataset hashes, config and numbers." "Every number on this page is generated from that ledger." "Decision rules for every hypothesis were committed before the runs they judge." Right: a dark code block (JetBrains Mono) with a copy button:
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
python scripts/fetch.py
python -m experiments.e1_audit
python -m experiments.e2_retrain kdd99 nslkdd unsw_nb15 cicids2017 ember2018 uci_phishing phiusiil sms_spam
python scripts/report.py
Caption: "Laptop CPU only. 91 tests."

12. "Limitations" — a plain bulleted list, muted text:
- A copy is not always a bug; real traffic repeats. Both scores are reported.
- The cleaned test set is harder by construction; the gap measures how much of a score was earned on copies.
- Distance and Jaccard thresholds are fixed choices; a curve and a training baseline are reported alongside.
- Baselines are untuned; training sets are capped at 300k rows.
- EMBER is scanned on leakcheck's own numeric view, not the official feature vector.
- CIC-IDS2017's chronological split is confounded with unseen attack types, so H3 was decided on EMBER.

13. Footer: "leakcheck · MIT licence (placeholder) · Built with a ledger, not a slide deck." and links GitHub, Findings, Data provenance (all href "#").
