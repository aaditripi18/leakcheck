# Stitch prompts for the leakcheck project page

Use Stitch in Web mode. Paste prompt 1 and generate. Then paste prompts 2–4 one
at a time as refinements of the same design. Export the HTML/Tailwind code at
the end and upload it to the session. Numbers get wired to `site/data.json`
afterwards, so don't worry if Stitch rounds or tweaks a figure.

---

## Prompt 1: the page (paste first)

A one-page website for "leakcheck", a research tool that audits machine-learning security datasets and shows that many famous benchmarks score high because their test set copies their training set.

Feel: an editorial research report crossed with a terminal. Calm, precise, confident. Think Stripe Press typography meets a well-designed data journalism piece. No startup clichés, no gradients, no 3D blobs, no stock imagery.

Look: warm off-white background (#FAFAF7), ink-black text, one signal colour, amber #D97706, used only where something "leaks". Inter for prose, JetBrains Mono for numbers, dataset names and code. Thin hairline rules, lots of whitespace, 1100px content width, a dark mode.

Hero, the centrepiece: on the left, a large serif-weight headline "How much of a 99% score is the test set copying the training set?" and a short line under it: "leakcheck finds duplicated rows, contradictory labels, future data and one-column shortcuts across train/test splits, then retrains the standard models without them." Two buttons: "Read the findings" and "Reproduce it". On the right, a custom illustration built from UI elements: two narrow columns of monospace data rows labelled TRAIN and TEST. About a third of the TEST rows are tinted amber, with thin amber lines connecting each to an identical row in TRAIN. Under it, a huge mono number "68.4%" with the caption "of KDD Cup 99's test rows are exact copies of training rows".

Below the hero, a slim row of three stats separated by hairlines: "8 benchmarks · 9.5M rows", "5 of 8 leak ≥ 5% of their test set", "0 errors on 720 planted-leak tests".

Then sections, each with a small mono section number (01, 02, …) and a short title: The problem · What it checks · The audit · What the leakage is worth · Five hypotheses · What we learned · Checked against the literature · Reproduce · Limitations. Sticky top bar with the "leakcheck" wordmark in mono and links Results, Method, Reproduce, GitHub.

---

## Prompt 2: the evidence sections (paste second)

Design the "The audit" and "What the leakage is worth" sections in detail.

The audit: a clean, full-width table with hairline row dividers, right-aligned mono numbers, no zebra stripes. Columns: Dataset, Domain, Split, Test rows, Exact copies, Best single feature. Rows: KDD Cup 99 · Network intrusion · official · 311,029 · 68.4% · src_bytes 93.8%; NSL-KDD · Network intrusion · official · 22,544 · 3.0% · src_bytes 83.0%; UNSW-NB15 · Network intrusion · official · 82,332 · 10.4% · smean 79.1%; CIC-IDS2017 · Network intrusion · random · 566,149 · 14.1% · Destination Port 90.2%; EMBER 2018 · Malware · official · 200,000 · 0.01% · sec_entropy_max 71.7%; UCI Phishing · Phishing · random · 2,211 · 64.4% · SSLfinal_State 90.3%; PhiUSIIL · Phishing URLs · random · 47,159 · 0.0% · URLSimilarityIndex 99.6%; SMS Spam · Spam · random · 1,115 · 11.0% · —. Values of 5% or more in Exact copies, and 95% or more in Best single feature, get a soft amber background pill. On phones the table scrolls sideways inside its own container.

What the leakage is worth: a horizontal bar chart drawn with simple bars, one row per dataset, two thin bars per row (Logistic regression in amber, Gradient boosting in charcoal), showing macro-F1 points lost when the leaked test rows are removed: KDD Cup 99 17.5 / 15.0; NSL-KDD 11.9 / 8.8; SMS Spam 3.1 / 2.5; EMBER 2.4 / 1.3; UNSW-NB15 2.0 / 1.0; UCI Phishing 1.5 / 0.8; CIC-IDS2017 0 / 0; PhiUSIIL 0 / 0. Value labels at bar ends, a 0–20 axis, and a one-line takeaway above it in large text: "Same amount of copying, very different damage: KDD99 loses 17.5 points, UCI Phishing loses 1.5."

---

## Prompt 3: verdicts and findings (paste third)

Design "Five hypotheses" as five compact horizontal cards stacked vertically, each with a mono ID chip (H1–H5), the hypothesis in one line, a verdict pill on the right (green "Supported", grey "Not supported"), and a muted one-line piece of evidence:
H1 At least half the benchmarks leak ≥ 5% of their test set: Supported (5 of 8).
H2 Removing the copies costs ≥ 5 points on the two most-copied benchmarks: Not supported (KDD99 loses 15–17.5, UCI Phishing under 2).
H3 Splitting by time hurts more than removing copies: Supported, narrowly (EMBER, 0.3–0.9 points).
H4 Weaker models lean harder on leaked rows: Supported (6 of 6 datasets).
H5 Some benchmark is solved by a single feature: Supported (PhiUSIIL, 99.6%).
Add a small note: "Decision rules were committed before the experiments ran."

Design "What we learned" as a 2×2 grid of editorial cards, each with a bold headline and two sentences:
"Copying is not the same as inflation." UCI Phishing copies as much as KDD99 but barely moves; its copies are common patterns, KDD99's are memorised exceptions.
"The control comes back clean." EMBER's time-based split shows 0.009% copies and no future data. The audit doesn't flag everything.
"Too easy to rank models." PhiUSIIL stays at 99.95% even after removing its four giveaway features.
"Random splits can hide a collapse." CIC-IDS2017 drops from 99% to about 50% when tested on attacks it never saw.

---

## Prompt 4: trust and footer (paste fourth)

Design "Checked against the literature" as two side-by-side panels. Left: a small comparison table against Tavallaee et al. (2009) on KDD99 duplicate counts, five rows, four with a green check marked "exact match" and one marked with an amber "±2 records" chip. Right: three large mono figures with captions: "720 / 720 planted leaks found exactly", "97% interval coverage", "100% text-match recall".

Design "Reproduce" as a dark terminal-style code block with a copy button, containing:
python scripts/fetch.py
python -m experiments.e1_audit
python -m experiments.e2_retrain
python scripts/report.py
with a caption "Laptop CPU only · every number on this page comes from an append-only ledger · 91 tests".

Design "Limitations" as a quiet, muted list of six short lines, deliberately understated.

Footer: the mono wordmark, the line "Built with a ledger, not a slide deck.", and links GitHub, Findings, Data provenance.
