# leakcheck

Security ML benchmarks report 99% accuracy. leakcheck measures how much of that
comes from the test set overlapping the training set: exact and
near-duplicates, contradictory labels, future data in training, and single
columns that give the label away.

```
python -m leakcheck scan  --train train.csv --test test.csv --label label        [--time ts] [--group host] [--text url] [--ignore id] [--json report.json]

python -m leakcheck clean --train train.csv --test test.csv --label label --out cleaned/        [--near-eps 0.01] [--text-tau 0.8] [--drop-group-overlap] [--dedup-within]        [--conflicts drop|majority|keep]
```

`scan` exit code: 0 clean or minor, 1 medium findings, 2 high findings, 3 bad input.

| Check | What it reports |
|---|---|
| L1 exact | rows repeated within train / within test; test rows identical to a training row (per class) |
| L2 near (tabular) | test rows within an RMS distance of a training row, after signed-log + robust (median/IQR) scaling; categorical values must match exactly; exact nearest neighbour; 95% Wilson interval when sampled; same curve for held-out training rows as a baseline |
| L2 near (text) | `--text` column: MinHash + LSH over 5-character shingles, every reported pair verified by exact Jaccard |
| L3 conflict | identical features with different labels; the test-accuracy ceiling this forces; test rows whose training copies all disagree |
| L4 temporal | training rows newer than the earliest / median / latest test row (TESSERACT constraint C1); per-class time ranges; timestamp as a one-feature classifier |
| L5 shortcut | best one-feature classifier (balanced test accuracy); identifier-like columns |
| L6 group | test rows whose host / family / session also appears in training |

### `clean` proves its own output

`clean` resolves conflicts, optionally removes within-split repeats, then drops
test rows that leak from the final training set. It writes `train.csv`,
`test.csv` and `clean_log.json`, then **re-reads the written files and
re-checks every row** (no sampling). If any check it was asked to enforce still
finds leakage, it exits with code 4. The log records SHA-256 of inputs and
outputs, the options, and the input row number of every removed row with the
reason.

### Speed (laptop CPU)

| Workload | Time |
|---|---|
| L1 + L3 + L5, 1M train / 300k test, 40 features | 31 s |
| L2 tabular, same split, 20k test rows sampled | 59 s |
| L2 text, 120k / 40k URLs | 16 s |

`clean --near-eps` checks every test row, so expect roughly 30 s per 10k test
rows against a 1M-row training set.

## Status

- Weeks 1–4 done. E1 audit of 8 benchmarks (KDD99, NSL-KDD, UNSW-NB15,
  CIC-IDS2017, EMBER 2018, UCI Phishing, PhiUSIIL, SMS Spam): 5 of 8 have
  ≥ 5% of test rows exactly copying a training row (H1 supported); PhiUSIIL
  has a single feature at 99.6% balanced accuracy (H5 supported); EMBER's
  temporal split comes back clean (negative control).
- Weeks 1–3: E0 validation: exact recovery of planted leakage (0 error
  over 720 configurations), and the KDD99 redundancy figures of Tavallaee et
  al. (2009) reproduced exactly for the training set and to within 2 records
  for the test set. See [docs/RESULTS.md](docs/RESULTS.md) (generated) and
  [docs/FINDINGS.md](docs/FINDINGS.md).
- Data provenance: [data/README.md](data/README.md).

```
python scripts/fetch.py
python -m experiments.e0_sources && python -m experiments.e0_sources_week4
python -m experiments.e0_synthetic && python -m experiments.e0_kdd
python -m experiments.e1_audit      # ~45 min on a laptop CPU, EMBER is 30 of them
python scripts/report.py
```

See [PLAN.md](PLAN.md) for hypotheses, datasets, experiments and timeline.

## Development

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

Tests use synthetic splits (`leakcheck/synth.py`) where every planted
duplicate, conflict and shortcut is counted, and the checks must recover the
counts exactly.
