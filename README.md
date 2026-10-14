# leakcheck

Security ML benchmarks report 99% accuracy. leakcheck measures how much of that
comes from the test set overlapping the training set: exact and
near-duplicates, contradictory labels, future data in training, and single
columns that give the label away.

```
python -m leakcheck scan --train train.csv --test test.csv --label label --json report.json
```

Exit code: 0 clean or minor, 1 medium findings, 2 high findings, 3 bad input.

| Check | Status |
|---|---|
| L1 exact duplicates (within and across splits, per class) | done |
| L3 label conflicts, test accuracy ceiling, cross-split contradictions | done |
| L5 single-feature shortcuts, identifier-like columns | done |
| L2 near-duplicates (kNN, MinHash) | week 2 |
| L4 temporal leakage | week 2 |
| L6 group overlap | week 2 |
| `leakcheck clean` + re-scan round trip | week 2 |

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
