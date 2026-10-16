# Using leakcheck

The command line, the Python function behind it, and the browser scanner
(`site/scan.html`, same counts as Python) give the same findings.

## Install

```
pip install "git+https://github.com/aaditripi18/leakcheck"      # numpy, pandas, scikit-learn
docker build -t leakcheck https://github.com/aaditripi18/leakcheck.git
docker run --rm -v "$PWD:/data" leakcheck scan --train /data/train.csv --test /data/test.csv --label label
```

Python 3.10 to 3.13; tested on Linux, Windows and macOS. CPU only.

## Command line

```
leakcheck scan  --train A.csv --test B.csv --label COL [--time COL] [--group COL] [--text COL]
                [--ignore COL ...] [--checks L1 L2 ...] [--near-eps 0.01] [--text-tau 0.8]
                [--sample 20000] [--seed 0] [--json report.json]

leakcheck clean --train A.csv --test B.csv --label COL --out DIR [--near-eps E] [--text-tau T]
                [--drop-group-overlap] [--dedup-within] [--conflicts drop|majority|keep]
```

| Exit status | `scan` | `clean` |
|---|---|---|
| 0 | worst finding is info or low (or nothing found) | cleaned files written and verified |
| 1 | worst finding is medium | |
| 2 | worst finding is high | |
| 3 | input could not be read (missing file, missing column, no features) | same |
| 4 | | the written files did not re-scan clean (nothing is kept unverified) |

`clean` removes in a fixed order (conflicting rows, then repeats within a split
if asked, then test rows that copy or nearly copy the final training set), writes
`train.csv`, `test.csv` and a log with the SHA-256 of each file and every removed
row number, then reads the written files back and re-scans them.

## Python

```python
import pandas as pd
from leakcheck import Split, scan

split = Split(pd.read_csv("train.csv"), pd.read_csv("test.csv"), label="label",
              time="timestamp", group="host", text=None, ignore=["id"])
report = scan(split)                       # all checks; scan(split, checks=("L1", "L3"))
report.worst()                             # "high" | "medium" | "low" | "info" | "ok"
report.get("L1.cross").value               # share of test rows that exactly copy a training row
report.to_dict()                           # the JSON report below
```

`Split`, `scan` and `Report`/`Finding` are the supported API; the rest of the
package is internal and may change between versions.

## The report

```json
{
  "meta": {"version": "0.2.0", "n_train": 150, "n_test": 100, "n_features": 2, "label": "y", "...": "..."},
  "worst": "high",
  "findings": [
    {"code": "L1.cross", "severity": "high", "value": 1.0,
     "summary": "100.00% of test rows (100) exactly duplicate a training row",
     "details": {"...": "..."}, "examples": ["..."]}
  ]
}
```

Every finding has one headline number (`value`, a share unless the code says
otherwise), a severity from the thresholds in `leakcheck/checks/`, a one-line
summary, the counts behind it in `details`, and a few example row numbers.

## Checks and codes

| Code | `value` | What it means |
|---|---|---|
| `L1.cross` | share of test rows | test rows identical to a training row (after canonical hashing) |
| `L1.within_train`, `L1.within_test` | share of rows | rows that repeat another row of the same split |
| `L2.near` | share of (sampled) test rows | test rows within RMS distance `--near-eps` of a training row in scaled space; `details` has the curve over thresholds and how close training rows sit to each other |
| `L2.near_text` | share of test rows | text with Jaccard >= `--text-tau` to a training document (MinHash candidates, exact check) |
| `L3.conflict` | share of rows | identical features with different labels |
| `L3.ceiling` | accuracy | the best test accuracy any classifier can reach on these features |
| `L3.cross_contradict` | share of test rows | test rows whose identical training copy has the other label |
| `L4.after_test_start` | share of training rows | training rows newer than the start of the test period (`--time`) |
| `L4.time_shortcut` | balanced accuracy | the timestamp alone predicts the label |
| `L5.shortcut` | balanced accuracy | the best single feature, as a small tree, predicts the label on its own |
| `L5.identifier` | — | columns that look like identifiers (advisory; one known false positive in KDD99) |
| `L6.group` | share of test rows | test rows whose `--group` value also occurs in training |

Sampling: L2 checks up to `--sample` test rows (default 20,000) with a Wilson
interval when it samples; L5 reads up to 200,000 rows per split. Both are
deterministic for a given `--seed` and say in `details` when they sampled.
