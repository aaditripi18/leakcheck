# Changelog

Versions follow `MAJOR.MINOR.PATCH`; while the major version is 0, a minor
release may change check thresholds and report fields. Experiments and their
decision rules are pre-registered in [PLAN.md](PLAN.md); results are generated
into [docs/RESULTS.md](docs/RESULTS.md).

## 1.0.0 — 2026-11-21

- `leakcheck explain`: are the leaked rows easy or memorised, judged without
  training a model (neighbour-vote statistic S, threshold s* fixed on eight dev
  cases), with an "uncertain" band inside the threshold's 90% interval.
- Four pre-registered fresh rounds (47 fresh cases) testing that diagnosis
  (H19, H19b, H22, H23) and rare classes (H20, H24); every prediction committed
  before any model was trained on its case. Phase 3 (current benchmarks, E8–E13)
  and E15 (EMBER's pre-2018 files).
- Clean benchmark splits: dropped-row lists for 40 cases, including CSE-CIC-IDS2018
  and CIC-IoT2023 with the full near search, published as release assets with
  their SHA-256 in `splits/index.json`.
- GPU near search (`LEAKCHECK_NEAR=torch`): a certified float32 matrix-product
  pass with float64 only for rows it cannot certify, repeated training rows
  searched once; identical leak decisions to the CPU search, pinned by tests on
  large norms, near and exact ties and duplicates (CI job `gpu-search`).
- Near search in parts keyed by row content, for benchmarks too large for one
  session.
- GitHub Action (`action.yml`): scan, plus explain when rows leak, with a job
  summary and a fail-on threshold.
- Fix: text near-copy candidates are chosen in a stable order, so a test row's
  result no longer depends on the other test rows.
- Security: CI job `security` (pip-audit, bandit, npm audit) fails on any known
  advisory; build requires setuptools 84.1 or later; the Action never
  glob-expands its extra arguments.

## 0.2.0 — 2026-10-16

- E2 uncertainty: seed intervals, row-level Newcombe intervals, and a paired
  test for H4 (holds by its rule on 6/6 datasets; the paired interval is above
  zero on 4/6).
- Phase 2 pre-registered: E5 (uncapped and tuned models, per-row predictions),
  E6 (threshold sweeps, cell-count bootstrap), E7 (raw UNSW-NB15, a chronological
  split without unseen attack types), hypotheses H6–H9.
- Essay page "Same copying, different damage", generated from the ledger.
- `docs/API.md`: the supported Python API, exit statuses and every check code.
- CI on Python 3.10–3.13, Linux, Windows and macOS, and the Docker image built
  and run; distributed from GitHub.
- The browser scanner's measured size range is documented (about 500,000 rows).

## 0.1.0 — 2026-10-15

First release: checks L1–L6, `scan` and `clean` (with verification), the audit
of eight security benchmarks (E1), retraining (E2), shortcut ablation (E4), and
the browser scanner with a JavaScript engine at parity with Python.
