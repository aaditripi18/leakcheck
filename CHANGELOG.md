# Changelog

Versions follow `MAJOR.MINOR.PATCH`; while the major version is 0, a minor
release may change check thresholds and report fields. Experiments and their
decision rules are pre-registered in [PLAN.md](PLAN.md); results are generated
into [docs/RESULTS.md](docs/RESULTS.md).

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
