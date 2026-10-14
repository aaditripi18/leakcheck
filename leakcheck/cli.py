"""Command line.

  python -m leakcheck scan  --train a.csv --test b.csv --label y [...]
  python -m leakcheck clean --train a.csv --test b.csv --label y --out DIR [...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .checks import near, near_text
from .clean import CONFLICT_MODES, CleanOptions, clean_split, options_dict, verify
from .scan import ALL_CHECKS, scan
from .split import Split

# Exit codes: 0 nothing above low, 1 medium, 2 high, 3 usage / input error,
# 4 clean wrote output that failed its own re-scan.
EXIT = {"ok": 0, "info": 0, "low": 0, "medium": 1, "high": 2}


def _split_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--time", help="timestamp column (L4)")
    p.add_argument("--group", help="host / family / session column (L6)")
    p.add_argument("--text", help="free-text feature for MinHash near-duplicates (L2)")
    p.add_argument("--ignore", nargs="*", default=[], help="columns to leave out of the features")


def _load(args) -> Split:
    return Split.from_files(args.train, args.test, args.label, time=args.time, group=args.group,
                            ignore=args.ignore, text=args.text)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="leakcheck")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a train/test split for leakage")
    _split_args(s)
    s.add_argument("--checks", nargs="*", default=list(ALL_CHECKS), choices=ALL_CHECKS)
    s.add_argument("--near-eps", type=float, default=near.EPS)
    s.add_argument("--text-tau", type=float, default=near_text.TAU)
    s.add_argument("--sample", type=int, default=near.SAMPLE, help="test rows checked by L2")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", help="write the full report as JSON to this path")

    c = sub.add_parser("clean", help="remove leakage, write the cleaned split, verify by re-scanning")
    _split_args(c)
    c.add_argument("--out", required=True, help="output directory")
    c.add_argument("--near-eps", type=float, help="also drop tabular near-duplicates (e.g. 0.01)")
    c.add_argument("--text-tau", type=float, help="also drop text near-duplicates (e.g. 0.8)")
    c.add_argument("--drop-group-overlap", action="store_true")
    c.add_argument("--dedup-within", action="store_true", help="drop repeated rows inside each split")
    c.add_argument("--conflicts", choices=CONFLICT_MODES, default="drop")

    args = p.parse_args(argv)
    try:
        split = _load(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"leakcheck: {e}", file=sys.stderr)
        return 3
    return _scan(args, split) if args.cmd == "scan" else _clean(args, split)


def _scan(args, split: Split) -> int:
    report = scan(split, checks=tuple(args.checks), seed=args.seed, near_eps=args.near_eps,
                  text_tau=args.text_tau, sample=args.sample)
    print(report.to_text())
    if args.json:
        Path(args.json).write_text(report.to_json(), encoding="utf-8")
    return EXIT[report.worst()]


def _clean(args, split: Split) -> int:
    opts = CleanOptions(near_eps=args.near_eps, text_tau=args.text_tau,
                        drop_group_overlap=args.drop_group_overlap,
                        dedup_within=args.dedup_within, conflicts=args.conflicts)
    result = clean_split(split, opts)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_path, te_path = out / "train.csv", out / "test.csv"
    result.train.to_csv(tr_path, index=False)
    result.test.to_csv(te_path, index=False)

    # The claim is checked on the files as written, not on the in-memory frames.
    reread = Split.from_files(tr_path, te_path, args.label, time=args.time, group=args.group,
                              ignore=args.ignore, text=args.text)
    checks = verify(reread, opts)
    ok = all(v["ok"] for v in checks.values())

    log = {
        "version": __version__,
        "options": options_dict(opts),
        "input": {"train": {"path": args.train, "sha256": sha256(args.train), "rows": len(split.train)},
                  "test": {"path": args.test, "sha256": sha256(args.test), "rows": len(split.test)}},
        "output": {"train": {"sha256": sha256(tr_path), "rows": len(result.train)},
                   "test": {"sha256": sha256(te_path), "rows": len(result.test)}},
        "removed_counts": result.counts(),
        "removed_rows": result.removed,
        "verification": {"ok": ok, "checks": checks},
    }
    (out / "clean_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print(f"train {len(split.train)} -> {len(result.train)} rows, test {len(split.test)} -> {len(result.test)} rows")
    for reason, c in result.counts().items():
        print(f"  {reason:16} train -{c['train']}, test -{c['test']}")
    for name, v in checks.items():
        print(f"  verify {name:16} {'PASS' if v['ok'] else 'FAIL'} ({v['value']})")
    print(f"wrote {tr_path}, {te_path}, {out / 'clean_log.json'}")
    return 0 if ok else 4
