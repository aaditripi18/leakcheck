"""Command line: python -m leakcheck scan --train a.csv --test b.csv --label y"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .scan import ALL_CHECKS, scan
from .split import Split

# Exit codes: 0 nothing above low, 1 medium, 2 high, 3 usage / input error.
EXIT = {"ok": 0, "info": 0, "low": 0, "medium": 1, "high": 2}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="leakcheck")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a train/test split for leakage")
    s.add_argument("--train", required=True)
    s.add_argument("--test", required=True)
    s.add_argument("--label", required=True)
    s.add_argument("--time")
    s.add_argument("--group")
    s.add_argument("--ignore", nargs="*", default=[])
    s.add_argument("--checks", nargs="*", default=list(ALL_CHECKS), choices=ALL_CHECKS)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json", help="write the full report as JSON to this path")
    args = p.parse_args(argv)

    try:
        split = Split.from_files(
            args.train, args.test, args.label, time=args.time, group=args.group, ignore=args.ignore
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"leakcheck: {e}", file=sys.stderr)
        return 3

    report = scan(split, checks=tuple(args.checks), seed=args.seed)
    print(report.to_text())
    if args.json:
        Path(args.json).write_text(report.to_json(), encoding="utf-8")
    return EXIT[report.worst()]
