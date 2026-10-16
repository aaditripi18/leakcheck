"""Append-only experiment ledger: one JSON line per run in results/ledger.jsonl.

Every number in docs/RESULTS.md must trace to a ledger line, and every ledger
line names the git commit and dataset hashes it came from.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from importlib.metadata import version

import leakcheck
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "ledger.jsonl"


def git_state() -> dict:
    def run(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:  # no git (e.g. a container built from an archive)
            return ""

    commit = run("rev-parse", "--short", "HEAD")
    return {"commit": commit or "unknown", "dirty": bool(run("status", "--porcelain", "--", "leakcheck", "experiments"))}


def record(experiment: str, config: dict, results: dict, datasets: dict | None = None, ledger: Path = LEDGER) -> dict:
    line = {
        "experiment": experiment,
        "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        # leakcheck's version from the package itself: the distribution is named
        # leakcheck-audit, so importlib.metadata.version("leakcheck") fails in a
        # fresh install.
        "env": {"python": platform.python_version(), "leakcheck": leakcheck.__version__,
                **{p: version(p) for p in ("numpy", "pandas", "scikit-learn")}},
        "datasets": datasets or {},
        "config": config,
        "results": results,
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, default=_jsonable) + "\n")
    return line


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
