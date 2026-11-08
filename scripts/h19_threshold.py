"""Fix H19's threshold s* on the eight eligible dev cases (PLAN.md amendment 2026-11-09).

Reads each dev case's E13 outcome (easy if the copy-free gap is >= 80% of the full gap) and
its S (`e14_stat:<case>`), applies the pre-registered rule, records `h19_threshold` in the
ledger and writes s* into leakcheck/explain.py (S_THRESHOLD), which `leakcheck explain` uses.

    python scripts/h19_threshold.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import record  # noqa: E402
from scripts.report import _e13, latest  # noqa: E402

DEV = ["kdd99", "nslkdd", "unsw_nb15", "ember2018", "ember2018_official", "uci_phishing", "sms_spam",
       "cseids2018:no_identifiers"]


def outcome(e: dict) -> str | None:
    """None if not eligible (full gap < 1 point), else easy / memorised."""
    if e["full_gap"] < 0.01:
        return None
    return "easy" if e["copy_free_gap"] >= 0.8 * e["full_gap"] else "memorised"


def choose(cases: list[tuple[str, float, str]]) -> tuple[float, str]:
    """The pre-registered rule: midpoint between the classes if S separates them (easy
    above), else the cut with the fewest dev errors, ties broken by the widest margin."""
    easy = [s for _, s, c in cases if c == "easy"]
    mem = [s for _, s, c in cases if c == "memorised"]
    if min(easy) > max(mem):
        return (min(easy) + max(mem)) / 2, "separates: midpoint between lowest easy and highest memorised S"
    vals = sorted(s for _, s, _ in cases)
    cuts = [(a + b) / 2 for a, b in zip(vals, vals[1:])]

    def errors(t):
        return sum((s > t) != (c == "easy") for _, s, c in cases)

    def margin(t):
        return min(abs(s - t) for _, s, _ in cases)

    best = min(cuts, key=lambda t: (errors(t), -margin(t)))
    return best, f"no separation: fewest dev errors ({errors(best)}), widest margin"


def main() -> None:
    runs = latest()
    cases = []
    for n in DEV:
        e, st = _e13(runs, n), runs.get(f"e14_stat:{n}")
        if e is None or st is None:
            sys.exit(f"{n}: needs e13_difficulty and e14_stat in the ledger")
        c = outcome(e)
        if c is None:
            sys.exit(f"{n}: not eligible (full gap {e['full_gap']:.4f}); the amendment lists it as eligible")
        cases.append((n, st["results"]["S"], c))
        print(f"  {n:<28} S {st['results']['S']:+.4f}  {c}")
    t, how = choose(cases)
    print(f"s* = {t:+.4f} ({how})")
    record("h19_threshold", {"dev": DEV, "rule": "PLAN.md amendment 2026-11-09"},
           {"s_star": t, "how": how, "cases": [{"case": n, "S": s, "outcome": c} for n, s, c in cases]})
    p = ROOT / "leakcheck" / "explain.py"
    src = p.read_text(encoding="utf-8")
    src = re.sub(r"S_THRESHOLD: float \| None = .*", f"S_THRESHOLD: float | None = {t!r}", src, count=1)
    p.write_text(src, encoding="utf-8")


if __name__ == "__main__":
    main()
