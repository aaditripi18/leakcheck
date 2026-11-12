"""Refit explain's threshold with an "uncertain" band, before a third fresh round (PLAN.md
amendment 2026-11-13, H22).

1. Outcome clarity. A labelled case's outcome (easy if the copy-free gap keeps >= 80% of the
   full gap) is *clear* when 0.8 lies outside the 90% bootstrap interval of that ratio over its
   12 cells (4 models x 3 seeds; 4,000 resamples, seed 0). Otherwise it is *borderline*: the
   cells do not settle which side of 0.8 it is on.
2. Threshold. s* is refit on every clear labelled case (the eight dev cases and the eligible
   cases of rounds 1 and 2) with H19's rule (scripts/h19_threshold.choose).
3. Band. The 90% bootstrap interval of s* (resampling the clear cases, 10,000 resamples, seed
   0, the same rule on each). explain says "uncertain" for S inside it.

Records `h22_band` in the ledger and writes S_THRESHOLD and S_BAND into leakcheck/explain.py.

    python scripts/h22_band.py
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.ledger import record  # noqa: E402
from scripts.h19_threshold import DEV, choose  # noqa: E402
from scripts.report import E14_FRESH, P5_FRESH, _e13, _e14, _h19_outcome, latest  # noqa: E402

CUT = 0.8


def keep_interval(cells: list[dict], n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """90% bootstrap interval of (mean copy-free gap) / (mean full gap) over a case's cells."""
    cells = [c for c in cells if c["full"]["gap"] is not None and (c.get("copy_free") or {}).get("gap") is not None]
    rng, ks = random.Random(seed), []
    for _ in range(n):
        b = [rng.choice(cells) for _ in cells]
        f = sum(c["full"]["gap"] for c in b) / len(b)
        if f > 0:
            ks.append(sum(c["copy_free"]["gap"] for c in b) / len(b) / f)
    ks.sort()
    return ks[int(0.05 * len(ks))], ks[int(0.95 * len(ks))]


def clear(lo: float, hi: float) -> bool:
    return not lo <= CUT <= hi


def labelled(runs: dict) -> list[dict]:
    out = []
    for n in DEV:
        e = _e13(runs, n)
        out.append({"case": n, "round": "dev", "S": runs[f"e14_stat:{n}"]["results"]["S"],
                    "outcome": _h19_outcome(e), "cells": e["cells"]})
    for rnd, names in (("fresh 1", E14_FRESH), ("fresh 2", P5_FRESH)):
        for n in names:
            e = _e14(runs, n)
            if e is None or _h19_outcome(e) is None:
                continue
            out.append({"case": n, "round": rnd, "S": runs[f"h19_prediction:{n}"]["results"]["S"],
                        "outcome": _h19_outcome(e), "cells": e["cells"]})
    for c in out:
        c["keep_90"] = keep_interval(c.pop("cells"))
        c["clear"] = clear(*c["keep_90"])
    return out


def band(cases: list[tuple[str, float, str]], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng, ts = random.Random(seed), []
    for _ in range(n):
        b = [rng.choice(cases) for _ in cases]
        if len({o for _, _, o in b}) == 2:
            ts.append(choose(b)[0])
    ts.sort()
    return ts[int(0.05 * len(ts))], ts[int(0.95 * len(ts))]


def main() -> None:
    runs = latest()
    lab = labelled(runs)
    for c in lab:
        lo, hi = c["keep_90"]
        print(f"  {c['case']:<32} {c['round']:<8} S {c['S']:+.4f}  {c['outcome']:<9} keep 90% [{lo:+.2f}, {hi:+.2f}]"
              f"  {'clear' if c['clear'] else 'borderline'}")
    cases = [(c["case"], c["S"], c["outcome"]) for c in lab if c["clear"]]
    t, how = choose(cases)
    lo, hi = band(cases)
    print(f"s* = {t:+.4f} ({how}; {len(cases)} clear cases); uncertain band [{lo:+.4f}, {hi:+.4f}]")
    record("h22_band", {"rule": "PLAN.md amendment 2026-11-13", "cut": CUT, "keep_resamples": 4000,
                        "band_resamples": 10_000, "seed": 0},
           {"s_star": t, "how": how, "band": [lo, hi], "cases": lab})
    p = ROOT / "leakcheck" / "explain.py"
    src = p.read_text(encoding="utf-8")
    src = re.sub(r"S_THRESHOLD: float \| None = .*", f"S_THRESHOLD: float | None = {t!r}", src, count=1)
    src = re.sub(r"S_BAND: tuple\[float, float\] \| None = .*", f"S_BAND: tuple[float, float] | None = ({lo!r}, {hi!r})",
                 src, count=1)
    p.write_text(src, encoding="utf-8")


if __name__ == "__main__":
    main()
