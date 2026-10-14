"""Run every implemented check on a split."""

from __future__ import annotations

from . import __version__
from .checks import exact, shortcut
from .findings import Report
from .split import Split

ALL_CHECKS = ("L1", "L3", "L5")


def scan(split: Split, checks: tuple[str, ...] = ALL_CHECKS, seed: int = 0) -> Report:
    findings = []
    if "L1" in checks:
        findings += exact.check_within(split)
        findings.append(exact.check_cross(split))
    if "L3" in checks:
        findings += exact.check_conflicts(split)
    if "L5" in checks:
        findings.append(shortcut.check_shortcuts(split, seed=seed))
        findings.append(shortcut.check_identifiers(split))
    meta = {
        "version": __version__,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "n_features": len(split.features),
        "label": split.label,
        "time": split.time,
        "group": split.group,
        "ignore": list(split.ignore),
        "checks": list(checks),
        "seed": seed,
    }
    return Report(meta=meta, findings=findings)
