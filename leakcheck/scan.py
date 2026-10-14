"""Run every implemented check on a split."""

from __future__ import annotations

from . import __version__
from .checks import exact, group, near, near_text, shortcut, temporal
from .findings import Report
from .split import Split

ALL_CHECKS = ("L1", "L2", "L3", "L4", "L5", "L6")


def scan(
    split: Split,
    checks: tuple[str, ...] = ALL_CHECKS,
    seed: int = 0,
    near_eps: float = near.EPS,
    text_tau: float = near_text.TAU,
    sample: int = near.SAMPLE,
) -> Report:
    findings = []
    skipped = []

    def add(name, result):
        if result is None or result == []:
            skipped.append(name)
        elif isinstance(result, list):
            findings.extend(result)
        else:
            findings.append(result)

    if "L1" in checks:
        add("L1.within", exact.check_within(split))
        add("L1.cross", exact.check_cross(split))
    if "L2" in checks:
        add("L2.near", near.check_near(split, eps=near_eps, sample=sample, seed=seed))
        add("L2.near_text", near_text.check_near_text(split, tau=text_tau, sample=sample, seed=seed))
    if "L3" in checks:
        add("L3", exact.check_conflicts(split))
    if "L4" in checks:
        add("L4", temporal.check_temporal(split))
    if "L5" in checks:
        add("L5.shortcut", shortcut.check_shortcuts(split, seed=seed))
        add("L5.identifier", shortcut.check_identifiers(split))
    if "L6" in checks:
        add("L6.group", group.check_group(split))

    meta = {
        "version": __version__,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "n_features": len(split.features),
        "label": split.label,
        "time": split.time,
        "group": split.group,
        "text": split.text,
        "ignore": list(split.ignore),
        "checks": list(checks),
        "skipped": skipped,  # not applicable: no numeric features / no --time / no --group / no --text
        "seed": seed,
        "near_eps": near_eps,
        "text_tau": text_tau,
        "sample": sample,
    }
    return Report(meta=meta, findings=findings)
