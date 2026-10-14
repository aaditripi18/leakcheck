"""Finding and Report: the only things checks return.

Every finding carries a single headline number (`value`), a severity, and
example row indices so any claim can be checked by hand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("high", "medium", "low", "info", "ok")
_RANK = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Finding:
    code: str  # e.g. "L1.cross"
    severity: str
    value: float
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    examples: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.severity not in _RANK:
            raise ValueError(f"unknown severity {self.severity!r}")


def grade(value: float, high: float, medium: float, low: float = 0.0) -> str:
    """Severity for a rate where larger is worse."""
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    if value > low:
        return "low"
    return "ok"


@dataclass
class Report:
    meta: dict[str, Any]
    findings: list[Finding]

    def get(self, code: str) -> Finding:
        for f in self.findings:
            if f.code == code:
                return f
        raise KeyError(code)

    def worst(self) -> str:
        if not self.findings:
            return "ok"
        return min((f.severity for f in self.findings), key=_RANK.__getitem__)

    def to_dict(self) -> dict[str, Any]:
        return {"meta": self.meta, "worst": self.worst(), "findings": [asdict(f) for f in self.findings]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_jsonable)

    def to_text(self) -> str:
        m = self.meta
        lines = [
            f"leakcheck {m.get('version', '')}: train={m['n_train']} rows, test={m['n_test']} rows, "
            f"{m['n_features']} features, label={m['label']!r}",
            "",
        ]
        for f in sorted(self.findings, key=lambda f: _RANK[f.severity]):
            lines.append(f"[{f.severity.upper():6}] {f.code:19} {f.summary}")
        lines += ["", f"worst severity: {self.worst()}"]
        return "\n".join(lines)


def _jsonable(o: Any) -> Any:
    # numpy scalars and arrays
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
