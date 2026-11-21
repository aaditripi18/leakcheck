"""leakcheck: find train/test leakage in security ML datasets."""

__version__ = "1.0.0"

from .findings import Finding, Report  # noqa: E402
from .split import Split  # noqa: E402
from .scan import scan  # noqa: E402

__all__ = ["Finding", "Report", "Split", "scan"]
