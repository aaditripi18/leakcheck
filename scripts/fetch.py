"""Download datasets listed in data/registry.json and verify their SHA-256.

  python scripts/fetch.py kdd99 nslkdd
  python scripts/fetch.py --pin NAME   # first download only: record hashes into the registry
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.data import REGISTRY, HashMismatch, fetch, load_registry  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", help="datasets (default: all)")
    p.add_argument("--pin", action="store_true", help="accept and record hashes of unpinned files")
    a = p.parse_args()
    reg = load_registry()
    for name in a.names or list(reg):
        try:
            for path in fetch(name, registry=reg, pin=a.pin):
                print(f"ok  {name}: {path}")
        except HashMismatch as e:
            print(f"FAIL {name}: {e}", file=sys.stderr)
            return 1
    if a.pin:
        REGISTRY.write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
        print(f"pinned hashes written to {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
