"""The published clean-split lists agree with their manifests (scripts/splits_index.py), and the
source-numbered lists are rebuilt from the drop list (experiments/clean_splits.py)."""
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import clean_splits as cs

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("splits_index", ROOT / "scripts" / "splits_index.py")
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


def _gz(p, rows):
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("\n".join(map(str, rows)) + "\n")


def _read(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return [int(x) for x in f.read().split()]


def test_a_stale_source_list_stops_the_index(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "SPLITS", tmp_path)
    d = {"case": "x", "n_drop": 3, "n_test": 10, "source_numbering": "rows of the pinned file"}
    _gz(tmp_path / "x.drop.txt.gz", [1, 4, 7])
    _gz(tmp_path / "x.drop_source.txt.gz", [4])             # an exact-only list left from before
    with pytest.raises(SystemExit, match="x.drop_source.txt.gz: 1 rows"):
        si.check_lists("x", d)
    _gz(tmp_path / "x.drop_source.txt.gz", [11, 14, 17])
    si.check_lists("x", d)
    (tmp_path / "x.drop_source.txt.gz").unlink()             # missing where the manifest numbers rows
    with pytest.raises(SystemExit, match="None rows"):
        si.check_lists("x", d)
    si.check_lists("x", {**d, "source_numbering": None})


def test_an_empty_drop_list_counts_zero_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(si, "SPLITS", tmp_path)
    _gz(tmp_path / "y.drop.txt.gz", [])
    _gz(tmp_path / "y.drop_source.txt.gz", [])
    si.check_lists("y", {"case": "y", "n_drop": 0, "n_test": 5, "source_numbering": "file order"})


def test_sources_maps_the_drop_list_to_source_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "OUT", tmp_path)
    src = np.array([50, 10, 40, 30, 20])                     # test row i is source row src[i]

    class S:
        test = [None] * 5
    monkeypatch.setattr(cs, "split_for", lambda case: (None, S, src))
    _gz(tmp_path / "rnd.drop.txt.gz", [0, 3])
    (tmp_path / "rnd.json").write_text(json.dumps({"n_drop": 2, "n_test": 5}), encoding="utf-8")
    cs.sources("rnd")
    assert _read(tmp_path / "rnd.drop_source.txt.gz") == [30, 50]
    assert _read(tmp_path / "rnd.test_source.txt.gz") == [10, 20, 30, 40, 50]
    (tmp_path / "rnd.json").write_text(json.dumps({"n_drop": 3, "n_test": 5}), encoding="utf-8")
    with pytest.raises(AssertionError, match="differ"):
        cs.sources("rnd")
