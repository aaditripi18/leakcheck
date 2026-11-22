"""E14's per-seed run (experiments/e14_explain.py): the training sample for a too-large case, the S
reported on that sample, and the copy-free model skipped when one class is left."""
import sklearn as _sk
from sklearn.utils.fixes import parse_version as _pv
import pytest

pytestmark = pytest.mark.skipif(_pv(_sk.__version__) < _pv("1.8"),
                                reason="experiments need scikit-learn >= 1.8 (Python >= 3.11)")
import numpy as np
import pandas as pd

from experiments import e14_explain as e14
from leakcheck import Split


def _split(n_train=1200, n_test=300, copies=60, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(8)]
    tr = pd.DataFrame(rng.normal(size=(n_train, 8)), columns=cols)
    tr["y"] = pd.array(np.where(tr["f0"] > 0, "a", "b"), dtype="string")
    te = pd.DataFrame(rng.normal(size=(n_test, 8)), columns=cols)
    te["y"] = pd.array(np.where(te["f0"] > 0, "a", "b"), dtype="string")
    te.iloc[:copies] = tr.iloc[:copies].to_numpy()          # planted exact copies
    return Split(tr, te, "y")


def test_a_capped_case_trains_on_its_sample_and_reports_s_there(monkeypatch):
    monkeypatch.setattr(e14, "original_split", lambda name, seed: _split())
    monkeypatch.setitem(e14.TRAIN_CAP, "synthcase", 600)
    out = e14.run_full_seed("synthcase:no_identifiers", 0)
    assert out["n_train"] == 600 and out["n_test"] == 300
    assert 0 < out["n_leaked"] <= 60                         # only copies of rows in the sample leak
    assert "S_on_this_sample" in out
    assert set(out["models"]) == set(e14.MODELS)
    for r in out["models"].values():
        assert r["full"]["gap"] is not None and "copy_free" in r


def test_an_uncapped_case_keeps_its_training_set_and_reports_no_extra_s(monkeypatch):
    monkeypatch.setattr(e14, "original_split", lambda name, seed: _split())
    out = e14.run_full_seed("othercase", 0)
    assert out["n_train"] == 1200 and out["n_leaked"] == 60
    assert "S_on_this_sample" not in out


def test_one_class_left_after_removing_copies_trains_no_copy_free_model(monkeypatch):
    s = _split(n_train=400, n_test=100, copies=0)
    tr = s.train
    te = pd.concat([tr[tr["y"] == "b"], s.test], ignore_index=True)   # every "b" training row is copied into test
    monkeypatch.setattr(e14, "original_split", lambda name, seed: Split(tr, te, "y"))
    out = e14.run_full_seed("onecls", 0)
    assert out["copy_free_note"].startswith("the copy-free training set has a single class")
    assert all("copy_free" not in r and r["full"]["gap"] is not None for r in out["models"].values())
