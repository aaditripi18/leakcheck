import numpy as np
import pandas as pd
import pytest

from experiments import e2_retrain as e2
from experiments.e1_audit import PREP, Prepared
from leakcheck import Split
from leakcheck.synth import make_split


def test_subsample_caps_and_keeps_class_ratio():
    train, test, _ = make_split(n_train=5000, n_test=100, seed=1)
    s = e2.subsample(Split(train, test, "label"), seed=0, cap=1000)
    assert len(s.train) == 1000
    assert s.train["label"].mean() == pytest.approx(train["label"].mean(), abs=0.002)
    assert len(s.test) == 100


def test_subsample_leaves_small_sets_alone():
    train, test, _ = make_split(n_train=500, n_test=100, seed=1)
    s = Split(train, test, "label")
    assert e2.subsample(s, 0, cap=1000) is s


def test_leak_masks_match_planted_counts():
    train, test, t = make_split(cross_dup=0.1, near_dup=0.05, seed=2)
    m = e2.leak_masks(Split(train, test, "label"))
    assert m["exact"].sum() == t["cross_dup_rows"]
    assert m["near"].sum() == t["cross_dup_rows"] + t["near_dup_rows"]
    assert (m["near"] | ~m["exact"]).all()  # exact implies near


def test_leak_masks_text():
    train = pd.DataFrame({"msg": ["free prize call now to claim your reward today", "see you at lunch"], "label": [1, 0]})
    test = pd.DataFrame({"msg": ["FREE prize call now to claim your reward today!", "free prize call now to claim your reward today",
                                 "totally different text here"], "label": [1, 1, 0]})
    m = e2.leak_masks(Split(train, test, "label", text="msg"))
    assert m["near"].tolist() == [True, True, False]


def test_chrono_cut():
    order = pd.Series([0] * 50 + [1] * 25 + [2] * 15 + [3] * 10)
    assert e2.chrono_cut(order, 0.8) == 1  # cumulative 0.5, 0.75, 0.9, 1.0 -> 0.75 is closest


def test_run_one_end_to_end(tmp_path, monkeypatch):
    train, test, t = make_split(n_train=1500, n_test=600, cross_dup=0.3, near_dup=0.1, seed=3)
    split = Split(train, test, "label")
    monkeypatch.setitem(PREP, "synthetic", lambda seed=0: Prepared([("all", split, ())], "official synthetic"))
    monkeypatch.setattr(e2, "OUT", tmp_path)
    res = e2.run_one("synthetic", 0)
    assert res["leak_rate"]["exact"] == pytest.approx(t["cross_dup_rows"] / 600)
    for m in e2.MODELS:
        r = res["models"][m]
        assert r["original"]["n"] == 600
        assert r["no_near"]["n"] == 600 - t["cross_dup_rows"] - t["near_dup_rows"]
        assert r["no_near"]["n"] + r["leaked"]["n"] == 600
    # a full-depth forest has memorised the exact copies
    assert res["models"]["rf"]["leaked"]["accuracy"] > 0.95
    assert (tmp_path / "synthetic_seed0.json").exists()
    assert e2.run_one("synthetic", 0) == res  # cached

    summary = e2.summarize([res])
    g = summary["macro_f1_gap"]["rf"]
    assert g["orig_minus_no_near"] > 0  # leaked rows inflate the score
