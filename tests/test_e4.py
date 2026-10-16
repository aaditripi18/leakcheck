
# The experiments (not the leakcheck tool) need scikit-learn >= 1.8: in 1.7,
# MLPClassifier's early stopping fails on string class labels (it calls isnan on
# the predictions). scikit-learn 1.8 needs Python >= 3.11; the published runs used 1.9.
import sklearn as _sk
from sklearn.utils.fixes import parse_version as _pv
import pytest as _pytest
pytestmark = _pytest.mark.skipif(_pv(_sk.__version__) < _pv("1.8"),
                                 reason="experiments need scikit-learn >= 1.8 (Python >= 3.11)")
import json

import pytest

from experiments import e4_ablation as e4
from experiments.e1_audit import PREP, Prepared
from leakcheck import Split
from leakcheck.synth import make_split


def _ledger(tmp_path, name, flagged, top, ids):
    line = {"experiment": f"e1_audit:{name}", "results": {"views": {"all": {"findings": {
        "L5.shortcut": {"flagged": flagged, "top": [{"feature": top, "balanced_acc": 0.99}]},
        "L5.identifier": {"columns": ids}}}}}}
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps({"experiment": "other"}) + "\n" + json.dumps(line) + "\n", encoding="utf-8")
    return p


def test_drop_sets_from_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(e4, "LEDGER", _ledger(tmp_path, "x", ["a", "b"], "a",
                                               [{"feature": "Dst Port", "reasons": ["name"]},
                                                {"feature": "Title", "reasons": ["high-cardinality text"]}]))
    assert e4.drop_sets("x") == {"full": [], "minus_top": ["a"], "minus_flagged": ["a", "b", "Dst Port"]}


def test_drop_sets_dedupes_and_skips_known_false_positives(tmp_path, monkeypatch):
    monkeypatch.setattr(e4, "LEDGER", _ledger(tmp_path, "kdd99", [], "src_bytes",
                                               [{"feature": "dst_host_same_src_port_rate", "reasons": ["name"]}]))
    assert e4.drop_sets("kdd99") == {"full": [], "minus_top": ["src_bytes"]}


def test_without_leaves_original_untouched():
    train, test, _ = make_split(shortcut=True, seed=1)
    s = Split(train, test, "label")
    s2 = e4.without(s, ["leaky"])
    assert "leaky" not in s2.features and "leaky" in s.features


def test_ablating_planted_shortcut(tmp_path, monkeypatch):
    train, test, _ = make_split(n_train=1500, n_test=600, shortcut=True, seed=4)
    split = Split(train, test, "label")
    monkeypatch.setitem(PREP, "synthetic", lambda seed=0: Prepared([("all", split, ())], "official synthetic"))
    monkeypatch.setattr(e4, "OUT", tmp_path)
    res = e4.run_one("synthetic", 0, {"full": [], "minus_top": ["leaky"]})
    full = res["conditions"]["full"]
    gone = res["conditions"]["minus_top"]
    assert full["best_single_feature_after"][0]["feature"] == "leaky"
    assert gone["best_single_feature_after"][0]["feature"] != "leaky"
    for m in e4.MODELS:
        assert full["models"][m]["macro_f1"] > gone["models"][m]["macro_f1"] + 0.1
    summary = e4.summarize([res, res])
    assert summary["macro_f1_drop_vs_full"]["minus_top"]["rf"] > 0.1
