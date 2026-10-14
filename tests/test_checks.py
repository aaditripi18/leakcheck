import json

import pandas as pd
import pytest

from leakcheck import Split, scan
from leakcheck.checks import exact, shortcut
from leakcheck.cli import main
from leakcheck.synth import make_split


def split_of(**kw):
    train, test, truth = make_split(**kw)
    return Split(train, test, "label"), truth


# --- L1 -----------------------------------------------------------------------


def test_clean_split_has_no_duplicates():
    s, _ = split_of(seed=1)
    f = exact.check_cross(s)
    assert f.value == 0.0 and f.severity == "ok"
    assert all(g.value == 0.0 for g in exact.check_within(s))


@pytest.mark.parametrize("rate", [0.003, 0.02, 0.1, 0.5])
def test_cross_rate_recovered_exactly(rate):
    s, truth = split_of(cross_dup=rate, seed=2)
    f = exact.check_cross(s)
    assert f.details["n_hit"] == truth["cross_dup_rows"]
    assert f.value == pytest.approx(truth["cross_dup_rate"])


def test_cross_severity_thresholds():
    assert exact.check_cross(split_of(cross_dup=0.1)[0]).severity == "high"
    assert exact.check_cross(split_of(cross_dup=0.02)[0]).severity == "medium"
    assert exact.check_cross(split_of(cross_dup=0.003)[0]).severity == "low"


def test_cross_examples_point_at_real_matches():
    s, _ = split_of(cross_dup=0.05, seed=3)
    f = exact.check_cross(s)
    feats = s.features
    for ex in f.examples:
        assert (s.test.loc[ex["test"], feats] == s.train.loc[ex["train"], feats]).all()


def test_within_train_rate_recovered():
    s, truth = split_of(train_dup=0.2, seed=4)
    f = next(g for g in exact.check_within(s) if g.code == "L1.within_train")
    assert f.value == pytest.approx(truth["train_dup_rows"] / truth["n_train"])


def test_per_class_rates_sum_to_total():
    s, _ = split_of(cross_dup=0.1, n_classes=3, seed=5)
    f = exact.check_cross(s)
    total = sum(v["n"] * v["rate"] for v in f.details["per_class"].values())
    assert total == pytest.approx(f.details["n_hit"])


# --- L3 -----------------------------------------------------------------------


def test_conflicts_recovered_exactly():
    s, truth = split_of(conflict=0.05, seed=6)
    f = {g.code: g for g in exact.check_conflicts(s)}
    assert f["L3.conflict"].details["n_groups"] == truth["conflict_pairs"]
    assert f["L3.conflict"].details["n_rows"] == truth["conflict_rows"]


def test_no_conflicts_on_clean_split():
    s, _ = split_of(seed=7)
    f = {g.code: g for g in exact.check_conflicts(s)}
    assert f["L3.conflict"].value == 0
    assert f["L3.ceiling"].value == 1.0
    assert f["L3.cross_contradict"].value == 0


def test_ceiling_hand_computed():
    # Test set: three identical rows labelled a, a, b -> best is 2 of 3; plus one unique row.
    train = pd.DataFrame({"x": [9.0], "label": ["a"]})
    test = pd.DataFrame({"x": [1.0, 1.0, 1.0, 2.0], "label": ["a", "a", "b", "b"]})
    f = {g.code: g for g in exact.check_conflicts(Split(train, test, "label"))}
    assert f["L3.ceiling"].value == pytest.approx(3 / 4)


def test_cross_contradiction():
    train = pd.DataFrame({"x": [1.0, 1.0, 2.0], "label": ["a", "a", "b"]})
    test = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0], "label": ["a", "b", "b", "a"]})
    f = {g.code: g for g in exact.check_conflicts(Split(train, test, "label"))}
    assert f["L3.cross_contradict"].details["n_rows"] == 1
    assert f["L3.cross_contradict"].examples == [1]


# --- L5 -----------------------------------------------------------------------


def test_planted_shortcut_flagged():
    s, _ = split_of(shortcut=True, seed=8)
    f = shortcut.check_shortcuts(s)
    assert f.severity == "high"
    assert f.details["top"][0]["feature"] == "leaky"
    assert f.details["flagged"] == ["leaky"]


def test_no_shortcut_on_clean_split():
    s, _ = split_of(seed=9)
    f = shortcut.check_shortcuts(s)
    assert f.severity == "ok" and f.value < 0.9


def test_categorical_shortcut():
    train = pd.DataFrame({"proto": ["tcp", "udp"] * 50, "x": range(100), "label": ["benign", "attack"] * 50})
    test = pd.DataFrame({"proto": ["udp", "tcp", "icmp"], "x": [1, 2, 3], "label": ["attack", "benign", "benign"]})
    f = shortcut.check_shortcuts(Split(train, test, "label"))
    top = f.details["top"][0]
    assert top["feature"] == "proto" and top["balanced_acc"] == 1.0


@pytest.mark.parametrize("name", ["Src IP", "dst_port", "Flow ID", "Timestamp", "sha256", "md5"])
def test_identifier_names_flagged(name):
    df = pd.DataFrame({name: [1, 2], "bytes": [3, 4], "label": [0, 1]})
    f = shortcut.check_identifiers(Split(df, df.copy(), "label"))
    assert [c["feature"] for c in f.details["columns"]] == [name]


@pytest.mark.parametrize("name", ["bytes", "duration", "spot", "valid", "ttl"])
def test_benign_names_not_flagged(name):
    df = pd.DataFrame({name: [1, 2], "label": [0, 1]})
    assert shortcut.check_identifiers(Split(df, df.copy(), "label")).value == 0


# --- scan + CLI ---------------------------------------------------------------


def test_scan_combined():
    s, truth = split_of(cross_dup=0.1, conflict=0.02, shortcut=True, seed=10)
    r = scan(s)
    assert r.get("L1.cross").details["n_hit"] == truth["cross_dup_rows"]
    assert r.get("L3.conflict").details["n_groups"] == truth["conflict_pairs"]
    assert r.get("L5.shortcut").severity == "high"
    assert r.worst() == "high"
    json.loads(r.to_json())


def test_cli_end_to_end(tmp_path):
    train, test, truth = make_split(cross_dup=0.1, seed=11)
    train.to_csv(tmp_path / "tr.csv", index=False)
    test.to_csv(tmp_path / "te.csv", index=False)
    out = tmp_path / "r.json"
    code = main(["scan", "--train", str(tmp_path / "tr.csv"), "--test", str(tmp_path / "te.csv"),
                 "--label", "label", "--json", str(out)])
    assert code == 2
    rep = json.loads(out.read_text())
    cross = next(f for f in rep["findings"] if f["code"] == "L1.cross")
    # Survives a CSV round trip: floats printed and re-parsed must still match.
    assert cross["details"]["n_hit"] == truth["cross_dup_rows"]


def test_cli_clean_exit_zero(tmp_path):
    train, test, _ = make_split(seed=12)
    train.to_csv(tmp_path / "tr.csv", index=False)
    test.to_csv(tmp_path / "te.csv", index=False)
    assert main(["scan", "--train", str(tmp_path / "tr.csv"), "--test", str(tmp_path / "te.csv"),
                 "--label", "label"]) == 0


def test_cli_bad_label(tmp_path, capsys):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "x.csv", index=False)
    assert main(["scan", "--train", str(tmp_path / "x.csv"), "--test", str(tmp_path / "x.csv"),
                 "--label", "nope"]) == 3
