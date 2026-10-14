import json
import string

import numpy as np
import pandas as pd
import pytest

from leakcheck import Split, scan
from leakcheck.checks import group, near, near_text, temporal
from leakcheck.clean import CleanOptions, clean_split, verify
from leakcheck.cli import main
from leakcheck.synth import make_split


def split_of(**kw):
    train, test, truth = make_split(**kw)
    return Split(train, test, "label"), truth


# --- L2 tabular -----------------------------------------------------------------


def test_no_near_duplicates_on_clean_split():
    s, _ = split_of(seed=20)
    f = near.check_near(s)
    assert f.value == 0 and f.severity == "ok"


@pytest.mark.parametrize("rate", [0.01, 0.1])
def test_near_recovered_exactly_and_kept_apart_from_exact(rate):
    s, truth = split_of(cross_dup=0.05, near_dup=rate, seed=21)
    f = near.check_near(s)
    assert f.details["n_near"] == truth["near_dup_rows"]
    assert f.details["exact_or_near_rate"] == pytest.approx(
        (truth["near_dup_rows"] + truth["cross_dup_rows"]) / truth["n_test"])


def test_near_examples_are_genuinely_close():
    s, _ = split_of(near_dup=0.05, seed=22)
    f = near.check_near(s)
    feats = s.features
    for ex in f.examples:
        a = s.test.loc[ex["test"], feats].to_numpy(float)
        b = s.train.loc[ex["train"], feats].to_numpy(float)
        assert ex["distance"] < 1e-3
        assert np.allclose(a, b, rtol=1e-4) and not np.array_equal(a, b)


def test_near_curve_is_monotone():
    s, _ = split_of(near_dup=0.05, near_noise=0.01, seed=23)
    curve = list(near.check_near(s).details["curve"].values())
    assert curve == sorted(curve)


def test_categorical_values_must_match():
    train = pd.DataFrame({"proto": ["tcp"], "bytes": [100.0], "label": [1]})
    test = pd.DataFrame({"proto": ["udp", "tcp"], "bytes": [100.0, 100.001], "label": [1, 1]})
    d, j = near.nearest(Split(train, test, "label"))
    assert d[0] == np.inf and j[0] == -1
    assert d[1] < 1e-3 and j[1] == 0


def test_missing_values_only_match_missing():
    train = pd.DataFrame({"a": [np.nan, 5.0], "b": [1.0, 1.0], "label": [0, 0]})
    test = pd.DataFrame({"a": [np.nan, 0.0], "b": [1.000001, 2.0], "label": [0, 0]})
    d, j = near.nearest(Split(train, test, "label"))
    assert j[0] == 0 and d[0] < 1e-3
    assert d[1] > 0.1


def test_sampled_estimate_interval_covers_truth():
    s, truth = split_of(n_test=5000, near_dup=0.1, seed=24)
    f = near.check_near(s, sample=1000)
    lo, hi = f.details["ci95"]
    assert f.details["sampled"] and lo <= truth["near_dup_rows"] / 5000 <= hi


def test_no_numeric_features_skips_l2():
    df = pd.DataFrame({"proto": ["tcp", "udp"], "label": [0, 1]})
    s = Split(df, df.copy(), "label")
    assert near.check_near(s) is None
    assert "L2.near" in scan(s).meta["skipped"]


# --- L2 text ----------------------------------------------------------------------


def test_shingles_and_jaccard():
    assert near_text.shingles("ab") == frozenset(["ab"])
    assert near_text.jaccard(near_text.shingles("hello world"), near_text.shingles("Hello   World")) == 1.0


def _random_strings(n, length, seed):
    rng = np.random.default_rng(seed)
    letters = np.array(list(string.ascii_lowercase + string.digits + "/.-"))
    return ["".join(rng.choice(letters, length)) for _ in range(n)]


def test_text_near_copies_all_found_unrelated_not():
    rng = np.random.default_rng(30)
    base = _random_strings(300, 80, 31)
    edited = []
    for s in base[:100]:
        i = rng.integers(len(s))
        edited.append(s[:i] + "#" + s[i + 1 :])  # one character changed: Jaccard >= 0.87
    unrelated = _random_strings(100, 80, 32)
    train = pd.DataFrame({"url": base, "label": 0})
    test = pd.DataFrame({"url": edited + unrelated + base[200:210], "label": 0})
    f = near_text.check_near_text(Split(train, test, "label", text="url"))
    assert f.details["n_near"] == 100
    assert f.details["identical_rate"] == pytest.approx(10 / 210)
    for ex in f.examples:
        assert ex["jaccard"] >= 0.8 and ex["train"] == ex["test"]


def test_text_skipped_without_text_column():
    s, _ = split_of(seed=33)
    assert near_text.check_near_text(s) is None


# --- L4 ---------------------------------------------------------------------------


def test_temporal_overlap_hand_computed():
    train = pd.DataFrame({"t": np.arange(100), "x": np.arange(100) % 7, "label": 0})
    test = pd.DataFrame({"t": np.arange(50, 150), "x": np.arange(100) % 7, "label": 0})
    f = {g.code: g for g in temporal.check_temporal(Split(train, test, "label", time="t"))}
    assert f["L4.after_test_start"].value == pytest.approx(49 / 100)
    assert f["L4.after_test_start"].details["after_test_end"] == 0


def test_temporal_clean_when_train_precedes_test():
    train = pd.DataFrame({"t": ["2017-07-03 09:00", "2017-07-04 10:00"], "x": [1, 2], "label": [0, 1]})
    test = pd.DataFrame({"t": ["2017-07-05 09:00", "2017-07-06 09:00"], "x": [1, 2], "label": [0, 1]})
    f = {g.code: g for g in temporal.check_temporal(Split(train, test, "label", time="t"))}
    assert f["L4.after_test_start"].value == 0 and f["L4.after_test_start"].severity == "ok"
    assert "2017-07-05" in f["L4.after_test_start"].details["test_range"][0]


def test_time_as_shortcut():
    # Benign collected early, attacks late: time alone separates them.
    t = np.arange(400)
    y = (t % 200 >= 100).astype(int)
    df = pd.DataFrame({"t": t, "x": np.random.default_rng(0).normal(size=400), "label": y})
    s = Split(df.iloc[::2], df.iloc[1::2], "label", time="t")
    f = {g.code: g for g in temporal.check_temporal(s)}
    assert f["L4.time_shortcut"].severity == "high"


def test_unparseable_time_reported():
    df = pd.DataFrame({"t": ["soon", "later"], "x": [1, 2], "label": [0, 1]})
    f = temporal.check_temporal(Split(df, df.copy(), "label", time="t"))
    assert f[0].severity == "info" and "could not be parsed" in f[0].summary


# --- L6 ---------------------------------------------------------------------------


def test_group_overlap_hand_computed():
    train = pd.DataFrame({"host": ["a", "b", "c"], "x": [1, 2, 3], "label": 0})
    test = pd.DataFrame({"host": ["a", "a", "d", "e"], "x": [4, 5, 6, 7], "label": 0})
    f = group.check_group(Split(train, test, "label", group="host"))
    assert f.value == 0.5 and f.details["shared_groups"] == 1 and f.details["test_groups"] == 3


# --- clean ------------------------------------------------------------------------

ALL_OPTS = CleanOptions(near_eps=0.01, dedup_within=True, conflicts="drop")


def test_clean_removes_exactly_what_was_planted():
    s, truth = split_of(cross_dup=0.1, near_dup=0.05, train_dup=0.05, conflict=0.03, seed=40)
    r = clean_split(s, ALL_OPTS)
    c = r.counts()
    assert c["conflict_drop"] == {"train": truth["conflict_rows"], "test": 0}
    assert c["dedup_within"] == {"train": truth["train_dup_rows"], "test": 0}
    assert c["cross_exact"]["test"] == truth["cross_dup_rows"]
    assert c["near_tabular"]["test"] == truth["near_dup_rows"]
    assert len(r.test) == truth["n_test"] - truth["cross_dup_rows"] - truth["near_dup_rows"]


def test_clean_output_passes_verification():
    s, _ = split_of(cross_dup=0.1, near_dup=0.05, train_dup=0.05, conflict=0.03, seed=41)
    r = clean_split(s, ALL_OPTS)
    checks = verify(Split(r.train, r.test, "label"), ALL_OPTS)
    assert checks and all(v["ok"] for v in checks.values())


def test_verify_is_not_vacuous():
    s, _ = split_of(cross_dup=0.1, near_dup=0.05, train_dup=0.05, conflict=0.03, seed=42)
    checks = verify(s, ALL_OPTS)
    for code in ("L1.cross", "L1.within_train", "L3.conflict", "L2.near"):
        assert not checks[code]["ok"], code
    assert checks["L1.within_test"]["ok"]  # nothing planted there


def test_removed_rows_logged_by_input_position():
    s, _ = split_of(cross_dup=0.1, seed=43)
    r = clean_split(s, CleanOptions())
    expected = np.flatnonzero(np.isin(s.test_hash, s.train_hash)).tolist()
    assert r.removed["cross_exact"]["test"] == expected


def test_conflict_majority_and_tie():
    train = pd.DataFrame({"x": [1.0, 1.0, 1.0, 2.0, 2.0, 3.0], "label": ["a", "a", "b", "a", "b", "c"]})
    test = pd.DataFrame({"x": [9.0], "label": ["a"]})
    r = clean_split(Split(train, test, "label"), CleanOptions(conflicts="majority"))
    assert r.removed["conflict_majority"]["train"] == [2, 3, 4]
    assert r.train["x"].tolist() == [1.0, 1.0, 3.0]


def test_conflict_spanning_splits_is_resolved():
    train = pd.DataFrame({"x": [1.0, 1.0], "label": ["a", "a"]})
    test = pd.DataFrame({"x": [1.0, 2.0], "label": ["b", "b"]})
    r = clean_split(Split(train, test, "label"), CleanOptions(conflicts="majority"))
    assert r.train["x"].tolist() == [1.0, 1.0] and r.test["x"].tolist() == [2.0]


def test_group_and_text_cleaning():
    base = _random_strings(50, 80, 50)
    train = pd.DataFrame({"url": base, "host": [f"h{i % 5}" for i in range(50)], "x": np.arange(50.0), "label": 0})
    test = pd.DataFrame({
        "url": [base[0][:-1] + "#", _random_strings(1, 80, 51)[0], _random_strings(1, 80, 52)[0]],
        "host": ["zz", "h1", "zz"],
        "x": [1000.0, 2000.0, 3000.0],
        "label": 0,
    })
    s = Split(train, test, "label", group="host", text="url")
    opts = CleanOptions(text_tau=0.8, drop_group_overlap=True)
    r = clean_split(s, opts)
    assert r.removed["group_overlap"]["test"] == [1]
    assert r.removed["near_text"]["test"] == [0]
    assert len(r.test) == 1
    assert all(v["ok"] for v in verify(Split(r.train, r.test, "label", group="host", text="url"), opts).values())


# --- CLI round trip ---------------------------------------------------------------


def test_cli_clean_then_rescan_from_files(tmp_path):
    train, test, truth = make_split(cross_dup=0.1, near_dup=0.05, train_dup=0.05, conflict=0.03, seed=60)
    train.to_csv(tmp_path / "tr.csv", index=False)
    test.to_csv(tmp_path / "te.csv", index=False)
    out = tmp_path / "clean"
    code = main(["clean", "--train", str(tmp_path / "tr.csv"), "--test", str(tmp_path / "te.csv"),
                 "--label", "label", "--out", str(out), "--near-eps", "0.01", "--dedup-within"])
    assert code == 0
    log = json.loads((out / "clean_log.json").read_text())
    assert log["verification"]["ok"]
    assert log["removed_counts"]["cross_exact"]["test"] == truth["cross_dup_rows"]
    assert len(log["input"]["train"]["sha256"]) == 64

    rescan = scan(Split.from_files(out / "train.csv", out / "test.csv", "label"))
    assert rescan.get("L1.cross").value == 0
    assert rescan.get("L1.within_train").value == 0
    assert rescan.get("L3.conflict").value == 0
    assert rescan.get("L2.near").value == 0


def test_cli_scan_with_all_options(tmp_path):
    df = pd.DataFrame({"t": np.arange(20), "host": ["a", "b"] * 10, "url": [f"http://x{i}.com" for i in range(20)],
                       "x": np.arange(20.0), "label": [0, 1] * 10})
    df.iloc[:10].to_csv(tmp_path / "tr.csv", index=False)
    df.iloc[5:].to_csv(tmp_path / "te.csv", index=False)
    out = tmp_path / "r.json"
    code = main(["scan", "--train", str(tmp_path / "tr.csv"), "--test", str(tmp_path / "te.csv"), "--label", "label",
                 "--time", "t", "--group", "host", "--text", "url", "--json", str(out)])
    assert code == 2
    codes = {f["code"] for f in json.loads(out.read_text())["findings"]}
    assert {"L2.near", "L2.near_text", "L4.after_test_start", "L4.time_shortcut", "L6.group"} <= codes
