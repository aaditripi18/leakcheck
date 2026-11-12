"""`leakcheck explain` and H19's statistic S on planted cases whose answer is known."""
import json

import numpy as np
import pandas as pd

from leakcheck import Split
from leakcheck import explain as ex
from leakcheck.cli import main


D = 10


def _frame(x, y):
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["y"] = y
    return df


def _clusters(rng, y):
    return rng.normal(0, 1.0, (len(y), D)) + np.where(y[:, None] == 1, 1.0, -1.0)


def _case(kind: str, seed: int = 0) -> Split:
    """Two classes in ten dimensions. The test set's first 300 rows copy training rows.
    easy:      the copied rows are ordinary rows of their class (their neighbours agree),
               the other test rows lie near the class boundary (harder).
    memorised: the copied rows carry the other class's label (only the copy knows it),
               the other test rows are ordinary rows (easy)."""
    rng = np.random.default_rng(seed)
    n, k = 3000, 300
    y = rng.integers(0, 2, n)
    x = _clusters(rng, y)
    pick = rng.choice(n, k, replace=False)
    if kind == "memorised":
        y[pick] = 1 - y[pick]                       # isolated label noise, copied into test
    yo = rng.integers(0, 2, k)
    xo = _clusters(rng, yo) if kind == "memorised" else rng.normal(0, 0.4, (k, D))
    test = _frame(np.vstack([x[pick], xo]), np.concatenate([y[pick], yo]))
    return Split(_frame(x, y), test, "y")


def test_easy_and_memorised_leaks_get_opposite_signs():
    for kind, sign in (("easy", 1), ("memorised", -1)):
        split = _case(kind)
        leaked = ex.leak_masks(split)["near"]
        assert leaked[:300].all() and not leaked[300:].any()
        keep, left = ex.copies_in_train(split, leaked)
        assert left == 0                            # every copy removed
        st = ex.statistic(split, leaked, keep)
        assert np.sign(st["S"]) == sign, (kind, st)
        assert abs(st["S"]) > 0.15


def test_votes_fall_back_to_the_majority_class():
    # a test row whose category never occurs in training has no neighbour in its bucket
    tr = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 2, "v": np.arange(7.0), "y": [1, 1, 1, 0, 0, 0, 0]})
    te = pd.DataFrame({"c": ["z", "b"], "v": [0.0, 5.0], "y": [0, 0]})
    votes = ex.neighbour_votes(Split(tr, te, "y"), np.arange(2))
    assert votes[0] == 0                            # majority class of training (4 of 7)
    assert votes[1] == 0                            # nearest row in bucket "b" (v=5)


def test_explain_cli(tmp_path, monkeypatch):
    split = _case("memorised")
    split.train.to_csv(tmp_path / "train.csv", index=False)
    split.test.to_csv(tmp_path / "test.csv", index=False)
    monkeypatch.setattr(ex, "S_THRESHOLD", 0.0)
    out = tmp_path / "r.json"
    code = main(["explain", "--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
                 "--label", "y", "--json", str(out)])
    r = json.loads(out.read_text())
    assert code == 0 and r["leaked"] == 300 and r["verdict"] == "leaked rows look memorised"


def test_no_leak():
    rng = np.random.default_rng(1)
    tr = _frame(rng.normal(size=(200, 2)), rng.integers(0, 2, 200))
    te = _frame(rng.normal(size=(50, 2)) + 50, rng.integers(0, 2, 50))
    assert ex.explain(Split(tr, te, "y"))["verdict"] == "no leaked rows"


def test_every_training_row_a_copy_leaves_s_undefined():
    tr = _frame(np.zeros((20, 2)), np.array([0, 1] * 10))
    te = _frame(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]))
    r = ex.explain(Split(tr, te, "y"))
    assert r["leaked"] == 5 and r["S"] is None
    assert r["verdict"] == "no verdict (every training row is a copy of a leaked test row)"


def test_explain_cli_when_every_training_row_is_a_copy(tmp_path, capsys):
    tr = _frame(np.zeros((20, 2)), np.array([0, 1] * 10))
    te = _frame(np.zeros((5, 2)), np.array([0, 1, 0, 1, 0]))
    tr.to_csv(tmp_path / "train.csv", index=False)
    te.to_csv(tmp_path / "test.csv", index=False)
    code = main(["explain", "--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv"),
                 "--label", "y"])
    out = capsys.readouterr().out
    assert code == 0 and "S undefined" in out and "every training row is a copy" in out


def test_text_neighbours_do_not_depend_on_the_other_test_rows():
    from leakcheck.checks import near_text
    rng = np.random.default_rng(3)
    words = ["login", "secure", "bank", "verify", "account", "update", "paypal", "mail", "home", "page"]
    train = pd.Series([" ".join(rng.choice(words, 6)) for _ in range(400)])
    test = pd.Series([" ".join(rng.choice(words, 6)) for _ in range(200)])
    sim_all, nn_all = near_text.nearest_text(train, test, tau=0.8)
    keep = np.arange(0, 200, 3)
    sim_sub, nn_sub = near_text.nearest_text(train, test.iloc[keep], tau=0.8)
    assert np.array_equal(sim_all[keep], sim_sub) and np.array_equal(nn_all[keep], nn_sub)
