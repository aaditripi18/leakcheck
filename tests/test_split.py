import numpy as np
import pandas as pd
import pytest

from leakcheck.split import Split, canonicalize, row_hashes


def h(df):
    return row_hashes(canonicalize(df))


def test_float_print_noise_hashes_equal():
    a = pd.DataFrame({"x": [0.1 + 0.2], "y": [1.0]})
    b = pd.DataFrame({"x": [0.3], "y": [1.0]})
    assert h(a)[0] == h(b)[0]


def test_real_differences_hash_differently():
    a = pd.DataFrame({"x": [0.3000001]})
    b = pd.DataFrame({"x": [0.3]})
    assert h(a)[0] != h(b)[0]


def test_negative_zero_and_nan_canonical():
    a = pd.DataFrame({"x": [-0.0, np.nan]})
    b = pd.DataFrame({"x": [0.0, float("nan")]})
    assert (h(a) == h(b)).all()


def test_string_whitespace_and_missing():
    a = pd.DataFrame({"s": [" tcp", None]})
    b = pd.DataFrame({"s": ["tcp ", np.nan]})
    assert (h(a) == h(b)).all()


def test_int_in_train_float_in_test_still_match():
    train = pd.DataFrame({"x": [1, 2, 3], "label": [0, 1, 0]})
    test = pd.DataFrame({"x": [2.0, np.nan], "label": [1, 0]})
    s = Split(train, test, "label")
    assert s.test_hash[0] == s.train_hash[1]


def test_column_order_does_not_matter():
    train = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})
    test = pd.DataFrame({"b": [4], "label": [1], "a": [2]})
    s = Split(train, test, "label")
    assert s.test_hash[0] == s.train_hash[1]


def test_label_time_group_ignore_excluded_from_features():
    df = pd.DataFrame({"a": [1], "t": [5], "g": ["h"], "id": [9], "label": [0]})
    s = Split(df, df.copy(), "label", time="t", group="g", ignore=["id"])
    assert s.features == ["a"]


def test_mismatched_columns_rejected():
    with pytest.raises(ValueError, match="different columns"):
        Split(pd.DataFrame({"a": [1], "label": [0]}), pd.DataFrame({"b": [1], "label": [0]}), "label")


def test_missing_label_rejected():
    with pytest.raises(ValueError, match="label column"):
        Split(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), "label")
