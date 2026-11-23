"""E16's reading of numeric text as numbers (experiments/e16_edgeiiot_format.py)."""
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from experiments.e16_edgeiiot_format import as_numbers, normalised  # noqa: E402
from leakcheck import Split  # noqa: E402


def test_zero_and_zero_point_zero_become_one_value():
    s = pd.Series(["0", "0.0", "MQTT", "1574358", "0x00000000"], dtype=object)
    out = as_numbers(s)
    assert out[0] == out[1] == repr(0.0) and out[2] == "MQTT" and out[3] == repr(1574358.0)
    assert out[4] == "0x00000000"                       # hex text is not a decimal number


def test_an_all_numeric_text_column_becomes_numeric():
    out = as_numbers(pd.Series(["0", "0.0", "2"], dtype=object))
    assert out.dtype.kind == "f" and list(out) == [0.0, 0.0, 2.0]


def test_train_and_test_are_read_the_same_way():
    tr = pd.DataFrame({"f": ["0", "MQTT"], "y": [0, 0]})
    te = pd.DataFrame({"f": ["0.0", "x"], "y": [1, 1]})
    s = normalised(Split(tr, te, "y"))
    assert s.train["f"].iloc[0] == s.test["f"].iloc[0]
