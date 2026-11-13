import numpy as np
import pandas as pd
import pytest

from anndata_fcs._payload import (
    UnsupportedValueError,
    decode_frame,
    decode_series,
    dumps,
    encode_frame,
    encode_series,
    from_jsonable,
    loads,
    to_jsonable,
)


def test_dumps_escapes_fcs_characters() -> None:
    obj = {"a/b": "$x/y$", "nested": ["/", "$"], "nan": float("nan")}
    text = dumps(obj)

    assert "/" not in text.replace("\\/", "")
    assert "$" not in text
    restored = loads(text)
    assert restored["a/b"] == "$x/y$"
    assert restored["nested"] == ["/", "$"]
    assert np.isnan(restored["nan"])


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([0.1, np.nan, np.inf, 1 / 3], dtype=np.float64),
        pd.Series([0.1, 0.2], dtype=np.float32),
        pd.Series([1, -2, 2**62], dtype=np.int64),
        pd.Series([1, 2], dtype=np.uint8),
        pd.Series([True, False]),
        pd.Series([1, None], dtype="Int64"),
        pd.Series([True, None], dtype="boolean"),
        pd.Series(["a", None, "c/$"], dtype="string"),
        pd.Series(["a", "b"], dtype=object),
        pd.Series(pd.Categorical(["x", None, "y", "x"])),
        pd.Series(pd.Categorical(["low", "high"], categories=["low", "high"], ordered=True)),
        pd.Series(pd.Categorical([1, 2, 1])),
    ],
    ids=lambda s: str(s.dtype),
)
def test_series_roundtrip(series: pd.Series) -> None:
    restored = decode_series(loads(dumps(encode_series(series))))
    pd.testing.assert_series_equal(restored, series)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(pd.date_range("2020-01-01", periods=2)),
        pd.Series([1 + 2j]),
        pd.Series(["a", 1], dtype=object),
    ],
)
def test_series_unsupported(series: pd.Series) -> None:
    with pytest.raises(UnsupportedValueError):
        encode_series(series)


def test_frame_roundtrip() -> None:
    frame = pd.DataFrame(
        {"a": [1.5, 2.5], "b": pd.Categorical(["x", "y"]), "date": pd.date_range("2020-01-01", periods=2)},
        index=pd.Index(["cell/1", "cell$2"], dtype=object),
    )
    encoded, skipped = encode_frame(frame)
    assert skipped == ["date"]

    restored = decode_frame(loads(dumps(encoded)))
    pd.testing.assert_frame_equal(restored, frame[["a", "b"]])


def test_jsonable_roundtrip() -> None:
    obj = {
        "int": 1,
        "float": 0.1,
        "str": "a/b",
        "none": None,
        "list": [1, "a", [True]],
        "scalar": np.float32(0.5),
        "array": np.arange(6, dtype=np.int16).reshape(2, 3),
        "empty": np.zeros((0, 3)),
        "strings": np.array(["a", "bc"]),
        "objects": np.array(["a", "b"], dtype=object),
        "nested": {"x": np.array([1.0, np.nan])},
    }
    restored = from_jsonable(loads(dumps(to_jsonable(obj))))

    assert restored["int"] == 1 and restored["float"] == 0.1 and restored["str"] == "a/b"
    assert restored["none"] is None
    assert restored["list"] == [1, "a", [True]]
    assert restored["scalar"] == 0.5
    for key in ("array", "empty", "strings", "objects"):
        assert restored[key].dtype == obj[key].dtype
        np.testing.assert_array_equal(restored[key], obj[key])
    np.testing.assert_array_equal(restored["nested"]["x"], obj["nested"]["x"])


@pytest.mark.parametrize("value", [pd.DataFrame({"a": [1]}), {1: "a"}, {"a"}, np.array([object()])])
def test_jsonable_unsupported(value: object) -> None:
    with pytest.raises(UnsupportedValueError):
        to_jsonable(value)
