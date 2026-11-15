import json

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
)


def test_dumps_escapes_fcs_characters() -> None:
    obj = {"a/b": "$x/y$", "nested": ["/", "$"], "nan": float("nan")}
    text = dumps(obj)

    assert "/" not in text.replace("\\/", "")
    assert "$" not in text
    restored = json.loads(text)
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
    restored = decode_series(json.loads(dumps(encode_series(series))))
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

    restored = decode_frame(json.loads(dumps(encoded)))
    pd.testing.assert_frame_equal(restored, frame[["a", "b"]])
