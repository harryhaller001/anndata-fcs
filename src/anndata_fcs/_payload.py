import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


class UnsupportedValueError(TypeError):
    """Raised when a value cannot be encoded."""


def dumps(obj: Any) -> str:
    """Serialise to JSON that survives the FCS TEXT segment as parsed by FlowIO.

    FlowIO removes every ``$`` from the TEXT segment and cannot parse values starting or ending with the delimiter
    ``/``. Both characters only occur inside JSON strings and are replaced by JSON escape sequences, so the result is
    decoded by :func:`json.loads`.
    """
    text = json.dumps(obj, ensure_ascii=True, allow_nan=True, separators=(",", ":"))
    return text.replace("/", "\\/").replace("$", "\\u0024")


def _nullable_values(series: pd.Series) -> list[Any]:
    """Values as Python scalars, with ``None`` for missing values."""
    # Series.tolist returns Python scalars. The mask is computed at once instead of calling pd.isna per value.
    missing = series.isna().to_numpy()
    values = series.tolist()
    if missing.any():
        for position in np.flatnonzero(missing):
            values[position] = None
    return values


def encode_series(series: pd.Series) -> dict[str, Any]:
    """Encode a column as JSON compatible dictionary.

    Raises:
        UnsupportedValueError: If the dtype is not supported.
    """
    dtype = series.dtype

    if isinstance(dtype, pd.CategoricalDtype):
        return {
            "kind": "categorical",
            "ordered": bool(dtype.ordered),
            "categories": encode_series(pd.Series(dtype.categories)),
            "codes": series.cat.codes.tolist(),
        }

    if pd.api.types.is_complex_dtype(dtype):
        raise UnsupportedValueError(f"Unsupported dtype '{dtype}'.")

    if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
        # NaN and inf are kept, since float values are written with allow_nan=True
        return {"kind": "numpy", "dtype": dtype.str, "values": series.to_numpy().tolist()}

    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        # Nullable extension dtypes (Int64, Float64, boolean)
        return {"kind": "extension", "dtype": str(dtype), "values": _nullable_values(series)}

    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        is_object = pd.api.types.is_object_dtype(dtype)
        # String dtypes only hold strings. Object columns are checked in C instead of per value in Python.
        if is_object and pd.api.types.infer_dtype(series, skipna=True) not in ("string", "empty"):
            raise UnsupportedValueError("Column contains values which are neither strings nor missing.")
        return {"kind": "object" if is_object else "extension", "dtype": str(dtype), "values": _nullable_values(series)}

    raise UnsupportedValueError(f"Unsupported dtype '{dtype}'.")


def decode_series(spec: Mapping[str, Any], index: pd.Index | None = None) -> pd.Series:
    """Inverse of :func:`encode_series`."""
    kind = spec["kind"]

    if kind == "categorical":
        categories = pd.Index(decode_series(spec["categories"]))
        codes = np.asarray(spec["codes"], dtype=np.int64)
        dtype = pd.CategoricalDtype(categories=categories, ordered=bool(spec["ordered"]))
        values = pd.Categorical.from_codes(codes, dtype=dtype)
        return pd.Series(values, index=index)

    if kind == "numpy":
        return pd.Series(np.asarray(spec["values"], dtype=np.dtype(spec["dtype"])), index=index)

    if kind == "extension":
        return pd.Series(pd.array(spec["values"], dtype=spec["dtype"]), index=index)

    if kind == "object":
        return pd.Series(spec["values"], index=index, dtype=object)

    raise ValueError(f"Unknown column encoding '{kind}'.")


def encode_frame(frame: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Encode a data frame including its index.

    Returns:
        tuple[dict[str, Any], list[str]]: Encoded frame and names of columns that could not be encoded.
    """
    columns: list[list[Any]] = []
    skipped: list[str] = []

    for name in frame.columns:
        try:
            columns.append([str(name), encode_series(frame[name])])
        except UnsupportedValueError:
            skipped.append(str(name))

    encoded = {"index": [str(value) for value in frame.index], "columns": columns}
    return encoded, skipped


def decode_frame(spec: Mapping[str, Any]) -> pd.DataFrame:
    """Inverse of :func:`encode_frame`."""
    index = pd.Index(spec["index"], dtype=object)
    data = {name: decode_series(column, index=index) for name, column in spec["columns"]}
    return pd.DataFrame(data, index=index)
