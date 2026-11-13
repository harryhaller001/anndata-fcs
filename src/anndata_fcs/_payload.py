"""JSON encoding of AnnData specific information stored in FCS TEXT keywords.

The encoding is exact for numeric, boolean, string and categorical columns: floats are serialised with ``repr``
precision and dtypes are stored alongside the values.
"""

import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

_NDARRAY_TAG = "__anndata_fcs_ndarray__"


class UnsupportedValueError(TypeError):
    """Raised when a value cannot be encoded."""


def dumps(obj: Any) -> str:
    """Serialise to JSON that survives the FCS TEXT segment as parsed by FlowIO.

    FlowIO removes every ``$`` from the TEXT segment and cannot parse values starting or ending with the delimiter
    ``/``. Both characters only occur inside JSON strings and are replaced by JSON escape sequences.
    """
    text = json.dumps(obj, ensure_ascii=True, allow_nan=True, separators=(",", ":"))
    return text.replace("/", "\\/").replace("$", "\\u0024")


def loads(text: str) -> Any:
    """Inverse of :func:`dumps`."""
    return json.loads(text)


# ---------------------------------------------------------------------------------------------------------------------
# Data frames
# ---------------------------------------------------------------------------------------------------------------------


def _nullable_values(series: pd.Series) -> list[Any]:
    return [None if pd.isna(value) else _python_scalar(value) for value in series.astype(object)]


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


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
        values = _nullable_values(series)
        if not all(value is None or isinstance(value, str) for value in values):
            raise UnsupportedValueError("Column contains values which are neither strings nor missing.")
        kind = "object" if pd.api.types.is_object_dtype(dtype) else "extension"
        return {"kind": kind, "dtype": str(dtype), "values": values}

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


# ---------------------------------------------------------------------------------------------------------------------
# Unstructured data (uns)
# ---------------------------------------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert nested ``uns`` values to JSON compatible objects.

    Supported: ``None``, ``bool``, ``int``, ``float``, ``str``, NumPy scalars and arrays (numeric, boolean or string),
    and dictionaries with string keys and lists/tuples of supported values.

    Raises:
        UnsupportedValueError: If a value is not supported.
    """
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj

    if isinstance(obj, np.generic):
        if obj.dtype.kind not in "biufU":
            raise UnsupportedValueError(f"Unsupported NumPy scalar of dtype '{obj.dtype}'.")
        return obj.item()

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            if not all(isinstance(value, str) for value in obj.flat):
                raise UnsupportedValueError("Object arrays are only supported if they contain strings.")
        elif obj.dtype.kind not in "biufU":
            raise UnsupportedValueError(f"Unsupported NumPy array of dtype '{obj.dtype}'.")
        return {_NDARRAY_TAG: obj.tolist(), "dtype": obj.dtype.str, "shape": list(obj.shape)}

    if isinstance(obj, Mapping):
        if not all(isinstance(key, str) for key in obj):
            raise UnsupportedValueError("Only dictionaries with string keys are supported.")
        return {key: to_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, list | tuple):
        return [to_jsonable(value) for value in obj]

    raise UnsupportedValueError(f"Unsupported type '{type(obj).__name__}'.")


def from_jsonable(obj: Any) -> Any:
    """Inverse of :func:`to_jsonable`."""
    if isinstance(obj, dict):
        if _NDARRAY_TAG in obj:
            array = np.array(obj[_NDARRAY_TAG], dtype=np.dtype(obj["dtype"]))
            return array.reshape(obj["shape"])
        return {key: from_jsonable(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [from_jsonable(value) for value in obj]

    return obj
