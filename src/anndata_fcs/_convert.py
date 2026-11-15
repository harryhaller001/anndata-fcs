import json
import math
from array import array
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import IO, Any

import flowio
import numpy as np
import pandas as pd
from anndata import AnnData

from anndata_fcs._exceptions import warn
from anndata_fcs._keywords import (
    assign_channels,
    build_metadata,
    check_spillover,
    escape_key,
    format_timestamp,
    linear_range,
    normalize_keyword,
    parse_channel_keyword,
    parse_pne,
    remove_channel,
    split_text,
    unescape_key,
    unique_channel_names,
)
from anndata_fcs._payload import (
    UnsupportedValueError,
    decode_frame,
    dumps,
    encode_frame,
    from_jsonable,
    to_jsonable,
)
from anndata_fcs._schema import (
    FCS_DELIMITER,
    KEYWORD_PREFIX,
    KW_INDEX_CHANNEL,
    KW_OBS,
    KW_SCHEMA,
    KW_UNS,
    KW_VAR,
    MAX_EXACT_FLOAT32_INT,
    MAX_TEXT_BYTES,
    PREPROCESSED_LAYER,
    SCHEMA_VERSION,
    SPILLOVER_KEYWORDS,
    TEXT_SIZE_WARNING_BYTES,
    UNS_KEY,
)
from anndata_fcs._types import FCSMetadata

_DEFAULT_PNR = 262144

_CHUNK_ROWS = 65_536
"""Number of events processed at once, which bounds the memory of temporary arrays."""

# =====================================================================================================================
# FCS -> AnnData
# =====================================================================================================================


def fcs_to_anndata(fdata: flowio.FlowData, *, preprocess: bool, owns_events: bool) -> AnnData:
    """Convert a :class:`flowio.FlowData` instance to an AnnData object. See :func:`anndata_fcs.read_fcs`.

    Args:
        fdata: FlowData instance to convert.
        preprocess: See :func:`anndata_fcs.read_fcs`.
        owns_events: If ``True``, the event buffer of ``fdata`` is used without copying. Only allowed if ``fdata`` is
            not accessible to the caller, since the result shares memory with it.
    """
    if fdata.events is None:
        raise ValueError("FlowData instance contains no event data. Was it read with 'only_text=True'?")

    n_channels = int(fdata.channel_count)
    split = split_text(fdata.text, n_channels)
    events = _event_view(fdata.events, n_channels)

    channel_names = [str(label) for label in fdata.pnn_labels]
    if len(set(channel_names)) < len(channel_names):
        # Writing de-duplicates $PnN with the same function, so per-channel keywords stay with their channel
        warn("Channel names ($PnN) are not unique. Duplicates are made unique.")
        channel_names = unique_channel_names(channel_names)
    var = pd.DataFrame(index=pd.Index(channel_names, dtype=object))
    obs: pd.DataFrame | None = None
    uns: dict[str, Any] = {}
    text = split.text

    index_position: int | None = None
    if KW_SCHEMA in split.payload:
        index_position, var, obs, uns = _decode_payload(split.payload, events, var)
        if index_position is not None:
            text = remove_channel(text, index_position + 1, n_channels)

    # Every array is created at most once: removing the index channel copies the event data
    def without_index(matrix: np.ndarray, *, copy: bool) -> np.ndarray:
        if index_position is not None:
            return np.delete(matrix, index_position, axis=1)
        return matrix.copy() if copy else matrix

    X = without_index(events, copy=not owns_events)
    layers: dict[str, np.ndarray] = {}
    if preprocess:
        # as_array returns a new array
        layers[PREPROCESSED_LAYER] = without_index(np.asarray(fdata.as_array(preprocess=True)), copy=False)

    uns_fcs: FCSMetadata = {
        "schema_version": SCHEMA_VERSION,
        "version": str(fdata.version),
        "text": {escape_key(key): value for key, value in text.items()},
        "analysis": {escape_key(key): value for key, value in dict(fdata.analysis or {}).items()},
    }
    if fdata.name is not None:
        uns_fcs["filename"] = str(fdata.name)

    uns[UNS_KEY] = uns_fcs
    if obs is None:
        obs = pd.DataFrame(index=_string_index(range(events.shape[0])))

    return AnnData(X=X, obs=obs, var=var, uns=uns, layers=layers)


def _string_index(values: Iterable[int]) -> pd.Index:
    """``obs_names`` from event positions (faster than converting a NumPy array to strings)."""
    return pd.Index([str(value) for value in values], dtype=object)


def _event_view(events: Any, n_channels: int) -> np.ndarray:
    """Event matrix (events x channels) viewing the event buffer of a FlowData instance, without copying it."""
    if isinstance(events, array):
        matrix = np.frombuffer(events, dtype=np.dtype(events.typecode))
    else:
        matrix = np.asarray(events)
    return matrix.reshape(-1, n_channels)


def _decode_payload(
    payload: Mapping[str, str],
    events: np.ndarray,
    var: pd.DataFrame,
) -> tuple[int | None, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    """Restore AnnData specific information written by :func:`anndata_fcs.write_fcs`.

    Returns:
        Position of the synthetic event index channel (to be removed from the event data, ``None`` if absent), ``var``
        without the index channel, ``obs`` (``None`` for default ``obs``) and ``uns``.
    """
    obs: pd.DataFrame | None = None
    schema_version = int(payload[KW_SCHEMA])
    if schema_version > SCHEMA_VERSION:
        warn(
            f"FCS file was written with a newer anndata-fcs schema (version {schema_version}). "
            "Some information may not be restored."
        )

    # Remove synthetic event index channel
    event_index: np.ndarray | None = None
    position: int | None = None
    index_channel = payload.get(KW_INDEX_CHANNEL)
    if index_channel is not None and index_channel in var.index:
        position = int(var.index.get_loc(index_channel))  # ty: ignore[invalid-argument-type]
        event_index = _parse_event_index(events[:, position], index_channel)
        var = var.drop(index=index_channel)

    # Restore obs. The event index maps events to rows of the original obs, which allows restoring obs for FCS files
    # containing a subset of events (e.g. exported gates).
    if KW_OBS in payload:
        full_obs = decode_frame(json.loads(payload[KW_OBS]))
        if event_index is not None and event_index.max(initial=0) >= len(full_obs):
            warn(
                f"Event index channel '{index_channel}' refers to more observations than stored ({len(full_obs)}). "
                "It is not used to restore observation metadata."
            )
            event_index = None

        if event_index is not None:
            obs = full_obs.iloc[event_index]
        elif len(full_obs) == events.shape[0]:
            obs = full_obs
        else:
            warn(
                "Number of events does not match the stored observations and no valid event index channel is "
                "available. Observation metadata is not restored."
            )
    elif event_index is not None:
        # obs was not stored (default obs_names without columns, or store_obs=False). The event index keeps the
        # mapping of events to the written AnnData object, e.g. for FCS files containing a subset of events.
        obs = pd.DataFrame(index=_string_index(event_index.tolist()))

    # Restore var columns
    if KW_VAR in payload:
        extra_var = decode_frame(json.loads(payload[KW_VAR])).reindex(var.index)
        var = pd.concat([var, extra_var], axis=1)

    uns: dict[str, Any] = from_jsonable(json.loads(payload[KW_UNS])) if KW_UNS in payload else {}

    return position, var, obs, uns


def _parse_event_index(values: np.ndarray, index_channel: str) -> np.ndarray | None:
    """Positions of the events in the written AnnData object, ``None`` if the channel holds no valid positions.

    Software that modifies the event data (e.g. rescaling or compensating all channels) can break the index channel.
    Truncating such values would silently map events to wrong observations.
    """
    with np.errstate(invalid="ignore"):
        valid = bool(np.all(np.isfinite(values)) and np.all(values >= 0) and np.all(values == np.round(values)))
    if not valid:
        warn(
            f"Event index channel '{index_channel}' does not hold valid event positions (non-negative integers), e.g. "
            "because the event data was modified by other software. It is not used to restore observations."
        )
        return None
    return values.astype(np.int64)


# =====================================================================================================================
# AnnData -> FCS
# =====================================================================================================================


@dataclass
class _PreparedFCS:
    """Validated content of an FCS file, ready to be written by :func:`flowio.create_fcs`."""

    events: array
    """Event data (float32, events x channels, row-major)."""

    channel_names: list[str]
    pns_labels: list[str]
    metadata: dict[str, str]

    def write(self, handle: IO[bytes]) -> None:
        """Write the FCS file, starting at position 0 of the seekable ``handle``."""
        flowio.create_fcs(
            file_handle=handle,
            event_data=self.events,
            channel_names=self.channel_names,
            opt_channel_names=self.pns_labels if any(self.pns_labels) else None,
            metadata_dict=self.metadata,
        )


def prepare_fcs(
    adata: AnnData,
    *,
    pns_key: str | None,
    index_channel: str | None,
    store_obs: bool,
) -> _PreparedFCS:
    """Convert and validate an AnnData object for writing. See :func:`anndata_fcs.write_fcs`."""
    uns_fcs: Mapping[str, Any] = adata.uns.get(UNS_KEY, {})
    var = _as_dataframe(adata.var, "var")
    obs = _as_dataframe(adata.obs, "obs")
    _check_unsupported_slots(adata)

    channel_names = [str(name) for name in adata.var_names]
    _check_channel_names(channel_names, adata.var_names.is_unique, index_channel)

    if pns_key is not None and pns_key not in var.columns:
        raise KeyError(f"Column '{pns_key}' not found in var.")

    # Keywords, per-channel keywords are assigned to the written channels by $PnN
    keywords = {
        normalize_keyword(unescape_key(str(key))): str(value) for key, value in dict(uns_fcs.get("text", {})).items()
    }
    channels, global_keywords = assign_channels(keywords, channel_names)
    _check_renamed_channels(keywords, channel_names, channels)
    if pns_key is not None:
        for position, channel in enumerate(channels):
            value = var[pns_key].iloc[position]
            channel.pop("s", None)
            if not _is_missing(value):
                channel["s"] = str(value)

    # Event data. Keywords are adjusted first, since they define the conversion to linear scale.
    log_scales = _log_scales(channels, channel_names)

    if index_channel is not None:
        if adata.n_obs > MAX_EXACT_FLOAT32_INT:
            raise ValueError(
                f"Event index cannot be stored exactly as float32 for more than {MAX_EXACT_FLOAT32_INT} events. "
                "Use 'index_channel=None'."
            )

    range_columns = [position for position, channel in enumerate(channels) if "r" not in channel]
    n_columns = len(channel_names) + (1 if index_channel is not None else 0)
    buffer, X, exact, maxima = _encode_events(adata.X, log_scales, n_columns, range_columns)
    if not exact:
        warn("Event data cannot be represented exactly as 32-bit float, the only data type supported by FlowIO.")
    for position in range_columns:
        channels[position]["r"] = _default_range(maxima[position])

    if index_channel is not None:
        X[:, -1] = np.arange(adata.n_obs)
        channel_names.append(index_channel)
        channels.append({"r": str(max(adata.n_obs, 1))})

    # Global keywords
    global_keywords = {key: value for key, value in global_keywords.items() if not key.startswith(KEYWORD_PREFIX)}

    for keyword in SPILLOVER_KEYWORDS:
        if keyword not in global_keywords:
            continue
        spillover, messages = check_spillover(global_keywords[keyword], channel_names, keyword=keyword)
        for message in messages:
            warn(message)
        if spillover is None:
            del global_keywords[keyword]
        else:
            global_keywords[keyword] = spillover

    # The written file is a derived copy: event data is re-encoded as float32 (and possibly linearised or extended by
    # the index channel). FCS 3.1 uses $ORIGINALITY and $LAST_MODIFIED to mark such copies.
    global_keywords["originality"] = "DataModified"
    global_keywords["last_modified"] = format_timestamp(_now())
    if len(uns_fcs.get("analysis", {})) > 0:
        warn("The ANALYSIS segment in uns['fcs']['analysis'] cannot be written by FlowIO and is dropped.")

    global_keywords.update(
        _encode_payload(
            adata,
            obs=obs,
            var=var,
            index_channel=index_channel,
            store_obs=store_obs,
        )
    )

    metadata, pns_labels, messages = build_metadata(channels, global_keywords)
    for message in messages:
        warn(message)

    _check_text_size(metadata, channel_names, pns_labels)

    return _PreparedFCS(events=buffer, channel_names=channel_names, pns_labels=pns_labels, metadata=metadata)


def _now() -> datetime:
    """Time stamp for ``$LAST_MODIFIED`` (local time, like ``$DATE``/``$BTIM``)."""
    return datetime.now()


def _as_dataframe(frame: object, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame. Load lazy AnnData objects into memory first.")
    return frame


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))  # ty: ignore[no-matching-overload]
    except (TypeError, ValueError):
        return False


def _check_channel_names(channel_names: list[str], is_unique: bool, index_channel: str | None) -> None:
    """Validate the written channel names (``$PnN``)."""
    if not is_unique:
        raise ValueError("var_names must be unique to be used as FCS channel names ($PnN).")
    if index_channel is not None and index_channel in channel_names:
        raise ValueError(f"Index channel '{index_channel}' already exists in var_names.")

    names = channel_names + ([index_channel] if index_channel is not None else [])
    if any(name == "" for name in names):
        raise ValueError("var_names must not be empty strings.")
    # FlowIO cannot parse values starting or ending with the delimiter, which also corrupts the following keywords
    with_delimiter = [name for name in names if name.startswith(FCS_DELIMITER) or name.endswith(FCS_DELIMITER)]
    if len(with_delimiter) > 0:
        raise ValueError(
            f"Channel names ($PnN) must not start or end with '{FCS_DELIMITER}', which FlowIO cannot parse: "
            f"{', '.join(with_delimiter)}"
        )

    with_comma = [name for name in names if "," in name]
    if len(with_comma) > 0:
        warn(
            "FCS 3.1 does not allow commas in channel names ($PnN), since $SPILLOVER uses comma separated channel "
            f"names: {', '.join(with_comma)}"
        )
    with_dollar = [name for name in names if "$" in name]
    if len(with_dollar) > 0:
        warn(f"Channel names ($PnN) contain '$', which FlowIO removes when parsing FCS files: {', '.join(with_dollar)}")


def _check_unsupported_slots(adata: AnnData) -> None:
    if adata.raw is not None:
        warn(
            "raw cannot be stored in FCS files and is dropped, only X is written. "
            "Use 'write_fcs(adata.raw.to_adata(), ...)' to write raw."
        )

    dropped = [
        name
        for name, present in (
            # anndata>=0.13 exposes X as layers[None]. The preprocessed layer is derived from X and the keywords.
            ("layers", any(key not in (None, PREPROCESSED_LAYER) for key in adata.layers.keys())),
            ("obsm", len(adata.obsm) > 0),
            ("obsp", len(adata.obsp) > 0),
            ("varm", len(adata.varm) > 0),
            ("varp", len(adata.varp) > 0),
        )
        if present
    ]
    if len(dropped) > 0:
        warn(f"{', '.join(dropped)} cannot be stored in FCS files and are dropped.")


def _check_renamed_channels(
    keywords: Mapping[str, str],
    channel_names: list[str],
    channels: list[dict[str, str]],
) -> None:
    """Report written channels that lost their per-channel keywords, because they were renamed.

    Per-channel keywords are assigned by ``$PnN``. A channel whose name is not a ``$PnN`` value of ``keywords`` gets no
    keywords. This is only reported if channels of ``keywords`` are missing from the written file as well, so channels
    added to the AnnData object (which have no keywords to lose) do not trigger it. A renamed channel cannot be told
    apart from a removed and an added channel, so the latter is reported as well.
    """
    source_names = set()
    for key, value in keywords.items():
        channel = parse_channel_keyword(key)
        if channel is not None and channel[1] == "n":
            source_names.add(value)

    unmatched = [name for name, channel in zip(channel_names, channels, strict=True) if "n" not in channel]
    missing = sorted(source_names - set(channel_names))
    if len(unmatched) > 0 and len(missing) > 0:
        warn(
            f"var_names without a matching $PnN keyword in uns['fcs']['text']: {', '.join(unmatched)}. "
            f"If they are renamed channels ({', '.join(missing)} are not written), their per-channel keywords "
            "($PnS, $PnR, $PnE, $PnG, ...) are lost, and event data stored on log scale ($PnE) is written without "
            "conversion to linear scale. Rename the channels after writing, or add $PnN keywords for them."
        )


def _row_blocks(X: Any) -> Iterator[np.ndarray]:
    """Dense blocks of at most :data:`_CHUNK_ROWS` rows of a dense, sparse (CSR/CSC, as in AnnData) or backed matrix."""
    for start in range(0, X.shape[0], _CHUNK_ROWS):
        yield _to_array(X[start : start + _CHUNK_ROWS])


def _equal_values(a: np.ndarray, b: np.ndarray) -> bool:
    """Element-wise equality of numeric arrays, with NaN equal to NaN."""
    equal = a == b
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        equal |= np.isnan(a) & np.isnan(b)
    return bool(np.all(equal))


def _to_array(X: Any) -> np.ndarray:
    # Sparse matrices (scipy, as used by AnnData) are detected without importing scipy
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _channel_pne(channel: Mapping[str, str], name: str) -> tuple[float, float]:
    """Parse ``$PnE`` of a channel, with the channel name in error messages."""
    try:
        return parse_pne(channel["e"])
    except ValueError as error:
        raise ValueError(f"Channel '{name}' in uns['fcs']['text']: {error}") from None


def _channel_number(channel: Mapping[str, str], suffix: str, name: str) -> float:
    """Parse a numeric per-channel keyword (``$PnR``, ``$PnG``), with the channel name in error messages."""
    try:
        return float(channel[suffix])
    except ValueError:
        raise ValueError(
            f"Channel '{name}' in uns['fcs']['text']: Invalid $Pn{suffix.upper()} value {channel[suffix]!r}, "
            "expected a number."
        ) from None


def _log_scales(channels: list[dict[str, str]], channel_names: list[str]) -> dict[int, tuple[float, float, float]]:
    """Channels stored on log scale (``$PnE != 0,0``), which are converted to linear scale when writing.

    FlowIO only writes float data, for which the FCS standard requires linear scale. ``$PnE`` and ``$PnR`` are updated
    in place to describe the linear values.

    Returns:
        dict[int, tuple[float, float, float]]: Channel position mapped to decades, value of log(0) and ``$PnR`` of the
        log scale.
    """
    scales: dict[int, tuple[float, float, float]] = {}
    for position, (channel, name) in enumerate(zip(channels, channel_names, strict=True)):
        if "e" not in channel:
            continue
        decades, log0 = _channel_pne(channel, name)
        if decades == 0:
            continue
        if "r" not in channel:
            raise ValueError(f"Channel '{name}' is stored on log scale but has no range ($PnR).")

        scales[position] = (decades, log0, _channel_number(channel, "r", name))
        channel["e"] = "0,0"
        channel["r"] = linear_range(decades, log0)
    return scales


def _exact_in_float32(dtype: np.dtype) -> bool:
    """Whether every value of ``dtype`` is exactly representable as float32."""
    return dtype in (np.float32, np.float16) or (dtype.kind in "biu" and dtype.itemsize <= 2)


def _encode_events(
    X: Any,
    log_scales: Mapping[int, tuple[float, float, float]],
    n_columns: int,
    range_columns: list[int],
) -> tuple[array, np.ndarray, bool, dict[int, float]]:
    """Convert event data to the float32 matrix written to the FCS file.

    The matrix is allocated once, in the buffer passed to :func:`flowio.create_fcs` (which would otherwise copy it
    value by value). Events are processed in blocks, so dense, sparse or non-float32 event data is not copied as a
    whole. Channels stored on log scale are converted to linear scale like :meth:`flowio.FlowData.as_array` does.

    Args:
        X: Event data (events x channels).
        log_scales: Channels to convert from log to linear scale, see :func:`_log_scales`.
        n_columns: Number of written channels. Columns beyond the channels of ``X`` are filled by the caller.
        range_columns: Channels for which the maximum of the finite values is computed.

    Returns:
        tuple[array, np.ndarray, bool, dict[int, float]]: Buffer for :func:`flowio.create_fcs`, matrix view of the
        buffer, whether all values are represented exactly and the maxima of ``range_columns`` (``-inf`` if none).

    Raises:
        TypeError: If there is no event data or it is not numeric.
    """
    if X is None:
        raise TypeError("AnnData object has no X.")
    if not (hasattr(X, "dtype") and hasattr(X, "shape")):
        X = np.asarray(X)
    if X.dtype.kind not in "biuf":
        raise TypeError(f"X must be numeric, got dtype '{X.dtype}'.")

    n_rows, n_vars = X.shape
    buffer = array("f", [0.0]) * (n_rows * n_columns)
    out = np.frombuffer(buffer, dtype=np.float32).reshape(n_rows, n_columns)
    check = not _exact_in_float32(X.dtype)
    exact = True
    maxima = np.full(len(range_columns), -np.inf)

    start = 0
    for block in _row_blocks(X):
        target = out[start : start + block.shape[0], :n_vars]
        start += block.shape[0]
        target[...] = block
        if check and exact:
            exact = _equal_values(target, block)
        for position, (decades, log0, value_range) in log_scales.items():
            linear = (10 ** (decades * block[:, position].astype(np.float64) / value_range)) * log0
            target[:, position] = linear
            exact = exact and _equal_values(target[:, position], linear)
        if len(range_columns) > 0 and block.shape[0] > 0:
            values = block[:, range_columns]
            if values.dtype.kind == "f":
                values = np.where(np.isfinite(values), values, -np.inf)
            maxima = np.maximum(maxima, values.max(axis=0))

    return buffer, out, exact, dict(zip(range_columns, maxima.tolist(), strict=True))


def _default_range(maximum: float) -> str:
    """``$PnR`` of a channel without range: :data:`_DEFAULT_PNR`, or the maximum value if it is larger."""
    return str(max(_DEFAULT_PNR, math.ceil(maximum))) if math.isfinite(maximum) else str(_DEFAULT_PNR)


def _check_text_size(metadata: Mapping[str, str], channel_names: list[str], pns_labels: list[str]) -> None:
    """Check the size of the TEXT segment written by :func:`flowio.create_fcs`.

    The FCS header stores the end of the TEXT segment in 8 ASCII digits. FlowIO does not check this and writes a
    corrupt header for larger TEXT segments.

    Raises:
        ValueError: If the TEXT segment is too large.
    """
    # Upper bound: '$', delimiters and doubled delimiters in values. Required keywords and $PnB/$PnE/$PnG/$PnR, which
    # are generated by flowio.create_fcs, are bounded by constants.
    pairs = [*metadata.items(), *(("pnn", name) for name in channel_names), *(("pns", label) for label in pns_labels)]
    size = 512 + 64 * len(channel_names)
    for key, value in pairs:
        encoded = value.encode()
        size += len(key) + len(encoded) + encoded.count(FCS_DELIMITER.encode()) + 3

    if size > MAX_TEXT_BYTES:
        raise ValueError(
            f"The FCS TEXT segment (about {size} bytes) exceeds the maximum size of {MAX_TEXT_BYTES} bytes. "
            "Use 'store_obs=False' or remove metadata."
        )
    if size > TEXT_SIZE_WARNING_BYTES:
        warn(
            f"The FCS TEXT segment is large (about {size} bytes). Some FCS software may not handle large TEXT "
            "segments. Consider 'store_obs=False'."
        )


def _encode_payload(
    adata: AnnData,
    *,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    index_channel: str | None,
    store_obs: bool,
) -> dict[str, str]:
    """Encode AnnData specific information as keywords."""
    payload: dict[str, str] = {KW_SCHEMA: str(SCHEMA_VERSION)}

    if index_channel is not None:
        payload[KW_INDEX_CHANNEL] = index_channel

    if store_obs:
        obs_frame, skipped = encode_frame(obs)
        if len(skipped) > 0:
            warn(f"obs columns with unsupported dtypes are dropped: {', '.join(skipped)}")
        default_names = [str(i) for i in range(adata.n_obs)]
        if len(obs_frame["columns"]) > 0 or obs_frame["index"] != default_names:
            payload[KW_OBS] = dumps(obs_frame)

    if len(var.columns) > 0:
        var_frame, skipped = encode_frame(var)
        if len(skipped) > 0:
            warn(f"var columns with unsupported dtypes are dropped: {', '.join(skipped)}")
        if len(var_frame["columns"]) > 0:
            payload[KW_VAR] = dumps(var_frame)

    uns: dict[str, Any] = {}
    for key, value in adata.uns.items():
        if key == UNS_KEY:
            continue
        try:
            uns[str(key)] = to_jsonable(value)
        except UnsupportedValueError as error:
            warn(f"uns['{key}'] cannot be stored in FCS files and is dropped: {error}")
    if len(uns) > 0:
        payload[KW_UNS] = dumps(uns)

    return payload
