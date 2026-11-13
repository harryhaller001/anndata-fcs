import io
import math
import warnings
from collections.abc import Mapping
from typing import Any

import flowio
import numpy as np
import pandas as pd
from anndata import AnnData
from anndata.utils import make_index_unique
from scipy import sparse

from anndata_fcs._exceptions import FCSConversionWarning
from anndata_fcs._keywords import build_metadata, escape_key, split_text, unescape_key
from anndata_fcs._payload import (
    UnsupportedValueError,
    decode_frame,
    dumps,
    encode_frame,
    from_jsonable,
    loads,
    to_jsonable,
)
from anndata_fcs._schema import (
    DEFAULT_INDEX_CHANNEL,
    KEYWORD_PREFIX,
    KW_INDEX_CHANNEL,
    KW_OBS,
    KW_SCHEMA,
    KW_UNS,
    KW_VAR,
    MAX_EXACT_FLOAT32_INT,
    MAX_TEXT_BYTES,
    SCHEMA_VERSION,
    STRUCTURAL_CHANNEL_SUFFIXES,
    TEXT_SIZE_WARNING_BYTES,
    UNS_KEY,
    VAR_PREFIX,
)
from anndata_fcs._types import FCSMetadata

_DEFAULT_PNR = 262144

# =====================================================================================================================
# FCS -> AnnData
# =====================================================================================================================


def fcs_to_anndata(fdata: flowio.FlowData, *, preprocess: bool = False) -> AnnData:
    """Convert a :class:`flowio.FlowData` instance to an AnnData object.

    All metadata of the FCS file is kept:

    * Channel names (``$PnN``) become ``var_names``.
    * Per-channel keywords become ``var`` columns named ``pn<suffix>`` (e.g. ``$PnS`` -> ``pns``, ``$PnV`` ->
      ``pnv``). Values are stored as raw strings, missing keywords as missing values.
    * Global keywords, the ANALYSIS segment and the FCS version are stored in ``uns["fcs"]``.
    * Keywords describing the binary layout (offsets, ``$DATATYPE``, ``$PnB``, ...) are stored in
      ``uns["fcs"]["source"]`` for reference. They are regenerated when writing.
    * If the file was written by :func:`anndata_to_fcs`, ``obs``, ``obs_names``, additional ``var`` columns and
      ``uns`` are restored and the synthetic event index channel is removed from ``X``.

    Args:
        fdata: FlowData instance to convert.
        preprocess: If ``True``, apply gain (``$PnG``), log (``$PnE``) and time (``$TIMESTEP``) scaling to the event
            data. The ``pne``/``png`` columns and the ``timestep`` keyword are updated, so the metadata keeps
            describing ``X`` correctly. If ``False`` (default), ``X`` holds the event data as stored in the file.

    Returns:
        AnnData: AnnData object with events as observations and channels as variables.

    Raises:
        ValueError: If the FlowData instance has no event data.
    """
    if fdata.events is None:
        raise ValueError("FlowData instance contains no event data. Was it read with 'only_text=True'?")

    n_channels = int(fdata.channel_count)
    split = split_text(fdata.text, n_channels)

    if preprocess:
        X = np.asarray(fdata.as_array(preprocess=True))
    else:
        X = np.array(fdata.events).reshape(-1, n_channels)

    # Channel metadata
    channel_names = [str(label) for label in fdata.pnn_labels]
    var = _channels_to_var(split.channels, n_channels)
    var.index = pd.Index(channel_names, dtype=object)

    source: dict[str, Any] = dict(split.structural)
    source["pnb"] = [split.channels[i].get("b", "") for i in range(1, n_channels + 1)]

    global_keywords = dict(split.global_)
    if preprocess:
        var[escape_key(VAR_PREFIX + "e")] = pd.Series(["0,0"] * n_channels, index=var.index, dtype="string")
        var[escape_key(VAR_PREFIX + "g")] = pd.Series(["1.0"] * n_channels, index=var.index, dtype="string")
        if fdata.time_index is not None and "timestep" in global_keywords:
            source["timestep"] = global_keywords.pop("timestep")

    uns_fcs: FCSMetadata = {
        "schema_version": SCHEMA_VERSION,
        "version": str(fdata.version),
        "preprocessed": preprocess,
        "text": {escape_key(key): value for key, value in global_keywords.items()},
        "analysis": {escape_key(key): value for key, value in dict(fdata.analysis or {}).items()},
        "source": source,
    }
    if fdata.name is not None:
        uns_fcs["filename"] = str(fdata.name)

    obs = pd.DataFrame(index=pd.Index([str(i) for i in range(X.shape[0])], dtype=object))
    uns: dict[str, Any] = {}

    if KW_SCHEMA in split.payload:
        X, var, obs, uns = _decode_payload(split.payload, X, var, obs, source)

    uns[UNS_KEY] = uns_fcs

    if not var.index.is_unique:
        warnings.warn(
            "Channel names ($PnN) are not unique. Duplicates are made unique.",
            FCSConversionWarning,
            stacklevel=2,
        )
        var.index = make_index_unique(var.index)

    return AnnData(X=X, obs=obs, var=var, uns=uns)


def _channels_to_var(channels: Mapping[int, Mapping[str, str]], n_channels: int) -> pd.DataFrame:
    """Create ``var`` columns from per-channel keywords (raw string values, missing as NA)."""
    suffixes: list[str] = []
    for channel in channels.values():
        for suffix in channel:
            if suffix not in STRUCTURAL_CHANNEL_SUFFIXES and suffix not in suffixes:
                suffixes.append(suffix)

    data = {
        escape_key(VAR_PREFIX + suffix): pd.array(
            [channels[i].get(suffix) for i in range(1, n_channels + 1)],
            dtype="string",
        )
        for suffix in suffixes
    }
    return pd.DataFrame(data, index=pd.RangeIndex(n_channels))


def _decode_payload(
    payload: Mapping[str, str],
    X: np.ndarray,
    var: pd.DataFrame,
    obs: pd.DataFrame,
    source: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Restore AnnData specific information written by :func:`anndata_to_fcs`."""
    schema_version = int(payload[KW_SCHEMA])
    if schema_version > SCHEMA_VERSION:
        warnings.warn(
            f"FCS file was written with a newer anndata-fcs schema (version {schema_version}). "
            "Some information may not be restored.",
            FCSConversionWarning,
            stacklevel=3,
        )

    # Remove synthetic event index channel
    event_index: np.ndarray | None = None
    index_channel = payload.get(KW_INDEX_CHANNEL)
    if index_channel is not None and index_channel in var.index:
        position = int(var.index.get_loc(index_channel))  # ty: ignore[invalid-argument-type]
        event_index = X[:, position].astype(np.int64)
        X = np.delete(X, position, axis=1)
        var = var.drop(index=index_channel)
        source["pnb"] = [value for i, value in enumerate(source["pnb"]) if i != position]

    # Restore obs. The event index maps events to rows of the original obs, which allows restoring obs for FCS files
    # containing a subset of events (e.g. exported gates).
    if KW_OBS in payload:
        full_obs = decode_frame(loads(payload[KW_OBS]))
        if event_index is not None and (event_index.min(initial=0) < 0 or event_index.max(initial=0) >= len(full_obs)):
            event_index = None

        if event_index is not None:
            obs = full_obs.iloc[event_index]
        elif len(full_obs) == X.shape[0]:
            obs = full_obs
        else:
            warnings.warn(
                "Number of events does not match the stored observations and no event index channel is available. "
                "Observation metadata is not restored.",
                FCSConversionWarning,
                stacklevel=3,
            )

    # Restore additional var columns
    if KW_VAR in payload:
        extra_var = decode_frame(loads(payload[KW_VAR])).reindex(var.index)
        var = pd.concat([var, extra_var], axis=1)

    uns: dict[str, Any] = from_jsonable(loads(payload[KW_UNS])) if KW_UNS in payload else {}

    return X, var, obs, uns


# =====================================================================================================================
# AnnData -> FCS
# =====================================================================================================================


def anndata_to_fcs(
    adata: AnnData,
    *,
    pns_key: str | None = None,
    index_channel: str | None = DEFAULT_INDEX_CHANNEL,
    store_obs: bool = True,
    strict: bool = False,
) -> flowio.FlowData:
    """Convert an AnnData object to a :class:`flowio.FlowData` instance.

    The FCS file holds the following information:

    * ``X`` as event data (32-bit float, the only data type supported by FlowIO) with ``var_names`` as ``$PnN``.
    * ``var`` columns named ``pn<suffix>`` as per-channel keywords (e.g. ``pns`` -> ``$PnS``).
    * ``uns["fcs"]["text"]`` as global keywords.
    * ``obs`` (including ``obs_names``), all other ``var`` columns and JSON serialisable ``uns`` entries as JSON encoded
      keywords in the ``ANNDATA_FCS_*`` namespace, which are restored by :func:`fcs_to_anndata`.

    Information that cannot be stored (``layers``, ``obsm``, ``obsp``, ``varm``, ``varp``, ``raw``, unsupported
    column dtypes, non serialisable ``uns`` entries, float precision beyond float32) triggers a
    :class:`FCSConversionWarning`, or a :class:`ValueError` if ``strict=True``.

    Note:
        To write the result to disk, use :func:`anndata_fcs.write_fcs`. :meth:`flowio.FlowData.write_fcs` only keeps
        a few keywords by default.

    Args:
        adata: AnnData object to convert. It is not modified.
        pns_key: Column of ``var`` used for ``$PnS`` (marker/antigen labels). Defaults to the ``pns`` column, if present.
        index_channel: Name of a synthetic channel holding the position of each event in ``adata``. It allows mapping
            events of exported gates back to observations. Use ``None`` to omit it.
        store_obs: If ``True``, store ``obs`` and ``obs_names`` in the FCS file.
        strict: If ``True``, raise an error instead of warning when information would be lost.

    Returns:
        flowio.FlowData: FlowData instance.

    Raises:
        KeyError: If ``pns_key`` is not a column of ``var``.
        TypeError: If ``X`` is not numeric.
        ValueError: If channel names are invalid, or information would be lost and ``strict=True``.
    """
    fcs_bytes = _anndata_to_fcs_bytes(
        adata,
        pns_key=pns_key,
        index_channel=index_channel,
        store_obs=store_obs,
        strict=strict,
    )
    return flowio.FlowData(io.BytesIO(fcs_bytes))


def _anndata_to_fcs_bytes(
    adata: AnnData,
    *,
    pns_key: str | None,
    index_channel: str | None,
    store_obs: bool,
    strict: bool,
) -> bytes:
    """Serialise an AnnData object to the bytes of an FCS file. See :func:`anndata_to_fcs`."""

    def lossy(message: str) -> None:
        if strict:
            raise ValueError(message)
        warnings.warn(message, FCSConversionWarning, stacklevel=4)

    _check_unsupported_slots(adata, lossy)
    obs = _as_dataframe(adata.obs, "obs")
    var = _as_dataframe(adata.var, "var")

    channel_names = [str(name) for name in adata.var_names]
    if not adata.var_names.is_unique:
        raise ValueError("var_names must be unique to be used as FCS channel names ($PnN).")
    if any(name == "" for name in channel_names):
        raise ValueError("var_names must not be empty strings.")

    if pns_key is not None and pns_key not in var.columns:
        raise KeyError(f"Column '{pns_key}' not found in var.")

    keyword_columns = [str(c) for c in var.columns if _is_keyword_column(c)]
    extra_columns = [c for c in var.columns if not _is_keyword_column(c)]

    # Per-channel keywords
    channels: list[dict[str, str]] = []
    for position in range(adata.n_vars):
        channel: dict[str, str] = {}
        for column in keyword_columns:
            value = var[column].iloc[position]
            if not _is_missing(value):
                channel[unescape_key(column)[len(VAR_PREFIX) :]] = str(value)
        if pns_key is not None:
            value = var[pns_key].iloc[position]
            channel.pop("s", None)
            if not _is_missing(value):
                channel["s"] = str(value)
        channels.append(channel)

    # Event data
    X = _dense_matrix(adata.X)
    X = _linearize_log_channels(X, channels)

    for position, channel in enumerate(channels):
        if "r" not in channel:
            channel["r"] = _default_range(X[:, position])

    if index_channel is not None:
        if index_channel in channel_names:
            raise ValueError(f"Index channel '{index_channel}' already exists in var_names.")
        if adata.n_obs > MAX_EXACT_FLOAT32_INT:
            raise ValueError(
                f"Event index cannot be stored exactly as float32 for more than {MAX_EXACT_FLOAT32_INT} events. "
                "Use 'index_channel=None'."
            )
        X = np.column_stack([X, np.arange(adata.n_obs, dtype=np.float64)])
        channel_names.append(index_channel)
        channels.append({"r": str(max(adata.n_obs, 1))})

    X32 = X.astype(np.float32)
    if not np.array_equal(X32.astype(np.float64), X, equal_nan=True):
        lossy("X cannot be represented exactly as 32-bit float, the only data type supported by FlowIO.")

    # Global keywords
    uns_fcs: Mapping[str, Any] = adata.uns.get(UNS_KEY, {})
    global_keywords = {
        unescape_key(str(key)): str(value)
        for key, value in dict(uns_fcs.get("text", {})).items()
        if not str(key).lower().startswith(KEYWORD_PREFIX)
    }
    if len(uns_fcs.get("analysis", {})) > 0:
        lossy("The ANALYSIS segment in uns['fcs']['analysis'] cannot be written by FlowIO and is dropped.")

    global_keywords.update(
        _encode_payload(
            adata,
            obs=obs,
            extra_var=var[extra_columns],
            index_channel=index_channel,
            store_obs=store_obs,
            lossy=lossy,
        )
    )

    metadata, pns_labels = build_metadata(channels, global_keywords)

    handle = io.BytesIO()
    flowio.create_fcs(
        file_handle=handle,
        event_data=X32.ravel(),
        channel_names=channel_names,
        opt_channel_names=pns_labels if any(pns_labels) else None,
        metadata_dict=metadata,
    )
    return handle.getvalue()


def _as_dataframe(frame: object, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame. Load lazy AnnData objects into memory first.")
    return frame


def _is_keyword_column(column: object) -> bool:
    return isinstance(column, str) and column.startswith(VAR_PREFIX) and len(column) > len(VAR_PREFIX)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))  # ty: ignore[no-matching-overload]
    except (TypeError, ValueError):
        return False


def _check_unsupported_slots(adata: AnnData, lossy: Any) -> None:
    dropped = [
        name
        for name, present in (
            # anndata>=0.13 exposes X as layers[None]
            ("layers", any(key is not None for key in adata.layers.keys())),
            ("obsm", len(adata.obsm) > 0),
            ("obsp", len(adata.obsp) > 0),
            ("varm", len(adata.varm) > 0),
            ("varp", len(adata.varp) > 0),
            ("raw", adata.raw is not None),
        )
        if present
    ]
    if len(dropped) > 0:
        lossy(f"{', '.join(dropped)} cannot be stored in FCS files and are dropped.")


def _dense_matrix(X: Any) -> np.ndarray:
    if X is None:
        raise TypeError("AnnData object has no X.")
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.dtype.kind not in "biuf":
        raise TypeError(f"X must be numeric, got dtype '{X.dtype}'.")
    return X.astype(np.float64)


def _linearize_log_channels(X: np.ndarray, channels: list[dict[str, str]]) -> np.ndarray:
    """Convert channels stored on log scale ($PnE != 0,0) to linear scale in place.

    FlowIO only writes float data, for which the FCS standard requires linear scale. The conversion matches
    :meth:`flowio.FlowData.as_array`, so preprocessing the written file gives the same result as preprocessing the
    source file.
    """
    for position, channel in enumerate(channels):
        if "e" not in channel:
            continue
        decades, log0 = (float(value) for value in channel["e"].split(","))
        if decades == 0:
            continue
        if log0 == 0:
            log0 = 1.0
        if "r" not in channel:
            raise ValueError(f"Channel {position + 1} is stored on log scale but has no range ($PnR).")

        X[:, position] = (10 ** (decades * X[:, position] / float(channel["r"]))) * log0
        channel["e"] = "0,0"
        channel["r"] = str(math.ceil(10**decades * log0))
    return X


def _default_range(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    maximum = math.ceil(float(finite.max())) if finite.size > 0 else 0
    return str(max(_DEFAULT_PNR, maximum))


def _encode_payload(
    adata: AnnData,
    *,
    obs: pd.DataFrame,
    extra_var: pd.DataFrame,
    index_channel: str | None,
    store_obs: bool,
    lossy: Any,
) -> dict[str, str]:
    """Encode AnnData specific information as keywords."""
    payload: dict[str, str] = {KW_SCHEMA: str(SCHEMA_VERSION)}

    if index_channel is not None:
        payload[KW_INDEX_CHANNEL] = index_channel

    if store_obs:
        obs_frame, skipped = encode_frame(obs)
        if len(skipped) > 0:
            lossy(f"obs columns with unsupported dtypes are dropped: {', '.join(skipped)}")
        default_names = [str(i) for i in range(adata.n_obs)]
        if len(obs_frame["columns"]) > 0 or obs_frame["index"] != default_names:
            payload[KW_OBS] = dumps(obs_frame)

    if len(extra_var.columns) > 0:
        var_frame, skipped = encode_frame(extra_var)
        if len(skipped) > 0:
            lossy(f"var columns with unsupported dtypes are dropped: {', '.join(skipped)}")
        if len(var_frame["columns"]) > 0:
            payload[KW_VAR] = dumps(var_frame)

    uns: dict[str, Any] = {}
    for key, value in adata.uns.items():
        if key == UNS_KEY:
            continue
        try:
            uns[str(key)] = to_jsonable(value)
        except UnsupportedValueError as error:
            lossy(f"uns['{key}'] cannot be stored in FCS files and is dropped: {error}")
    if len(uns) > 0:
        payload[KW_UNS] = dumps(uns)

    size = sum(len(value) for value in payload.values())
    if size > MAX_TEXT_BYTES:
        raise ValueError(
            f"Encoded AnnData metadata ({size} bytes) exceeds the maximum size of the FCS TEXT segment. "
            "Use 'store_obs=False' or remove metadata."
        )
    if size > TEXT_SIZE_WARNING_BYTES:
        warnings.warn(
            f"Encoded AnnData metadata is large ({size} bytes). Some FCS software may not handle large TEXT segments. "
            "Consider 'store_obs=False'.",
            FCSConversionWarning,
            stacklevel=4,
        )

    return payload
