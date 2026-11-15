"""Classification of FCS TEXT keywords and construction of FlowIO metadata dictionaries.

All functions in this module operate on plain dictionaries and do not depend on AnnData objects.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from anndata.utils import make_index_unique

from anndata_fcs._exceptions import warn
from anndata_fcs._schema import (
    CHANNEL_COLUMN_PREFIX,
    CHANNEL_KEYWORD_PATTERN,
    FCS_DELIMITER,
    KEYWORD_PREFIX,
    STRUCTURAL_CHANNEL_SUFFIXES,
    STRUCTURAL_KEYWORDS,
)

# Escaping for names stored in h5ad files, where '/' is not allowed in keys.
_ESCAPES: tuple[tuple[str, str], ...] = (("%", "%25"), ("/", "%2F"))


def escape_key(key: str) -> str:
    """Escape a keyword so it can be used as a key in h5ad groups (no ``/`` allowed)."""
    for raw, escaped in _ESCAPES:
        key = key.replace(raw, escaped)
    return key


def unescape_key(key: str) -> str:
    """Inverse of :func:`escape_key`."""
    for raw, escaped in reversed(_ESCAPES):
        key = key.replace(escaped, raw)
    return key


@dataclass
class SplitText:
    """TEXT segment split into its components."""

    structural: dict[str, str] = field(default_factory=dict)
    """Keywords describing the binary layout (offsets, byte order, data type, ``$PnB``, ...)."""

    channels: list[dict[str, str]] = field(default_factory=list)
    """Per-channel keywords as suffix/value mappings in channel order, e.g. ``{"n": "FSC-A", "s": "CD3"}``."""

    text: dict[str, str] = field(default_factory=dict)
    """All remaining (global) keywords."""

    payload: dict[str, str] = field(default_factory=dict)
    """Keywords in the anndata-fcs namespace."""


def parse_channel_keyword(key: str, n_channels: int | None = None) -> tuple[int, str] | None:
    """Parse a per-channel keyword such as ``p12s`` into channel number and suffix (``(12, "s")``).

    Args:
        key: Keyword in the FlowIO convention (lower case, without ``$``).
        n_channels: If given, only keywords of channels ``1 <= n <= n_channels`` are per-channel keywords.

    Returns:
        tuple[int, str] | None: Channel number (1-based) and suffix, or ``None`` if ``key`` is no per-channel keyword.
    """
    match = CHANNEL_KEYWORD_PATTERN.match(key)
    if match is None:
        return None
    number = int(match.group(1))
    if number < 1 or (n_channels is not None and number > n_channels):
        return None
    return number, match.group(2)


def split_text(text: Mapping[str, str], n_channels: int) -> SplitText:
    """Split parsed TEXT keywords into structural, per-channel, payload and global keywords.

    Args:
        text: Keyword/value mapping as parsed by FlowIO (lower case keys without ``$``).
        n_channels: Number of channels (``$PAR``). ``p<n>...`` keywords are only per-channel keywords when
            ``1 <= n <= n_channels``, others are global keywords.

    Returns:
        SplitText: Classified keywords.
    """
    result = SplitText(channels=[{} for _ in range(n_channels)])

    for key, value in text.items():
        channel = parse_channel_keyword(key, n_channels)
        if key in STRUCTURAL_KEYWORDS or (channel is not None and channel[1] in STRUCTURAL_CHANNEL_SUFFIXES):
            result.structural[key] = value.strip()
        elif channel is not None:
            result.channels[channel[0] - 1][channel[1]] = value
        elif key.startswith(KEYWORD_PREFIX):
            result.payload[key] = value
        else:
            result.text[key] = value

    return result


def channel_column(suffix: str) -> str:
    """Name of the ``var`` column holding a per-channel keyword, e.g. ``$PnS`` for suffix ``s``.

    ``/`` and ``%`` are escaped, since h5ad files do not allow ``/`` in column names.
    """
    return CHANNEL_COLUMN_PREFIX + escape_key(suffix).upper()


def column_suffix(column: str) -> str | None:
    """Inverse of :func:`channel_column`: the keyword suffix, or ``None`` if ``column`` is no keyword column."""
    suffix = column.removeprefix(CHANNEL_COLUMN_PREFIX)
    # Suffixes start with a non-digit, as in the FlowIO keyword pattern (p<n><suffix>)
    if suffix == column or suffix == "" or suffix[0].isdigit():
        return None
    return unescape_key(suffix).lower()


def unique_channel_names(names: Sequence[str]) -> list[str]:
    """Make channel names (``$PnN``) unique by appending ``-1``, ``-2``, ... to duplicates."""
    index = pd.Index([str(name) for name in names], dtype=object)
    if index.is_unique:
        return list(index)
    with warnings.catch_warnings():
        # anndata warns about suffix collisions, which make_index_unique resolves
        warnings.simplefilter("ignore", UserWarning)
        return [str(name) for name in make_index_unique(index)]


def normalize_keyword(key: str) -> str:
    """Normalise a keyword to the FlowIO convention (lower case, without leading ``$``)."""
    return key.lower().lstrip("$")


def _check_keyword(key: str, messages: list[str]) -> bool:
    """Check whether a keyword name can be written. Appends a message to ``messages`` if not.

    Keyword names must not be empty and may only contain printable ASCII characters (only values are UTF-8). FlowIO
    cannot parse keywords starting or ending with the delimiter, which also corrupts the neighbouring keywords.
    """
    if key == "" or not all(32 <= ord(character) <= 126 for character in key):
        messages.append(
            f"Keyword {key!r} is not valid in FCS 3.1, which only allows non-empty keywords of printable ASCII "
            "characters. It is not written."
        )
        return False
    if key.startswith(FCS_DELIMITER) or key.endswith(FCS_DELIMITER):
        messages.append(
            f"Keyword {key!r} starts or ends with the delimiter '{FCS_DELIMITER}', which FlowIO cannot parse. It is "
            "not written."
        )
        return False
    return True


def _check_value(key: str, value: str, messages: list[str]) -> bool:
    """Check whether a keyword value can be written and read back by FlowIO. Appends messages to ``messages``.

    Returns:
        bool: Whether the keyword is written. Values containing ``$`` are written, although FlowIO removes ``$``.
    """
    if value == "":
        messages.append(
            f"Keyword '{key}' has an empty value, which is not allowed by the FCS standard. It is not written."
        )
        return False

    if value.startswith(FCS_DELIMITER) or value.endswith(FCS_DELIMITER):
        # Written, it would also corrupt the following keywords when parsed by FlowIO
        messages.append(
            f"Value of keyword '{key}' starts or ends with the delimiter '{FCS_DELIMITER}', which FlowIO cannot parse. "
            "It is not written."
        )
        return False

    if "$" in value:
        messages.append(f"Value of keyword '{key}' contains '$', which FlowIO removes when parsing FCS files.")

    return True


def build_metadata(
    channels: Sequence[Mapping[str, str]],
    global_keywords: Mapping[str, str],
) -> tuple[dict[str, str], list[str], list[str]]:
    """Build the arguments for :func:`flowio.create_fcs` from channel and global keywords.

    ``$PnN``, ``$PnB`` and ``$PnE`` are ignored, since they are set by :func:`flowio.create_fcs`. ``$PnS`` is
    returned separately, because :func:`flowio.create_fcs` silently drops it from the metadata dictionary.

    Args:
        channels: Per-channel keyword suffix/value mappings, in channel order.
        global_keywords: Global keywords. Structural keywords are ignored.

    Returns:
        tuple[dict[str, str], list[str], list[str]]: Metadata dictionary, list of ``$PnS`` values (empty string if
        absent) and descriptions of the information that is lost (keywords that are not written or cannot be read
        back exactly).
    """
    metadata: dict[str, str] = {}
    pns_labels: list[str] = []
    messages: list[str] = []

    for channel_index, channel in enumerate(channels):
        channel_number = channel_index + 1
        pns_labels.append("")

        for suffix, value in channel.items():
            if suffix in ("n", "b", "e"):
                continue

            key = f"p{channel_number}{suffix}"
            if not _check_keyword(key, messages) or not _check_value(key, value, messages):
                continue

            if suffix == "s":
                pns_labels[channel_index] = value
            else:
                metadata[_escape_delimiter(key)] = value

    for key, value in global_keywords.items():
        key = normalize_keyword(key)
        if key in STRUCTURAL_KEYWORDS:
            continue

        channel = parse_channel_keyword(key, len(channels))
        if channel is not None:
            # Written as is, the keyword would be attributed to a channel of the written file
            messages.append(
                f"Keyword '{key}' in uns['fcs']['text'] refers to channel {channel[0]} of the written file. Per-channel "
                f"keywords are stored in var columns (e.g. '{channel_column(channel[1])}'). It is not written."
            )
            continue

        if not _check_keyword(key, messages) or not _check_value(key, value, messages):
            continue
        metadata[_escape_delimiter(key)] = value

    return metadata, pns_labels, messages


def _escape_delimiter(key: str) -> str:
    """Double the delimiter in keywords. FlowIO only does this for values."""
    return key.replace(FCS_DELIMITER, FCS_DELIMITER * 2)


# ---------------------------------------------------------------------------------------------------------------------
# Keyword values defined by the FCS 3.1 standard
# ---------------------------------------------------------------------------------------------------------------------

_MONTHS: tuple[str, ...] = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def format_timestamp(moment: datetime) -> str:
    """Format a time stamp as required by ``$LAST_MODIFIED`` (``dd-mmm-yyyy hh:mm:ss.cc``).

    The month abbreviation is independent of the locale.
    """
    return (
        f"{moment.day:02d}-{_MONTHS[moment.month - 1]}-{moment.year:04d} "
        f"{moment.hour:02d}:{moment.minute:02d}:{moment.second:02d}.{moment.microsecond // 10_000:02d}"
    )


def parse_pne(value: str) -> tuple[float, float]:
    """Parse ``$PnE`` (``f1,f2``: decades and value of log(0)).

    As required by FCS 3.1, the invalid value ``f2 = 0`` of a log scaled channel (e.g. ``4,0``) is interpreted as
    ``f2 = 1``. This matches :meth:`flowio.FlowData.as_array`.

    Raises:
        ValueError: If the value is not of the form ``f1,f2``.
    """
    parts = value.split(",")
    try:
        if len(parts) != 2:
            raise ValueError
        decades, log0 = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"Invalid $PnE value {value!r}, expected 'f1,f2' (e.g. '0,0' or '4,1').") from None
    if decades != 0 and log0 == 0:
        log0 = 1.0
    return decades, log0


def linear_range(decades: float, log0: float) -> str:
    """``$PnR`` of a log scaled channel (``$PnE/decades,log0/``) after conversion to linear scale values."""
    return str(math.ceil(10**decades * log0))


def check_spillover(
    value: str,
    channel_names: Sequence[str],
    keyword: str = "spillover",
) -> tuple[str | None, list[str]]:
    """Check ``$SPILLOVER`` against the channels of the file that is written.

    ``$SPILLOVER`` is ``n,name_1,...,name_n,s_11,s_12,...,s_nn``, where the names are ``$PnN`` values and the spillover
    coefficients are given row-wise. Channels which do not exist in the written file (e.g. after subsetting ``var``)
    are removed from the matrix, since FCS 3.1 requires all names to refer to parameters.

    Args:
        value: Value of the spillover keyword.
        channel_names: ``$PnN`` values of the written file.
        keyword: Name of the keyword in the FlowIO convention (e.g. ``spill``), used in messages.

    Warns if the spillover matrix is not invertible (it is still written).

    Returns:
        tuple[str | None, list[str]]: Value to write (``None`` if it cannot be written) and descriptions of the
        information that is lost.
    """
    label = f"${keyword.upper()}"
    parts = [part.strip() for part in value.split(",")]
    try:
        n = int(parts[0])
        names = parts[1 : n + 1]
        coefficients = np.asarray([float(part) for part in parts[n + 1 :]], dtype=np.float64)
    except ValueError:
        return None, [f"{label} is malformed and is dropped."]
    if n < 1 or len(names) != n or coefficients.size != n * n or len(set(names)) != n:
        return None, [f"{label} is malformed and is dropped."]

    messages: list[str] = []
    available = set(channel_names)
    keep = [i for i, name in enumerate(names) if name in available]
    if len(keep) < n:
        missing = [name for name in names if name not in available]
        if len(keep) == 0:
            return None, [f"{label} refers to channels not present in var_names ({', '.join(missing)}) and is dropped."]
        messages.append(
            f"{label} refers to channels not present in var_names ({', '.join(missing)}). "
            "They are removed from the spillover matrix."
        )
        matrix_parts = parts[n + 1 :]
        parts = [str(len(keep))] + [names[i] for i in keep] + [matrix_parts[i * n + j] for i in keep for j in keep]
        value = ",".join(parts)
        n = len(keep)
        coefficients = coefficients.reshape(-1, len(names))[np.ix_(keep, keep)].ravel()

    matrix = coefficients.reshape(n, n)
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < n:
        warn(f"{label} matrix is not invertible, so it cannot be used for compensation.")

    return value, messages
