"""Classification of FCS TEXT keywords and construction of FlowIO metadata dictionaries.

All functions in this module operate on plain dictionaries and are independent of AnnData.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from anndata_fcs._exceptions import FCSConversionWarning
from anndata_fcs._schema import (
    CHANNEL_KEYWORD_PATTERN,
    FCS_DELIMITER,
    KEYWORD_PREFIX,
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
    """Keywords describing the binary layout (offsets, byte order, data type, ...)."""

    channels: dict[int, dict[str, str]] = field(default_factory=dict)
    """Per-channel keywords, keyed by channel number (1-based) and keyword suffix (``"n"``, ``"s"``, ...)."""

    global_: dict[str, str] = field(default_factory=dict)
    """All remaining keywords."""

    payload: dict[str, str] = field(default_factory=dict)
    """Keywords in the anndata-fcs namespace."""


def split_text(text: Mapping[str, str], n_channels: int) -> SplitText:
    """Split parsed TEXT keywords into structural, per-channel, payload and global keywords.

    Args:
        text: Keyword/value mapping as parsed by FlowIO (lower case keys without ``$``).
        n_channels: Number of channels (``$PAR``). Keywords of the form ``p<n>...`` are only treated as channel
            keywords when ``1 <= n <= n_channels``.

    Returns:
        SplitText: Classified keywords.
    """
    result = SplitText(channels={i: {} for i in range(1, n_channels + 1)})

    for key, value in text.items():
        key = key.lower()

        if key in STRUCTURAL_KEYWORDS:
            result.structural[key] = value.strip()
            continue

        if key.startswith(KEYWORD_PREFIX):
            result.payload[key] = value
            continue

        match = CHANNEL_KEYWORD_PATTERN.match(key)
        if match is not None:
            channel_number = int(match.group(1))
            if 1 <= channel_number <= n_channels:
                result.channels[channel_number][match.group(2)] = value
                continue

        result.global_[key] = value

    return result


def normalize_keyword(key: str) -> str:
    """Normalise a keyword to the FlowIO convention (lower case, without leading ``$``)."""
    return key.lower().lstrip("$")


def _check_keyword(key: str) -> bool:
    """Check whether a keyword name is valid according to FCS 3.1. Warns if not.

    Keyword names must not be empty and may only contain printable ASCII characters (only values are UTF-8).
    """
    if key == "" or not all(32 <= ord(character) <= 126 for character in key):
        warnings.warn(
            f"Keyword {key!r} is not valid in FCS 3.1, which only allows non-empty keywords of printable ASCII "
            "characters. It is not written.",
            FCSConversionWarning,
            stacklevel=4,
        )
        return False
    return True


def _check_value(key: str, value: str) -> bool:
    """Check whether a keyword value can be written and read back by FlowIO. Warns if not."""
    if value == "":
        warnings.warn(
            f"Keyword '{key}' has an empty value, which is not allowed by the FCS standard. It is not written.",
            FCSConversionWarning,
            stacklevel=4,
        )
        return False

    if value.startswith(FCS_DELIMITER) or value.endswith(FCS_DELIMITER):
        warnings.warn(
            f"Value of keyword '{key}' starts or ends with the delimiter '{FCS_DELIMITER}' and cannot be parsed "
            "reliably by FlowIO.",
            FCSConversionWarning,
            stacklevel=4,
        )

    if "$" in value:
        warnings.warn(
            f"Value of keyword '{key}' contains '$', which FlowIO removes when parsing FCS files.",
            FCSConversionWarning,
            stacklevel=4,
        )

    return True


def build_metadata(
    channels: Sequence[Mapping[str, str]],
    global_keywords: Mapping[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Build the arguments for :func:`flowio.create_fcs` from channel and global keywords.

    ``$PnN``, ``$PnB`` and ``$PnE`` are ignored, since they are set by :func:`flowio.create_fcs`. ``$PnS`` is
    returned separately, because :func:`flowio.create_fcs` silently drops it from the metadata dictionary.

    Args:
        channels: Per-channel keyword suffix/value mappings, in channel order.
        global_keywords: Global keywords. Structural keywords are ignored.

    Returns:
        tuple[dict[str, str], list[str]]: Metadata dictionary and list of ``$PnS`` values (empty string if absent).
    """
    metadata: dict[str, str] = {}
    pns_labels: list[str] = []

    for channel_index, channel in enumerate(channels):
        channel_number = channel_index + 1
        pns_labels.append("")

        for suffix, value in channel.items():
            if suffix in ("n", "b", "e"):
                continue

            key = f"p{channel_number}{suffix}"
            if not _check_keyword(key) or not _check_value(key, value):
                continue

            if suffix == "s":
                pns_labels[channel_index] = value
            else:
                metadata[_escape_delimiter(key)] = value

    for key, value in global_keywords.items():
        key = normalize_keyword(key)
        if key in STRUCTURAL_KEYWORDS:
            continue

        match = CHANNEL_KEYWORD_PATTERN.match(key)
        if match is not None and 1 <= int(match.group(1)) <= len(channels):
            # Global keywords only hold $Pn keywords of channels beyond $PAR. Written as is, they would be attributed to
            # a different channel.
            warnings.warn(
                f"Global keyword '{key}' refers to channel {int(match.group(1))} of the written file. Per-channel "
                f"keywords are taken from var columns ('pn{match.group(2)}'). It is not written.",
                FCSConversionWarning,
                stacklevel=3,
            )
            continue

        if not _check_keyword(key) or not _check_value(key, value):
            continue
        metadata[_escape_delimiter(key)] = value

    return metadata, pns_labels


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
    decades, log0 = (float(part) for part in value.split(","))
    if decades != 0 and log0 == 0:
        log0 = 1.0
    return decades, log0


def linear_range(decades: float, log0: float) -> str:
    """``$PnR`` of a log scaled channel (``$PnE/decades,log0/``) after conversion to linear scale values."""
    return str(math.ceil(10**decades * log0))


def check_spillover(value: str, channel_names: Sequence[str]) -> tuple[str | None, list[str]]:
    """Check ``$SPILLOVER`` against the channels of the file that is written.

    ``$SPILLOVER`` is ``n,name_1,...,name_n,s_11,s_12,...,s_nn``, where the names are ``$PnN`` values and the spillover
    coefficients are given row-wise. Channels which do not exist in the written file (e.g. after subsetting ``var``)
    are removed from the matrix, since FCS 3.1 requires all names to refer to parameters.

    Args:
        value: Value of the ``$SPILLOVER`` keyword.
        channel_names: ``$PnN`` values of the written file.

    Warns if the spillover matrix is not invertible (it is still written).

    Returns:
        tuple[str | None, list[str]]: Value to write (``None`` if it cannot be written) and descriptions of the
        information that is lost.
    """
    parts = [part.strip() for part in value.split(",")]
    try:
        n = int(parts[0])
        names = parts[1 : n + 1]
        coefficients = np.asarray([float(part) for part in parts[n + 1 :]], dtype=np.float64)
    except ValueError:
        return None, ["$SPILLOVER is malformed and is dropped."]
    if n < 1 or len(names) != n or coefficients.size != n * n or len(set(names)) != n:
        return None, ["$SPILLOVER is malformed and is dropped."]

    messages: list[str] = []
    available = set(channel_names)
    keep = [i for i, name in enumerate(names) if name in available]
    if len(keep) < n:
        missing = [name for name in names if name not in available]
        if len(keep) == 0:
            return None, [
                f"$SPILLOVER refers to channels not present in var_names ({', '.join(missing)}) and is dropped."
            ]
        messages.append(
            f"$SPILLOVER refers to channels not present in var_names ({', '.join(missing)}). "
            "They are removed from the spillover matrix."
        )
        matrix_parts = parts[n + 1 :]
        parts = [str(len(keep))] + [names[i] for i in keep] + [matrix_parts[i * n + j] for i in keep for j in keep]
        value = ",".join(parts)
        n = len(keep)
        coefficients = coefficients.reshape(-1, len(names))[np.ix_(keep, keep)].ravel()

    matrix = coefficients.reshape(n, n)
    if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < n:
        warnings.warn(
            "$SPILLOVER matrix is not invertible, so it cannot be used for compensation.",
            FCSConversionWarning,
            stacklevel=4,
        )

    return value, messages
