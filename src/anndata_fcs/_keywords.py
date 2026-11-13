"""Classification of FCS TEXT keywords and construction of FlowIO metadata dictionaries.

All functions in this module operate on plain dictionaries and are independent of AnnData.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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
            if not _check_value(key, value):
                continue

            if suffix == "s":
                pns_labels[channel_index] = value
            else:
                metadata[_escape_delimiter(key)] = value

    for key, value in global_keywords.items():
        key = key.lower().lstrip("$")
        if key in STRUCTURAL_KEYWORDS:
            continue
        if not _check_value(key, value):
            continue
        metadata[_escape_delimiter(key)] = value

    return metadata, pns_labels


def _escape_delimiter(key: str) -> str:
    """Double the delimiter in keywords. FlowIO only does this for values."""
    return key.replace(FCS_DELIMITER, FCS_DELIMITER * 2)
