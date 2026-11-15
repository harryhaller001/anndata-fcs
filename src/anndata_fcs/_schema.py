"""Constants describing how FCS data is laid out in an AnnData object (schema v1).

Layout produced by :func:`anndata_fcs.fcs_to_anndata`:

* ``raw``: unprocessed event matrix (events x channels) and a copy of ``var``.
* ``X``: event matrix (events x channels), unprocessed or preprocessed (``uns["fcs"]["preprocessed"]``).
* ``var.index``: channel names (``$PnN``).
* ``uns["fcs"]``: keywords and file level metadata (see :class:`anndata_fcs._types.FCSMetadata`). Keywords keep the
  names parsed by FlowIO (lower case, without ``$``), including per-channel keywords such as ``p1s`` or ``p1r``.

AnnData specific information (``obs``, ``obs_names``, extra ``var`` columns and ``uns``) is embedded into the FCS
TEXT segment as JSON encoded keywords using the :data:`KEYWORD_PREFIX` namespace.
"""

import re

from flowio.fcs_keywords import FCS_STANDARD_REQUIRED_KEYWORDS

SCHEMA_VERSION: int = 1

UNS_KEY: str = "fcs"
"""Key in ``adata.uns`` holding FCS file level metadata."""

DEFAULT_INDEX_CHANNEL: str = "barcode_rank"
"""Default name of the synthetic event index channel added when writing FCS files."""

# Keyword namespace for AnnData payloads. FlowIO lower-cases keywords when parsing, so all keyword constants are
# lower case. They are written upper case to the FCS file.
KEYWORD_PREFIX: str = "anndata_fcs_"
KW_SCHEMA: str = KEYWORD_PREFIX + "schema"
KW_INDEX_CHANNEL: str = KEYWORD_PREFIX + "index_channel"
KW_OBS: str = KEYWORD_PREFIX + "obs"
KW_VAR: str = KEYWORD_PREFIX + "var"
KW_UNS: str = KEYWORD_PREFIX + "uns"

STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(FCS_STANDARD_REQUIRED_KEYWORDS)
"""Global keywords describing the binary layout. They are regenerated on write and never copied."""

STRUCTURAL_CHANNEL_SUFFIXES: frozenset[str] = frozenset({"b"})
"""Per-channel keyword suffixes describing the binary layout (``$PnB``). They are regenerated on write."""

CHANNEL_KEYWORD_PATTERN: re.Pattern[str] = re.compile(r"^p(\d+)(\D.*)$")
"""Matches per-channel keywords such as ``p1n``, ``p12s`` or ``p3display``."""

FCS_DELIMITER: str = "/"
"""Delimiter used by :func:`flowio.create_fcs` when writing the TEXT segment."""

MAX_EXACT_FLOAT32_INT: int = 2**24
"""Largest integer for which all smaller integers are exactly representable as float32."""

TEXT_SIZE_WARNING_BYTES: int = 10_000_000
"""Embedded payloads above this size trigger a warning, since many FCS readers struggle with large TEXT segments."""

MAX_TEXT_BYTES: int = 99_999_999 - 256
"""The FCS header stores the TEXT end offset in 8 ASCII digits, which limits the TEXT segment size."""
