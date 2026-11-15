"""Constants describing how FCS data is laid out in an AnnData object (schema v1).

Layout produced by :func:`anndata_fcs.read_fcs`:

* ``X``: event matrix (events x channels) as stored in the file.
* ``layers["preprocessed"]``: optional, ``X`` with gain, log and time scaling applied. Not written to FCS files.
* ``var.index``: channel names (``$PnN``).
* ``var`` columns ``$PnS``, ``$PnR``, ``$PnE``, ...: per-channel keywords (see :data:`CHANNEL_COLUMN_PREFIX`).
* ``uns["fcs"]``: global keywords and file level metadata (see :class:`anndata_fcs._types.FCSMetadata`). Keywords keep
  the names parsed by FlowIO (lower case, without ``$``).

AnnData specific information (``obs``, ``obs_names`` and ``var`` columns other than keywords) is embedded into the FCS
TEXT segment as JSON encoded keywords using the :data:`KEYWORD_PREFIX` namespace.
"""

import re

from flowio.fcs_keywords import FCS_STANDARD_REQUIRED_KEYWORDS

SCHEMA_VERSION: int = 1

UNS_KEY: str = "fcs"
"""Key in ``adata.uns`` holding FCS file level metadata."""

CHANNEL_COLUMN_PREFIX: str = "$Pn"
"""Prefix of ``var`` columns holding per-channel keywords: ``$Pn`` followed by the upper case keyword suffix, e.g.
``$PnS`` for ``$P1S``, ``$P2S``, ... or ``$PnDISPLAY`` for ``P1DISPLAY``, ``P2DISPLAY``, ..."""

PREPROCESSED_LAYER: str = "preprocessed"
"""Layer holding the event data with gain, log and time scaling applied (``read_fcs(..., preprocess=True)``)."""

DEFAULT_INDEX_CHANNEL: str = "barcode_rank"
"""Default name of the synthetic event index channel added when writing FCS files."""

# Keyword namespace for AnnData payloads. FlowIO lower-cases keywords when parsing, so all keyword constants are
# lower case. They are written upper case to the FCS file.
KEYWORD_PREFIX: str = "anndata_fcs_"
KW_SCHEMA: str = KEYWORD_PREFIX + "schema"
KW_INDEX_CHANNEL: str = KEYWORD_PREFIX + "index_channel"
KW_OBS: str = KEYWORD_PREFIX + "obs"
KW_VAR: str = KEYWORD_PREFIX + "var"

STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(FCS_STANDARD_REQUIRED_KEYWORDS)
"""Global keywords describing the binary layout. They are regenerated on write and never copied."""

STRUCTURAL_CHANNEL_SUFFIXES: frozenset[str] = frozenset({"b"})
"""Per-channel keyword suffixes describing the binary layout (``$PnB``). They are regenerated on write."""

CHANNEL_KEYWORD_PATTERN: re.Pattern[str] = re.compile(r"^p(\d+)(\D.*)$")
"""Matches per-channel keywords such as ``p1n``, ``p12s`` or ``p3display``."""

SPILLOVER_KEYWORDS: tuple[str, ...] = ("spillover", "spill")
"""Keywords holding a spillover matrix in the ``n,name_1,...,name_n,s_11,...,s_nn`` format: ``$SPILLOVER`` (FCS 3.1)
and ``$SPILL`` (FCS 3.0, e.g. written by BD FACSDiva)."""

FCS_DELIMITER: str = "/"
"""Delimiter used by :func:`flowio.create_fcs` when writing the TEXT segment."""

MAX_EXACT_FLOAT32_INT: int = 2**24
"""Largest integer for which all smaller integers are exactly representable as float32."""

TEXT_SIZE_WARNING_BYTES: int = 10_000_000
"""TEXT segments above this size trigger a warning, since many FCS readers struggle with large TEXT segments."""

MAX_TEXT_BYTES: int = 99_999_999 - 255
"""The FCS header stores the TEXT end offset in 8 ASCII digits, which limits the TEXT segment size. FlowIO starts the
TEXT segment at byte 256, so it ends at byte ``255 + size``."""
