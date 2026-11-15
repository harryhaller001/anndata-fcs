"""Constants describing how FCS data is laid out in an AnnData object (schema v1).

AnnData layout produced by read_fcs
-----------------------------------

    X                       Event matrix (events x channels) as stored in the file.
    layers["preprocessed"]  Optional. X with gain, log and time scaling applied.
                            Not written to FCS files.
    var.index               Channel names ($PnN).
    var["$PnS"], ...        Per-channel keywords, one column per keyword ($PnS, $PnR, $PnE, ...).
    uns["fcs"]              Global keywords and file level metadata (see FCSMetadata).
                            Keywords keep the names parsed by FlowIO (lower case, without "$").

AnnData specific information (obs, obs_names and var columns other than keywords) is
embedded into the FCS TEXT segment as JSON encoded keywords in the "anndata_fcs_" namespace.

Constants
---------

    SCHEMA_VERSION
        Version of this layout, written to the KW_SCHEMA keyword.

    UNS_KEY
        Key in adata.uns holding FCS file level metadata.

    CHANNEL_COLUMN_PREFIX
        Prefix of var columns holding per-channel keywords. It is followed by the upper case
        keyword suffix, e.g. "$PnS" stands for $P1S, $P2S, ... and "$PnDISPLAY" for
        P1DISPLAY, P2DISPLAY, ...

    PREPROCESSED_LAYER
        Layer holding the event data with gain, log and time scaling applied
        (read_fcs(..., preprocess=True)).

    DEFAULT_INDEX_CHANNEL
        Default name of the synthetic event index channel added when writing FCS files.

    KEYWORD_PREFIX, KW_SCHEMA, KW_INDEX_CHANNEL, KW_OBS, KW_VAR
        Keyword namespace and keywords for the AnnData payloads. FlowIO lower-cases keywords
        when parsing, so these constants are lower case. They are written upper case to the
        FCS file.

    STRUCTURAL_KEYWORDS
        Global keywords describing the binary layout. They are regenerated on write and
        never copied.

    STRUCTURAL_CHANNEL_SUFFIXES
        Per-channel keyword suffixes describing the binary layout ($PnB). They are
        regenerated on write.

    CHANNEL_KEYWORD_PATTERN
        Matches per-channel keywords such as p1n, p12s or p3display.

    SPILLOVER_KEYWORDS
        Keywords holding a spillover matrix in the format n,name_1,...,name_n,s_11,...,s_nn:
        $SPILLOVER (FCS 3.1) and $SPILL (FCS 3.0, e.g. written by BD FACSDiva).

    FCS_DELIMITER
        Delimiter used by flowio.create_fcs when writing the TEXT segment.

    MAX_EXACT_FLOAT32_INT
        Largest integer for which all smaller integers are exactly representable as float32.

    TEXT_SIZE_WARNING_BYTES
        TEXT segments above this size trigger a warning, since many FCS readers struggle
        with large TEXT segments.

    MAX_TEXT_BYTES
        Maximum size of the TEXT segment. The FCS header stores the TEXT end offset in
        8 ASCII digits. FlowIO starts the TEXT segment at byte 256, so it ends at byte
        255 + size.
"""

import re

from flowio.fcs_keywords import FCS_STANDARD_REQUIRED_KEYWORDS

SCHEMA_VERSION: int = 1
UNS_KEY: str = "fcs"
CHANNEL_COLUMN_PREFIX: str = "$Pn"
PREPROCESSED_LAYER: str = "preprocessed"
DEFAULT_INDEX_CHANNEL: str = "barcode_rank"

KEYWORD_PREFIX: str = "anndata_fcs_"
KW_SCHEMA: str = KEYWORD_PREFIX + "schema"
KW_INDEX_CHANNEL: str = KEYWORD_PREFIX + "index_channel"
KW_OBS: str = KEYWORD_PREFIX + "obs"
KW_VAR: str = KEYWORD_PREFIX + "var"

STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(FCS_STANDARD_REQUIRED_KEYWORDS)
STRUCTURAL_CHANNEL_SUFFIXES: frozenset[str] = frozenset({"b"})
CHANNEL_KEYWORD_PATTERN: re.Pattern[str] = re.compile(r"^p(\d+)(\D.*)$")
SPILLOVER_KEYWORDS: tuple[str, ...] = ("spillover", "spill")

FCS_DELIMITER: str = "/"
MAX_EXACT_FLOAT32_INT: int = 2**24
TEXT_SIZE_WARNING_BYTES: int = 10_000_000
MAX_TEXT_BYTES: int = 99_999_999 - 255
