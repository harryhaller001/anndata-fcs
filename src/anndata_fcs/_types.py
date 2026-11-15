from typing import NotRequired, TypedDict


class FCSMetadata(TypedDict):
    """Structure of adata.uns["fcs"].

    Fields
    ------

        schema_version
            Version of the anndata-fcs layout.

        version
            FCS version of the source file (e.g. "3.1").

        text
            Global TEXT keywords as parsed by FlowIO (lower case, without "$").
            Per-channel keywords are stored in var columns ($PnS, $PnR, ...).
            Structural keywords (offsets, $DATATYPE, $PnB, ...) are not stored,
            since they are regenerated when writing.

        analysis
            Keywords of the ANALYSIS segment. Not written back, since FlowIO cannot
            write ANALYSIS segments.

        filename
            Name of the source file, if known. Optional.
    """

    schema_version: int
    version: str
    text: dict[str, str]
    analysis: dict[str, str]
    filename: NotRequired[str]
