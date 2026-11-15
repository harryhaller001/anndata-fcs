from typing import NotRequired, TypedDict


class FCSMetadata(TypedDict):
    """Structure of ``adata.uns["fcs"]``."""

    schema_version: int
    """Version of the anndata-fcs layout."""

    version: str
    """FCS version of the source file (e.g. ``"3.1"``)."""

    text: dict[str, str]
    """Global TEXT keywords as parsed by FlowIO (lower case, without ``$``). Per-channel keywords are stored in ``var``
    columns (``$PnS``, ``$PnR``, ...). Structural keywords (offsets, ``$DATATYPE``, ``$PnB``, ...) are not stored,
    since they are regenerated when writing."""

    analysis: dict[str, str]
    """Keywords of the ANALYSIS segment. Not written back, since FlowIO cannot write ANALYSIS segments."""

    filename: NotRequired[str]
    """Name of the source file, if known."""
