from typing import NotRequired, TypedDict


class FCSMetadata(TypedDict):
    """Structure of meta data stored in `adata.uns["fcs"]`."""

    schema_version: int
    """Version of the `anndata-fcs` schema."""

    version: str
    """FCS version of the source file."""

    text: dict[str, str]
    """Global TEXT keywords as parsed by FlowIO. Per-channel keywords are stored in var columns."""

    analysis: dict[str, str]
    """Keywords of the ANALYSIS segment. Not written back, since FlowIO cannot write ANALYSIS segments."""

    filename: NotRequired[str]
    """Name of the source file, if known (optional)."""
