from typing import NotRequired, TypedDict


class FCSMetadata(TypedDict):
    """Structure of ``adata.uns["fcs"]``."""

    schema_version: int
    """Version of the anndata-fcs layout."""

    version: str
    """FCS version of the source file (e.g. ``"3.1"``)."""

    preprocessed: bool
    """Whether gain, log and time scaling were applied to ``X`` when reading."""

    text: dict[str, str]
    """Global TEXT keywords (lower case, without ``$``), excluding structural and per-channel keywords."""

    analysis: dict[str, str]
    """Keywords of the ANALYSIS segment. Not written back, since FlowIO cannot write ANALYSIS segments."""

    source: dict[str, object]
    """Structural keywords of the source file. Informational only, never written back."""

    filename: NotRequired[str]
    """Name of the source file, if known."""
