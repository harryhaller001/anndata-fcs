from typing import NotRequired, TypedDict


class FCSMetadata(TypedDict):
    """Structure of ``adata.uns["fcs"]``."""

    schema_version: int
    """Version of the anndata-fcs layout."""

    version: str
    """FCS version of the source file (e.g. ``"3.1"``)."""

    preprocessed: bool
    """Whether gain, log and time scaling were applied to ``X`` when reading. ``raw`` is never preprocessed."""

    text: dict[str, str]
    """TEXT keywords as parsed by FlowIO (lower case, without ``$``), including per-channel keywords (``p1n``,
    ``p1s``, ...). Structural keywords are stored in ``source``."""

    analysis: dict[str, str]
    """Keywords of the ANALYSIS segment. Not written back, since FlowIO cannot write ANALYSIS segments."""

    source: dict[str, str]
    """Structural keywords of the source file (offsets, ``$DATATYPE``, ``$PnB``, ...). Informational only, never
    written back."""

    filename: NotRequired[str]
    """Name of the source file, if known."""


class FCSWriteOptions(TypedDict, total=False):
    """Options of :func:`anndata_fcs.anndata_to_fcs`, also accepted by :func:`anndata_fcs.write_fcs`."""

    pns_key: str | None
    """Column of ``var`` (``raw.var``) used for ``$PnS``."""

    index_channel: str | None
    """Name of the synthetic event index channel, ``None`` to omit it."""

    store_obs: bool
    """Whether ``obs`` and ``obs_names`` are stored."""

    strict: bool
    """Raise an error instead of warning when information would be lost."""

    use_raw: bool | None
    """Write ``raw`` (``True``) or ``X`` (``False``)."""
