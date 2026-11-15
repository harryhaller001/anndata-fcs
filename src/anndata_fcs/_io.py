import os
from typing import IO, Any

import flowio
from anndata import AnnData

from anndata_fcs._convert import _anndata_to_fcs_bytes, fcs_to_anndata
from anndata_fcs._schema import DEFAULT_INDEX_CHANNEL


def read_fcs(
    file: str | os.PathLike[str] | IO[bytes] | flowio.FlowData,
    *,
    preprocess: bool = False,
    **flowio_kwargs: Any,
) -> AnnData:
    """Read an FCS file into an AnnData object.

    Args:
        file: Path, binary file handle or FlowData instance.
        preprocess: Apply gain, log and time scaling to ``X``. ``raw`` always holds the unprocessed event data. See
            :func:`anndata_fcs.fcs_to_anndata`.
        **flowio_kwargs: Passed to :class:`flowio.FlowData` (e.g. ``ignore_offset_error=True``).

    Returns:
        AnnData: AnnData object. See :func:`anndata_fcs.fcs_to_anndata` for the layout.
    """
    if isinstance(file, flowio.FlowData):
        if len(flowio_kwargs) > 0:
            raise TypeError("FlowIO arguments cannot be used with a FlowData instance.")
        fdata = file
    else:
        fdata = flowio.FlowData(file, **flowio_kwargs)
    return fcs_to_anndata(fdata, preprocess=preprocess)


def write_fcs(
    adata: AnnData,
    file: str | os.PathLike[str] | IO[bytes],
    *,
    pns_key: str | None = None,
    index_channel: str | None = DEFAULT_INDEX_CHANNEL,
    store_obs: bool = True,
    strict: bool = False,
    use_raw: bool | None = None,
) -> None:
    """Write an AnnData object to an FCS file.

    By default the unprocessed event data in ``raw`` is written, if present. See :func:`anndata_fcs.anndata_to_fcs` for the arguments and which information is stored.

    Args:
        adata: AnnData object to write. It is not modified.
        file: Path or binary file handle.
        pns_key: Column of ``var`` (``raw.var``) used for ``$PnS``. Defaults to ``None``, which keeps the ``p<n>s``
            keywords of ``uns["fcs"]["text"]``.
        index_channel: Name of the synthetic event index channel. Use ``None`` to omit it.
        store_obs: If ``True``, store ``obs`` and ``obs_names``.
        strict: If ``True``, raise an error instead of warning when information would be lost.
        use_raw: Write the unprocessed event data in ``raw`` (``True``) or ``X`` (``False``). Defaults to ``raw`` if
            present, otherwise ``X``.
    """
    fcs_bytes = _anndata_to_fcs_bytes(
        adata,
        pns_key=pns_key,
        index_channel=index_channel,
        store_obs=store_obs,
        strict=strict,
        use_raw=use_raw,
    )
    if isinstance(file, str | os.PathLike):
        with open(file, "wb") as handle:
            handle.write(fcs_bytes)
    else:
        file.write(fcs_bytes)
