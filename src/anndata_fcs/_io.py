import io
import os
from typing import IO, Any, Unpack

import flowio
from anndata import AnnData

from anndata_fcs._convert import _fcs_to_anndata, _prepare_fcs
from anndata_fcs._types import FCSWriteOptions


def read_fcs(
    file: str | os.PathLike[str] | IO[bytes] | flowio.FlowData,
    *,
    preprocess: bool = False,
    raw: bool = True,
    **flowio_kwargs: Any,
) -> AnnData:
    """Read an FCS file into an AnnData object.

    Note:
        FlowIO closes file handles after reading them.

    Args:
        file: Path, binary file handle or FlowData instance.
        preprocess: Apply gain, log and time scaling to ``X``. ``raw`` always holds the unprocessed event data. See
            :func:`anndata_fcs.fcs_to_anndata`.
        raw: Store the unprocessed event data in ``raw``. Use ``False`` to halve the memory usage.
        **flowio_kwargs: Passed to :class:`flowio.FlowData` (e.g. ``ignore_offset_error=True``).

    Returns:
        AnnData: AnnData object. See :func:`anndata_fcs.fcs_to_anndata` for the layout.
    """
    if isinstance(file, flowio.FlowData):
        if len(flowio_kwargs) > 0:
            raise TypeError("FlowIO arguments cannot be used with a FlowData instance.")
        return _fcs_to_anndata(file, preprocess=preprocess, raw=raw, owns_events=False)
    # The FlowData instance is not accessible to the caller, so its event buffer is used without copying
    return _fcs_to_anndata(flowio.FlowData(file, **flowio_kwargs), preprocess=preprocess, raw=raw, owns_events=True)


def write_fcs(adata: AnnData, file: str | os.PathLike[str] | IO[bytes], **options: Unpack[FCSWriteOptions]) -> None:
    """Write an AnnData object to an FCS file.

    By default the unprocessed event data in ``raw`` is written, if present. See :func:`anndata_fcs.anndata_to_fcs` for
    which information is stored.

    Args:
        adata: AnnData object to write. It is not modified.
        file: Path or binary file handle.
        **options: Options of :func:`anndata_fcs.anndata_to_fcs` (``pns_key``, ``index_channel``, ``store_obs``,
            ``strict`` and ``use_raw``).
    """
    # All checks run before the file is opened, so a failed conversion does not truncate an existing file
    prepared = _prepare_fcs(adata, **options)
    if isinstance(file, str | os.PathLike):
        with open(file, "wb") as handle:
            prepared.write(handle)
    elif file.seekable() and file.tell() == 0:
        prepared.write(file)
    else:
        # flowio.create_fcs writes from position 0 of a seekable handle
        buffer = io.BytesIO()
        prepared.write(buffer)
        file.write(buffer.getbuffer())
