import io
import os
from typing import IO, Any

import flowio
from anndata import AnnData

from anndata_fcs._convert import fcs_to_anndata, prepare_fcs
from anndata_fcs._schema import DEFAULT_INDEX_CHANNEL


def read_fcs(
    file: str | os.PathLike[str] | IO[bytes] | flowio.FlowData,
    *,
    preprocess: bool = False,
    **flowio_kwargs: Any,
) -> AnnData:
    """Read an FCS file into an AnnData object.

    ``X`` holds the event data as stored in the file. All metadata of the FCS file is kept:

    * Channel names (``$PnN``) become ``var_names``. Duplicate names are made unique by appending ``-1``, ``-2``, ...
    * Per-channel keywords are stored in ``var`` columns named after the keyword: ``$PnS`` holds ``$P1S``, ``$P2S``,
      ..., ``$PnR`` holds ``$P1R``, ``$P2R``, ... and so on for every keyword of the file (e.g. ``$PnDISPLAY``).
      Values are raw strings, missing for channels without the keyword.
    * Global keywords are stored in ``uns["fcs"]["text"]`` with the names parsed by FlowIO (lower case, without
      ``$``). Values are raw strings. Keywords describing the binary layout (offsets, ``$DATATYPE``, ``$PnB``, ...) are
      not stored, since they are regenerated when writing.
    * The ANALYSIS segment and the FCS version are stored in ``uns["fcs"]``.
    * If the file was written by :func:`write_fcs`, ``obs``, ``obs_names`` and additional ``var`` columns are
      restored and the synthetic event index channel is removed from the event data. If ``obs`` was not stored,
      ``obs_names`` are taken from the event index channel, i.e. the positions of the events in the written AnnData
      object.

    Note:
        FlowIO closes file handles after reading them.

    Args:
        file: Path, binary file handle or FlowData instance. A FlowData instance is not modified and does not share
            memory with the result.
        preprocess: If ``True``, additionally store the event data with gain (``$PnG``), log (``$PnE``) and time
            (``$TIMESTEP``) scaling applied in ``layers["preprocessed"]``. ``X`` and the keywords are not modified.
            The layer is derived from ``X`` and the keywords, so :func:`write_fcs` does not write it.
        **flowio_kwargs: Passed to :class:`flowio.FlowData` (e.g. ``ignore_offset_error=True``).

    Returns:
        AnnData: AnnData object with events as observations and channels as variables.

    Raises:
        TypeError: If FlowIO arguments are given with a FlowData instance.
        ValueError: If the file has no event data.
    """
    if isinstance(file, flowio.FlowData):
        if len(flowio_kwargs) > 0:
            raise TypeError("FlowIO arguments cannot be used with a FlowData instance.")
        return fcs_to_anndata(file, preprocess=preprocess, owns_events=False)
    # The FlowData instance is not accessible to the caller, so its event buffer is used without copying
    return fcs_to_anndata(flowio.FlowData(file, **flowio_kwargs), preprocess=preprocess, owns_events=True)


def write_fcs(
    adata: AnnData,
    file: str | os.PathLike[str] | IO[bytes],
    *,
    index_channel: str | None = DEFAULT_INDEX_CHANNEL,
    store_obs: bool = True,
) -> None:
    """Write an AnnData object to an FCS file.

    The FCS file holds the following information:

    * ``X`` as event data (32-bit float, the only data type supported by FlowIO) with ``var_names`` as ``$PnN``.
      Channels stored on log scale (``$PnE``) are converted to linear scale.
    * ``var`` columns ``$PnS``, ``$PnR``, ... as per-channel keywords (as strings, missing values are not written).
      They follow their channel if channels are removed, reordered or renamed. E.g. set marker labels with
      ``adata.var["$PnS"] = adata.var["gene_ids"]``.
    * ``uns["fcs"]["text"]`` as global keywords. ``$SPILLOVER`` and ``$SPILL`` are restricted to the written
      channels. ``$ORIGINALITY`` is set to ``DataModified`` and ``$LAST_MODIFIED`` to the current time,
      since the file is a derived copy of the data.
    * ``obs`` (including ``obs_names``) and other ``var`` columns as JSON encoded keywords in the ``ANNDATA_FCS_*``
      namespace, which are restored by :func:`read_fcs`.

    Only ``X`` is written. To write ``raw`` (e.g. counts of a scanpy object), use
    ``write_fcs(adata.raw.to_adata(), ...)``.

    Information that cannot be stored (``raw``, ``layers`` except ``layers["preprocessed"]``, ``obsm``, ``obsp``,
    ``varm``, ``varp``, ``uns`` entries other than ``uns["fcs"]``, unsupported column dtypes, keywords that cannot be
    written, float precision beyond float32) triggers a :class:`FCSConversionWarning`. To raise an error instead, turn
    the warning into an exception:

    .. code-block:: python

        with warnings.catch_warnings():
            warnings.simplefilter("error", FCSConversionWarning)
            write_fcs(adata, "out.fcs")

    All checks run before the file is opened, so a failed conversion does not modify an existing file.

    Args:
        adata: AnnData object to write. It is not modified.
        file: Path or binary file handle.
        index_channel: Name of a synthetic channel holding the position of each event in ``adata``. It allows mapping
            events of exported gates back to observations. Use ``None`` to omit it.
        store_obs: If ``True``, store ``obs`` and ``obs_names`` in the FCS file.

    Raises:
        TypeError: If the event data is not numeric.
        ValueError: If channel names or keywords are invalid, or the metadata exceeds the size of the FCS TEXT
            segment.
    """
    prepared = prepare_fcs(adata, index_channel=index_channel, store_obs=store_obs)
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
