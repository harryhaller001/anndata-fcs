import io
import warnings

import flowio
import numpy as np
import pandas as pd
from anndata import AnnData, ImplicitModificationWarning


def anndata_to_fcs(adata: AnnData) -> flowio.FlowData:
    """Create fcs object from anndata.

    Args:
        adata (anndata.AnnData): AnnData object to convert.

    Returns:
        flowio.FlowData: FlowData instance.

    Raises:
        AssertionError: If array conversion fails.
    """
    fcs_obj = flowio.create_fcs(
        file_handle=io.BytesIO(),
        event_data=np.column_stack([adata.X.toarray(), np.array(range(len(adata.obs)))]).flatten(),
        channel_names=adata.var.index.tolist() + ["barcode_rank"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ImplicitModificationWarning)

        # Add barcode rank to anndata object
        adata.obs["barcode_rank"] = pd.Series(
            range(len(adata.obs)),
            index=adata.obs.index,
            dtype="int64",
        )

    fdata = flowio.FlowData(fcs_obj)

    # Check if arrays are the same
    assert (
        np.reshape(fdata.events, (-1, fdata.channel_count))[:, : fdata.channel_count - 1] == adata.X.toarray()
    ).all()

    return fdata


def fcs_to_dataframe(fdata: flowio.FlowData) -> pd.DataFrame:
    """Converts FlowData instance to pandas DataFrame.

    Args:
        fdata (flowio.FlowData): FlowData instance to convert.

    Returns:
        pandas.DataFrame: Pandas dataframe.
    """
    return pd.DataFrame(
        np.reshape(fdata.events, (-1, fdata.channel_count)),
        columns=[v["PnN"] for v in fdata.channels.values()],
    )


def fcs_to_anndata(fdata: flowio.FlowData, include_metadata: bool = True) -> AnnData:
    """Converts FlowData instance to AnnData object.

    Args:
        fdata (flowio.FlowData): FlowData instance to convert.
        include_metadata (bool, optional): If `True` is adds to FCS file meta data and `.uns` of the AnnData object.

    Returns:
        anndata.AnnData: AnnData object.
    """
    data_array = np.reshape(fdata.events, (-1, fdata.channel_count))

    if include_metadata is True:
        adata = AnnData(X=data_array, uns=fdata.text)
    else:
        adata = AnnData(X=data_array)

    # Add channels as varnames
    adata.var_names = [item["PnN"] for item in fdata.channels.values()]

    return adata
