import pandas as pd
import flowio
import numpy as np
import anndata as ad
import io
import warnings


def anndata_to_fcs(adata_obj: ad.AnnData) -> flowio.FlowData:
    # create fcs object
    fcs_obj = flowio.create_fcs(
        file_handle=io.BytesIO(),
        event_data=np.column_stack([adata_obj.X.toarray(), np.array(range(len(adata_obj.obs)))]).flatten(),
        channel_names=adata_obj.var.index.tolist() + ["barcode_rank"],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ad.ImplicitModificationWarning)
        adata_obj.obs["barcode_rank"] = pd.Series(
            range(len(adata_obj.obs)),
            index=adata_obj.obs.index,
            dtype="int64",
        )

    fdata = flowio.FlowData(fcs_obj)

    # Check if arrays are the same
    assert (
        np.reshape(fdata.events, (-1, fdata.channel_count))[:, : fdata.channel_count - 1] == adata_obj.X.toarray()
    ).all()

    return fdata


def fcs_to_dataframe(fdata: flowio.FlowData) -> pd.DataFrame:
    return pd.DataFrame(
        np.reshape(fdata.events, (-1, fdata.channel_count)),
        columns=[v["PnN"] for v in fdata.channels.values()],
    )


def fcs_to_anndata(fdata: flowio.FlowData, include_metadata: bool = True) -> ad.AnnData:
    data_array = np.reshape(fdata.events, (-1, fdata.channel_count))

    if include_metadata is True:
        adata = ad.AnnData(X=data_array, uns=fdata.text)
    else:
        adata = ad.AnnData(X=data_array)

    # Add channels as varnames
    adata.var_names = [item["PnN"] for item in fdata.channels.values()]

    return adata
