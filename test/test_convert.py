import flowio
import numpy as np
import pandas as pd
from anndata import AnnData

from anndata_fcs import anndata_to_fcs, fcs_to_anndata, fcs_to_dataframe


def test_anndata_to_fcs(adata: AnnData) -> None:
    """Testing anndata convert to fcs file."""

    fcs_data = anndata_to_fcs(adata=adata)

    assert isinstance(fcs_data, flowio.FlowData)

    # Compare arrays (+1 to length of adata.var, because barcode_rank is added to obs)
    assert fcs_data.channel_count == len(adata.var) + 1
    assert "barcode_rank" in adata.obs

    assert (
        np.reshape(fcs_data.events, (-1, fcs_data.channel_count))[:, : fcs_data.channel_count - 1] == adata.X.toarray()
    ).all()

    fcs_df = fcs_to_dataframe(fdata=fcs_data)

    assert isinstance(fcs_df, pd.DataFrame)

    adata_convert_no_uns = fcs_to_anndata(fdata=fcs_data, include_metadata=False)

    assert isinstance(adata_convert_no_uns, AnnData)
    assert len(adata_convert_no_uns.uns.keys()) == 0

    adata_convert = fcs_to_anndata(fdata=fcs_data, include_metadata=True)

    assert isinstance(adata_convert, AnnData)

    default_keys = [
        "beginanalysis",
        "endanalysis",
        "beginstext",
        "endstext",
        "begindata",
        "enddata",
        "tot",
        "par",
        "mode",
    ]
    for key in default_keys:
        assert key in adata_convert.uns.keys()
