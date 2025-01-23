from anndata import AnnData
from matplotlib.axes import Axes

from anndata_fcs import anndata_to_fcs, fcs_to_dataframe, gate_polygon, scatter


def test_plotting(adata: AnnData) -> None:
    """Test plotting function."""

    ax = scatter(
        data=adata,
        x="CD3_TotalSeqB",
        y="CD4_TotalSeqB",
        gates={"CD3+ CD4+": [[90, 200], [5000, 200], [5000, 5000], [90, 5000]]},
        color="gray",
    )

    assert isinstance(ax, Axes)

    scatter(
        data=adata,
        x="CD3_TotalSeqB",
        y="CD4_TotalSeqB",
        gates={"CD3+ CD4+": [[90, 200], [5000, 200], [5000, 5000], [90, 5000]]},
        highlight=gate_polygon(
            fdata=anndata_to_fcs(adata=adata),
            polygon=[[90, 200], [5000, 200], [5000, 5000], [90, 5000]],
            x="CD3_TotalSeqB",
            y="CD4_TotalSeqB",
        ),
        color="gray",
    )

    scatter(
        data=anndata_to_fcs(adata),
        x="CD3_TotalSeqB",
        y="CD4_TotalSeqB",
        density=True,
    )

    scatter(
        data=fcs_to_dataframe(anndata_to_fcs(adata)),
        x="CD3_TotalSeqB",
        y="CD4_TotalSeqB",
    )
