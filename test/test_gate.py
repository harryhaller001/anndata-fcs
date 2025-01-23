from anndata import AnnData

from anndata_fcs import anndata_to_fcs, gate_polygon_subset


def test_gating(adata: AnnData) -> None:
    """Test gating."""

    gate_polygon_subset(
        fdata=anndata_to_fcs(adata=adata),
        polygon=[[90, 200], [5000, 200], [5000, 5000], [90, 5000]],
        x="CD3_TotalSeqB",
        y="CD4_TotalSeqB",
    )
