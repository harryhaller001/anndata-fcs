import io

import flowio
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from anndata_fcs import FCSConversionWarning, anndata_to_fcs, fcs_to_anndata
from anndata_fcs._schema import KW_OBS, KW_SCHEMA, STRUCTURAL_KEYWORDS

# ---------------------------------------------------------------------------------------------------------------------
# FCS -> AnnData
# ---------------------------------------------------------------------------------------------------------------------


def test_fcs_to_anndata_float(lsr_ii_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(lsr_ii_fdata)

    assert adata.shape == (lsr_ii_fdata.event_count, lsr_ii_fdata.channel_count)
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X, lsr_ii_fdata.as_array(preprocess=False))

    assert list(adata.var_names) == lsr_ii_fdata.pnn_labels
    assert adata.var.loc["FSC-A", "pnr"] == "262144"
    assert adata.var.loc["FSC-A", "pnv"] == "611"
    assert adata.var.loc["FSC-A", "pndisplay"] == "LOG"
    assert pd.isna(adata.var.loc["FSC-W", "pndisplay"])
    assert "pnn" not in adata.var.columns and "pnb" not in adata.var.columns

    meta = adata.uns["fcs"]
    assert meta["version"] == "3.0"
    assert meta["preprocessed"] is False
    assert meta["text"]["cyt"] == "LSRII"
    assert meta["text"]["spill"].startswith("4,FITC-A")
    assert meta["text"]["tube name"] == "D6"
    assert not STRUCTURAL_KEYWORDS & set(meta["text"])
    assert not any(key.startswith("p1") for key in meta["text"])
    assert meta["source"]["datatype"] == "F"
    assert meta["source"]["tot"] == str(lsr_ii_fdata.event_count)
    assert meta["filename"] == "HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs"


def test_fcs_to_anndata_int(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata)

    assert adata.X.dtype == np.uint16
    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=False))
    assert adata.var.loc["FL1-H", "pns"] == "CD4 FITC"
    assert adata.var.loc["FL2-H", "pns"] == "CD8/PE"
    assert adata.var.loc["FL1-H", "pne"] == "4,1"
    assert adata.uns["fcs"]["analysis"] == {"gate1": "lymphocytes"}
    assert adata.uns["fcs"]["source"]["datatype"] == "I"
    assert adata.uns["fcs"]["source"]["pnb"] == ["16"] * 4
    # Keywords containing '/' are escaped, so the object can be written to h5ad
    assert adata.uns["fcs"]["text"]["custom%2Fkey"] == "custom value"


def test_fcs_to_anndata_preprocess(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata, preprocess=True)

    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=True))
    assert adata.uns["fcs"]["preprocessed"] is True
    assert (adata.var["pne"] == "0,0").all()
    assert (adata.var["png"] == "1.0").all()
    assert "timestep" not in adata.uns["fcs"]["text"]
    assert adata.uns["fcs"]["source"]["timestep"] == "0.01"


def test_fcs_to_anndata_without_events(int_fcs_bytes: bytes) -> None:
    fdata = flowio.FlowData(io.BytesIO(int_fcs_bytes), only_text=True)
    with pytest.raises(ValueError, match="no event data"):
        fcs_to_anndata(fdata)


def test_fcs_to_anndata_duplicate_channels() -> None:
    handle = io.BytesIO()
    flowio.create_fcs(handle, np.zeros(4, dtype=np.float32), ["A", "A"])
    handle.seek(0)
    with pytest.warns(FCSConversionWarning, match="not unique"):
        adata = fcs_to_anndata(flowio.FlowData(handle))
    assert adata.var_names.is_unique


# ---------------------------------------------------------------------------------------------------------------------
# AnnData -> FCS
# ---------------------------------------------------------------------------------------------------------------------


def test_anndata_to_fcs(adata_annotated: AnnData) -> None:
    fdata = anndata_to_fcs(adata_annotated)

    assert isinstance(fdata, flowio.FlowData)
    assert fdata.channel_count == adata_annotated.n_vars + 1
    assert fdata.pnn_labels == [*adata_annotated.var_names, "barcode_rank"]

    events = fdata.as_array(preprocess=False)
    np.testing.assert_array_equal(events[:, :-1], adata_annotated.X.toarray())
    np.testing.assert_array_equal(events[:, -1], np.arange(adata_annotated.n_obs))

    assert fdata.text[KW_SCHEMA] == "1"
    assert KW_OBS in fdata.text


def test_anndata_to_fcs_does_not_modify_input(adata_annotated: AnnData) -> None:
    before = adata_annotated.copy()
    anndata_to_fcs(adata_annotated)

    pd.testing.assert_frame_equal(adata_annotated.obs, before.obs)
    pd.testing.assert_frame_equal(adata_annotated.var, before.var)
    assert adata_annotated.uns.keys() == before.uns.keys()


def test_anndata_to_fcs_index_channel(adata: AnnData) -> None:
    fdata = anndata_to_fcs(adata, index_channel=None)
    assert fdata.pnn_labels == list(adata.var_names)

    fdata = anndata_to_fcs(adata, index_channel="event_id")
    assert fdata.pnn_labels[-1] == "event_id"

    with pytest.raises(ValueError, match="already exists"):
        anndata_to_fcs(adata, index_channel=adata.var_names[0])


def test_anndata_to_fcs_pns(adata: AnnData) -> None:
    fdata = anndata_to_fcs(adata, pns_key="gene_ids", index_channel=None)
    assert fdata.pns_labels == list(adata.var["gene_ids"])

    adata.var["pns"] = "label"
    fdata = anndata_to_fcs(adata, index_channel=None)
    assert fdata.pns_labels == ["label"] * adata.n_vars

    with pytest.raises(KeyError):
        anndata_to_fcs(adata, pns_key="missing")


def test_anndata_to_fcs_range() -> None:
    adata = AnnData(np.array([[1.0, 500_000.5], [2.0, 3.0]], dtype=np.float32))
    fdata = anndata_to_fcs(adata, index_channel=None)
    assert fdata.text["p1r"] == "262144"
    assert fdata.text["p2r"] == "500001"


def test_anndata_to_fcs_precision_loss() -> None:
    adata = AnnData(np.array([[0.1, 0.2]], dtype=np.float64))
    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        anndata_to_fcs(adata)
    with pytest.raises(ValueError, match="32-bit float"):
        anndata_to_fcs(adata, strict=True)


def test_anndata_to_fcs_unsupported_slots(adata: AnnData) -> None:
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))
    with pytest.warns(FCSConversionWarning, match="layers, obsm"):
        anndata_to_fcs(adata)
    with pytest.raises(ValueError, match="layers, obsm"):
        anndata_to_fcs(adata, strict=True)


def test_anndata_to_fcs_unsupported_metadata(adata: AnnData) -> None:
    adata.obs["time"] = pd.Timestamp("2020-01-01")
    adata.uns["frame"] = pd.DataFrame({"a": [1]})

    with pytest.warns(FCSConversionWarning) as record:
        fdata = anndata_to_fcs(adata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "time" in messages and "uns['frame']" in messages

    restored = fcs_to_anndata(fdata)
    assert "time" not in restored.obs.columns
    assert "frame" not in restored.uns


def test_anndata_to_fcs_analysis(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata)
    # Linear scale keeps the integer data exact in float32, so only the ANALYSIS segment is lost
    adata.var["pne"] = "0,0"

    with pytest.warns(FCSConversionWarning, match="ANALYSIS"):
        anndata_to_fcs(adata)
    with pytest.raises(ValueError, match="ANALYSIS"):
        anndata_to_fcs(adata, strict=True)


def test_anndata_to_fcs_invalid_input() -> None:
    with pytest.raises(TypeError, match="numeric"):
        anndata_to_fcs(AnnData(np.array([["a"]], dtype=object)))

    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["A", "A"]
    with pytest.raises(ValueError, match="unique"):
        anndata_to_fcs(adata)


def test_anndata_to_fcs_store_obs(adata_annotated: AnnData) -> None:
    fdata = anndata_to_fcs(adata_annotated, store_obs=False)
    assert KW_OBS not in fdata.text

    restored = fcs_to_anndata(fdata)
    assert list(restored.obs_names) == [str(i) for i in range(adata_annotated.n_obs)]
    assert len(restored.obs.columns) == 0
