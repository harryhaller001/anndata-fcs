import io

import flowio
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from anndata_fcs import FCSConversionWarning, _convert, anndata_to_fcs, fcs_to_anndata, write_fcs
from anndata_fcs._schema import KW_OBS, KW_SCHEMA, STRUCTURAL_KEYWORDS


def test_fcs_to_anndata_float(lsr_ii_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(lsr_ii_fdata)

    assert adata.shape == (lsr_ii_fdata.event_count, lsr_ii_fdata.channel_count)
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X, lsr_ii_fdata.as_array(preprocess=False))
    np.testing.assert_array_equal(adata.raw.X, adata.X)
    assert adata.raw.X is not adata.X
    assert list(adata.raw.var_names) == list(adata.var_names)
    pd.testing.assert_frame_equal(adata.raw.var, adata.var)

    assert list(adata.var_names) == lsr_ii_fdata.pnn_labels
    assert len(adata.var.columns) == 0

    meta = adata.uns["fcs"]
    assert meta["version"] == "3.0"
    assert meta["preprocessed"] is False
    assert meta["text"]["cyt"] == "LSRII"
    assert meta["text"]["spill"].startswith("4,FITC-A")
    assert meta["text"]["tube name"] == "D6"
    # Per-channel keywords keep the names parsed by FlowIO
    assert meta["text"]["p1n"] == "FSC-A"
    assert meta["text"]["p1r"] == "262144"
    assert meta["text"]["p1v"] == "611"
    assert meta["text"]["p1display"] == "LOG"
    assert not STRUCTURAL_KEYWORDS & set(meta["text"])
    assert "p1b" not in meta["text"]
    assert meta["source"]["p1b"] == "32"
    assert meta["source"]["datatype"] == "F"
    assert meta["source"]["tot"] == str(lsr_ii_fdata.event_count)
    assert meta["filename"] == "HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs"


def test_fcs_to_anndata_int(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata)

    assert adata.X.dtype == np.uint16
    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=False))
    assert adata.uns["fcs"]["text"]["p2s"] == "CD4 FITC"
    assert adata.uns["fcs"]["text"]["p3s"] == "CD8/PE"
    assert adata.uns["fcs"]["text"]["p2e"] == "4,1"
    assert adata.uns["fcs"]["analysis"] == {"gate1": "lymphocytes"}
    assert adata.uns["fcs"]["source"]["datatype"] == "I"
    assert [adata.uns["fcs"]["source"][f"p{n}b"] for n in range(1, 5)] == ["16"] * 4
    # Keywords containing '/' are escaped, so the object can be written to h5ad
    assert adata.uns["fcs"]["text"]["custom%2Fkey"] == "custom value"


def test_fcs_to_anndata_preprocess(int_fdata: flowio.FlowData) -> None:
    """Preprocessing only modifies X. raw, var and uns describe the event data as stored in the file."""
    unprocessed = fcs_to_anndata(int_fdata)
    adata = fcs_to_anndata(int_fdata, preprocess=True)

    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=True))
    np.testing.assert_array_equal(adata.raw.X, int_fdata.as_array(preprocess=False))
    assert adata.raw.X.dtype == np.uint16
    assert adata.uns["fcs"]["preprocessed"] is True

    pd.testing.assert_frame_equal(adata.var, unprocessed.var)
    pd.testing.assert_frame_equal(adata.raw.var, unprocessed.raw.var)
    assert adata.uns["fcs"]["text"] == unprocessed.uns["fcs"]["text"]
    assert adata.uns["fcs"]["source"] == unprocessed.uns["fcs"]["source"]


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


def test_duplicate_channels_keep_keywords() -> None:
    """Duplicate $PnN are made unique on read and each duplicate keeps its per-channel keywords on write."""
    handle = io.BytesIO()
    flowio.create_fcs(handle, np.zeros(6, dtype=np.float32), ["A", "A", "B"], opt_channel_names=["CD3", "CD4", "CD8"])
    handle.seek(0)
    with pytest.warns(FCSConversionWarning, match="not unique"):
        adata = fcs_to_anndata(flowio.FlowData(handle))
    assert list(adata.var_names) == ["A", "A-1", "B"]

    fdata = anndata_to_fcs(adata, index_channel=None, strict=True)
    assert fdata.pnn_labels == ["A", "A-1", "B"]
    assert fdata.pns_labels == ["CD3", "CD4", "CD8"]

    reordered = adata[:, ["B", "A-1"]].copy()
    reordered.raw = None
    assert anndata_to_fcs(reordered, index_channel=None).pns_labels == ["CD8", "CD4"]


def _with_index_values(fdata: flowio.FlowData, values: np.ndarray) -> flowio.FlowData:
    """Copy of an FCS file written by anndata_to_fcs with modified values of the event index channel."""
    events = fdata.as_array(preprocess=False)[: len(values)]
    events[:, fdata.pnn_labels.index("barcode_rank")] = values
    metadata = {key: value for key, value in fdata.text.items() if key not in STRUCTURAL_KEYWORDS}
    handle = io.BytesIO()
    flowio.create_fcs(handle, events.ravel(), fdata.pnn_labels, metadata_dict=metadata)
    handle.seek(0)
    return flowio.FlowData(handle)


@pytest.mark.parametrize("values", [[0.5, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 0.0, 1.0]])
def test_fcs_to_anndata_invalid_event_index(adata_annotated: AnnData, values: list[float]) -> None:
    """A modified index channel is not used, since truncating it would map events to wrong observations."""
    fdata = _with_index_values(anndata_to_fcs(adata_annotated), np.array(values))
    with pytest.warns(FCSConversionWarning) as record:
        restored = fcs_to_anndata(fdata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "does not hold valid event positions" in messages
    assert "Observation metadata is not restored" in messages
    assert "barcode_rank" not in restored.var_names
    assert len(restored.obs.columns) == 0


def test_fcs_to_anndata_event_index_out_of_range(adata_annotated: AnnData) -> None:
    fdata = _with_index_values(anndata_to_fcs(adata_annotated), np.array([0.0, 1.0, 5000.0]))
    with pytest.warns(FCSConversionWarning) as record:
        restored = fcs_to_anndata(fdata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "refers to more observations than stored (1000)" in messages
    assert "Observation metadata is not restored" in messages
    assert len(restored.obs.columns) == 0


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

    adata.uns["fcs"] = {"text": {"p1n": adata.var_names[0], "p1s": "label"}}
    fdata = anndata_to_fcs(adata, index_channel=None)
    assert fdata.pns_labels == ["label"] + [""] * (adata.n_vars - 1)

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
    adata.uns["fcs"]["text"]["p2e"] = "0,0"

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


def test_anndata_to_fcs_strict_keywords() -> None:
    """Keywords that are not written are reported by strict=True like any other information loss."""
    adata = AnnData(np.zeros((2, 1), dtype=np.float32))
    adata.var_names = ["A"]
    adata.uns["fcs"] = {"text": {"com": ""}}
    with pytest.warns(FCSConversionWarning, match="empty value"):
        anndata_to_fcs(adata)
    with pytest.raises(ValueError, match="empty value"):
        anndata_to_fcs(adata, strict=True)


@pytest.mark.parametrize(
    ("keyword", "value", "match"), [("p2e", "4", r"'FL1-H'.*Invalid \$PnE"), ("p2r", "max", r"'FL1-H'.*Invalid \$PnR")]
)
def test_anndata_to_fcs_invalid_channel_keywords(
    int_fdata: flowio.FlowData, keyword: str, value: str, match: str
) -> None:
    adata = fcs_to_anndata(int_fdata)
    del adata.uns["fcs"]["analysis"]
    adata.uns["fcs"]["text"][keyword] = value
    with pytest.raises(ValueError, match=match):
        anndata_to_fcs(adata)


def test_anndata_to_fcs_store_obs(adata_annotated: AnnData) -> None:
    fdata = anndata_to_fcs(adata_annotated, store_obs=False)
    assert KW_OBS not in fdata.text

    restored = fcs_to_anndata(fdata)
    assert list(restored.obs_names) == [str(i) for i in range(adata_annotated.n_obs)]
    assert len(restored.obs.columns) == 0


def test_anndata_to_fcs_originality(int_fdata: flowio.FlowData) -> None:
    """Written files are derived copies, which FCS 3.1 marks with $ORIGINALITY and $LAST_MODIFIED."""
    adata = fcs_to_anndata(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    adata.uns["fcs"]["text"]["originality"] = "Original"
    adata.uns["fcs"]["text"]["last_modified"] = "01-JAN-2000 00:00:00.00"

    fdata = anndata_to_fcs(adata)
    assert fdata.text["originality"] == "DataModified"
    assert fdata.text["last_modified"] != "01-JAN-2000 00:00:00.00"

    handle = io.BytesIO()
    write_fcs(adata, handle)
    assert b"$ORIGINALITY/DataModified/" in handle.getvalue()


def test_anndata_to_fcs_spillover(lsr_ii_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(lsr_ii_fdata)
    spillover = "2,FITC-A,AmCyan-A,1,0.02,0.1,1"
    adata.uns["fcs"]["text"]["spillover"] = spillover

    assert anndata_to_fcs(adata).text["spillover"] == spillover

    # $SPILLOVER refers to $PnN values, so removed channels are removed from the matrix
    subset = adata[:, [name for name in adata.var_names if name != "AmCyan-A"]].copy()
    subset.raw = None
    with pytest.warns(FCSConversionWarning, match="AmCyan-A"):
        fdata = anndata_to_fcs(subset)
    assert fdata.text["spillover"] == "1,FITC-A,1"
    with pytest.raises(ValueError, match="AmCyan-A"):
        anndata_to_fcs(subset, strict=True)


def test_anndata_to_fcs_spill(lsr_ii_fdata: flowio.FlowData) -> None:
    """$SPILL (FCS 3.0, BD FACSDiva) is restricted to the written channels like $SPILLOVER."""
    adata = fcs_to_anndata(lsr_ii_fdata)
    assert adata.uns["fcs"]["text"]["spill"].startswith("4,FITC-A,PerCP-Cy5-5-A,AmCyan-A,PE-TxRed YG-A,1,0,")

    subset = adata[:, ["FSC-A", "SSC-A", "FITC-A"]].copy()
    subset.raw = None
    with pytest.warns(FCSConversionWarning, match=r"\$SPILL refers to channels not present in var_names \(PerCP"):
        fdata = anndata_to_fcs(subset, index_channel=None)
    assert fdata.text["spill"] == "1,FITC-A,1"
    with pytest.raises(ValueError, match=r"\$SPILL refers"):
        anndata_to_fcs(subset, strict=True)


def test_anndata_to_fcs_comma_in_channel_name() -> None:
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["CD3,CD4", "B"]
    with pytest.warns(FCSConversionWarning, match="commas"):
        fdata = anndata_to_fcs(adata)
    assert fdata.pnn_labels[0] == "CD3,CD4"
    with pytest.raises(ValueError, match="commas"):
        anndata_to_fcs(adata, strict=True)


def test_anndata_to_fcs_channel_keywords_follow_channels(int_fdata: flowio.FlowData) -> None:
    """Per-channel keywords are assigned to the written channels by $PnN, so subsetting and reordering keeps them."""
    adata = fcs_to_anndata(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    subset = adata[:, ["FL2-H", "FL1-H"]].copy()
    subset.raw = None

    fdata = anndata_to_fcs(subset, index_channel=None)
    assert fdata.pnn_labels == ["FL2-H", "FL1-H"]
    assert fdata.pns_labels == ["CD8/PE", "CD4 FITC"]
    assert fdata.text["p1g"] == "2.0"
    assert fdata.text["p2v"] == "650"
    assert "p3n" not in fdata.text and "p4g" not in fdata.text


def test_anndata_to_fcs_unassigned_channel_keyword() -> None:
    """Per-channel keywords without a matching $PnN would be attributed to another channel and are not written."""
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["A", "B"]
    adata.uns["fcs"] = {"text": {"p1n": "A", "p1s": "CD3", "p2n": "C", "p2s": "CD4", "p3s": "orphan"}}

    with pytest.warns(FCSConversionWarning) as record:
        fdata = anndata_to_fcs(adata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "'p3s' refers to channel 3" in messages
    # Channel C is replaced by B, so the keywords of C are not assigned to B
    assert "without a matching $PnN keyword" in messages
    assert fdata.pns_labels == ["CD3", "", ""]
    assert "p3s" not in fdata.text


def test_anndata_to_fcs_renamed_channels(int_fdata: flowio.FlowData) -> None:
    """Renamed channels lose their per-channel keywords, which can mislabel log scaled event data."""
    adata = fcs_to_anndata(int_fdata)
    del adata.uns["fcs"]["analysis"]
    adata.raw = None
    adata.var_names = ["FSC-A", "CD4", "FL2-H", "Time"]

    with pytest.warns(FCSConversionWarning, match=r"var_names without a matching \$PnN keyword .*: CD4\."):
        fdata = anndata_to_fcs(adata, index_channel=None)
    assert fdata.pns_labels[1] == ""
    with pytest.raises(ValueError, match="CD4"):
        anndata_to_fcs(adata, strict=True)


def test_anndata_to_fcs_added_channel(int_fdata: flowio.FlowData) -> None:
    """Channels added to the AnnData object have no keywords to lose and do not trigger a warning."""
    adata = fcs_to_anndata(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    adata.raw = None
    extended = AnnData(
        X=np.column_stack([adata.X, adata.X[:, 0]]),
        var=pd.DataFrame(index=[*adata.var_names, "ratio"]),
        uns=adata.uns,
    )

    fdata = anndata_to_fcs(extended, index_channel=None, strict=True)
    assert fdata.pnn_labels == ["FSC-A", "FL1-H", "FL2-H", "Time", "ratio"]
    assert fdata.pns_labels[1] == "CD4 FITC"


def test_anndata_to_fcs_unwritten_var_columns(lsr_ii_fdata: flowio.FlowData) -> None:
    """var columns of the slot that is not written are reported."""
    adata = fcs_to_anndata(lsr_ii_fdata)
    adata.var["highly_variable"] = True
    with pytest.warns(FCSConversionWarning, match="var columns are not written, .*: highly_variable"):
        anndata_to_fcs(adata)
    with pytest.raises(ValueError, match="highly_variable"):
        anndata_to_fcs(adata, strict=True)
    # Written from X, var is stored
    restored = fcs_to_anndata(anndata_to_fcs(adata, use_raw=False, strict=True))
    assert list(restored.var["highly_variable"]) == [True] * adata.n_vars

    adata = fcs_to_anndata(lsr_ii_fdata)
    adata.raw.var["detector"] = "PMT"
    with pytest.warns(FCSConversionWarning, match="raw.var columns are not written, .*: detector"):
        anndata_to_fcs(adata, use_raw=False)

    # Identical columns in var and raw.var are stored once without a warning
    adata = fcs_to_anndata(lsr_ii_fdata)
    adata.var["detector"] = "PMT"
    adata.raw = adata.copy()
    for use_raw in (True, False):
        anndata_to_fcs(adata, use_raw=use_raw, strict=True)


def test_anndata_to_fcs_var_columns(adata: AnnData) -> None:
    """var columns are stored as payload, also if they look like keywords."""
    adata.var["pns"] = "label"
    fdata = anndata_to_fcs(adata, index_channel=None)
    assert not any(fdata.pns_labels)

    restored = fcs_to_anndata(fdata)
    assert list(restored.var["pns"]) == ["label"] * adata.n_vars


def test_anndata_to_fcs_use_raw(lsr_ii_fdata: flowio.FlowData) -> None:
    """raw is written by default. X is written with use_raw=False or if there is no raw."""
    adata = fcs_to_anndata(lsr_ii_fdata)
    subset = adata[:, ["FSC-A", "SSC-A"]].copy()
    subset.X = subset.X * 2

    with pytest.warns(FCSConversionWarning, match="X differs from raw"):
        fdata = anndata_to_fcs(subset, index_channel=None)
    assert fdata.pnn_labels == lsr_ii_fdata.pnn_labels
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), lsr_ii_fdata.as_array(preprocess=False))

    with pytest.warns(FCSConversionWarning) as record:
        fdata = anndata_to_fcs(subset, index_channel=None, use_raw=False)
    messages = " ".join(str(warning.message) for warning in record)
    # $SPILL only refers to fluorescence channels, which are not written
    assert "raw differs from X" in messages and "$SPILL refers to channels" in messages
    assert fdata.pnn_labels == ["FSC-A", "SSC-A"]
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), subset.X)

    with pytest.raises(ValueError, match="raw differs from X"):
        anndata_to_fcs(subset, use_raw=False, strict=True)

    subset.raw = None
    with pytest.warns(FCSConversionWarning, match=r"\$SPILL refers to channels"):
        fdata = anndata_to_fcs(subset, index_channel=None)
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), subset.X)
    with pytest.raises(ValueError, match="no raw"):
        anndata_to_fcs(subset, use_raw=True)


def test_anndata_to_fcs_use_raw_same_events(int_fdata: flowio.FlowData) -> None:
    """No warning about dropping X or raw if both hold the same event data."""
    adata = fcs_to_anndata(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]

    for use_raw in (True, False):
        anndata_to_fcs(adata, use_raw=use_raw, strict=True)


def test_anndata_to_fcs_preprocessed_x(int_fdata: flowio.FlowData) -> None:
    """Keywords are adjusted to describe preprocessed X, so preprocessing the written file does not change it."""
    adata = fcs_to_anndata(int_fdata, preprocess=True)
    del adata.uns["fcs"]["analysis"]

    with pytest.warns(FCSConversionWarning) as record:
        fdata = anndata_to_fcs(adata, use_raw=False, index_channel=None)
    messages = " ".join(str(warning.message) for warning in record)
    assert "32-bit float" in messages and "raw differs from X" in messages

    np.testing.assert_allclose(fdata.as_array(preprocess=False), adata.X, rtol=1e-6)
    np.testing.assert_allclose(fdata.as_array(preprocess=True), adata.X, rtol=1e-6)
    assert fdata.text["p2e"] == "0,0"
    # $PnR of the log scaled channel ($PnE/4,1/) is the upper bound of the linear scale values
    assert fdata.text["p2r"] == "10000"
    assert fdata.text["p1r"] == "1024"
    assert fdata.text["p3g"] == "1.0"
    # Gain 2.0 halves the values, $TIMESTEP/0.01/ scales the time channel (its gain is ignored, like in FlowIO)
    assert fdata.text["p3r"] == "512"
    assert fdata.text["p4r"] == "11"
    assert fdata.as_array(preprocess=False)[:, 2].max() <= 512
    assert fdata.as_array(preprocess=False)[:, 3].max() <= 11
    assert fdata.text["timestep"] == "1"
    # The source object is not modified
    assert adata.uns["fcs"]["text"]["p2e"] == "4,1"
    assert adata.uns["fcs"]["text"]["timestep"] == "0.01"


@pytest.mark.parametrize("chunk_rows", [7, 65_536])
@pytest.mark.parametrize("layout", ["dense", "csr", "csc"])
def test_anndata_to_fcs_event_blocks(
    int_fdata: flowio.FlowData, monkeypatch: pytest.MonkeyPatch, chunk_rows: int, layout: str
) -> None:
    """Events are converted block-wise: results do not depend on the block size or the matrix layout."""
    monkeypatch.setattr(_convert, "_CHUNK_ROWS", chunk_rows)
    adata = fcs_to_anndata(int_fdata)
    del adata.uns["fcs"]["analysis"]
    expected = int_fdata.as_array(preprocess=True)
    if layout != "dense":
        adata.raw = None
        adata.X = getattr(sparse, f"{layout}_matrix")(adata.X)

    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        fdata = anndata_to_fcs(adata, index_channel=None)
    # The log scaled channel FL1-H is converted to linear scale
    np.testing.assert_allclose(fdata.as_array(preprocess=True), expected, rtol=1e-6)


def test_anndata_to_fcs_precision_by_dtype() -> None:
    """Only dtypes which are not exactly representable as float32 are checked and reported."""
    for dtype in (np.int16, np.uint16, np.float16, np.float32, np.bool_):
        anndata_to_fcs(AnnData(np.ones((3, 2), dtype=dtype)), strict=True)
    anndata_to_fcs(AnnData(np.full((3, 2), 2**24, dtype=np.int64)), strict=True)
    with pytest.raises(ValueError, match="32-bit float"):
        anndata_to_fcs(AnnData(np.full((3, 2), 2**24 + 1, dtype=np.int64)), strict=True)
    with pytest.raises(ValueError, match="32-bit float"):
        anndata_to_fcs(AnnData(np.array([[np.nan, 0.1]], dtype=np.float64)), strict=True)
    # NaN and inf are exactly representable
    anndata_to_fcs(AnnData(np.array([[np.nan, np.inf]], dtype=np.float64)), strict=True)


def test_anndata_to_fcs_same_events_sparse(int_fdata: flowio.FlowData) -> None:
    """X and raw are compared block-wise, also if one of them is sparse."""
    adata = fcs_to_anndata(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    adata.X = sparse.csr_matrix(adata.X)
    anndata_to_fcs(adata, strict=True)

    adata.X[0, 0] += 1
    with pytest.raises(ValueError, match="X differs from raw"):
        anndata_to_fcs(adata, strict=True)


def test_fcs_to_anndata_without_raw(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata, raw=False)
    assert adata.raw is None
    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=False))

    preprocessed = fcs_to_anndata(int_fdata, preprocess=True, raw=False)
    assert preprocessed.raw is None
    del preprocessed.uns["fcs"]["analysis"]
    # X is written, with keywords describing the preprocessed values
    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        fdata = anndata_to_fcs(preprocessed, index_channel=None)
    np.testing.assert_allclose(fdata.as_array(preprocess=True), preprocessed.X, rtol=1e-6)


def test_fcs_to_anndata_does_not_share_memory(lsr_ii_fdata: flowio.FlowData) -> None:
    before = np.array(lsr_ii_fdata.events)
    adata = fcs_to_anndata(lsr_ii_fdata)
    adata.X[:] = -1
    adata.raw.X[:] = -1
    np.testing.assert_array_equal(np.array(lsr_ii_fdata.events), before)


@pytest.mark.parametrize("name", ["/A", "A/"])
def test_anndata_to_fcs_delimiter_in_channel_name(name: str) -> None:
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = [name, "B"]
    with pytest.raises(ValueError, match="must not start or end with '/'"):
        anndata_to_fcs(adata)
    with pytest.raises(ValueError, match="must not start or end with '/'"):
        anndata_to_fcs(AnnData(np.zeros((2, 1), dtype=np.float32)), index_channel=name)


def test_anndata_to_fcs_dollar_in_channel_name() -> None:
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["CD3$", "B"]
    with pytest.warns(FCSConversionWarning, match=r"contain '\$'.*CD3\$"):
        anndata_to_fcs(adata, index_channel=None)


def test_anndata_to_fcs_text_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The size of the whole TEXT segment is checked, not only of the AnnData payload."""
    adata = AnnData(np.zeros((2, 1), dtype=np.float32))
    adata.uns["fcs"] = {"text": {"com": "x" * 5000}}
    size = len(anndata_to_fcs(adata).text["com"])
    assert size == 5000

    monkeypatch.setattr(_convert, "TEXT_SIZE_WARNING_BYTES", 4000)
    with pytest.warns(FCSConversionWarning, match="TEXT segment is large"):
        anndata_to_fcs(adata)

    monkeypatch.setattr(_convert, "MAX_TEXT_BYTES", 4000)
    with pytest.raises(ValueError, match="exceeds the maximum size"):
        anndata_to_fcs(adata)


def test_text_size_bound(adata_annotated: AnnData) -> None:
    """The estimated TEXT size is an upper bound of the TEXT segment written by FlowIO."""
    sizes: list[int] = []
    original = _convert._check_text_size

    def record(metadata, channel_names, pns_labels):
        original(metadata, channel_names, pns_labels)
        pairs = [*metadata.items(), *(("pnn", n) for n in channel_names), *(("pns", s) for s in pns_labels)]
        sizes.append(512 + 64 * len(channel_names) + sum(len(k) + len(v.encode()) + v.count("/") + 3 for k, v in pairs))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_convert, "_check_text_size", record)
        handle = io.BytesIO()
        write_fcs(adata_annotated, handle)
    header = handle.getvalue()[:58].decode()
    text_size = int(header[18:26]) - int(header[10:18]) + 1
    assert text_size <= sizes[0] < text_size + 512 + 64 * (adata_annotated.n_vars + 1)
