import io

import flowio
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from anndata_fcs import FCSConversionWarning, _convert, read_fcs, write_fcs
from anndata_fcs._schema import KW_OBS, KW_SCHEMA, STRUCTURAL_KEYWORDS
from test.conftest import to_flowdata


def test_read_fcs_float(lsr_ii_fdata: flowio.FlowData) -> None:
    adata = read_fcs(lsr_ii_fdata)

    assert adata.shape == (lsr_ii_fdata.event_count, lsr_ii_fdata.channel_count)
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X, lsr_ii_fdata.as_array(preprocess=False))
    assert adata.raw is None
    assert "preprocessed" not in adata.layers

    assert list(adata.var_names) == lsr_ii_fdata.pnn_labels
    assert len(adata.var.columns) == 0

    meta = adata.uns["fcs"]
    assert meta["version"] == "3.0"
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
    assert "source" not in meta
    assert meta["filename"] == "HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs"


def test_read_fcs_int(int_fdata: flowio.FlowData) -> None:
    adata = read_fcs(int_fdata)

    assert adata.X.dtype == np.uint16
    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=False))
    assert adata.uns["fcs"]["text"]["p2s"] == "CD4 FITC"
    assert adata.uns["fcs"]["text"]["p3s"] == "CD8/PE"
    assert adata.uns["fcs"]["text"]["p2e"] == "4,1"
    assert adata.uns["fcs"]["analysis"] == {"gate1": "lymphocytes"}
    # Keywords containing '/' are escaped, so the object can be written to h5ad
    assert adata.uns["fcs"]["text"]["custom%2Fkey"] == "custom value"


def test_read_fcs_preprocess(int_fdata: flowio.FlowData) -> None:
    """Preprocessed values are stored in a layer. X, var and uns describe the event data as stored in the file."""
    unprocessed = read_fcs(int_fdata)
    adata = read_fcs(int_fdata, preprocess=True)

    np.testing.assert_array_equal(adata.layers["preprocessed"], int_fdata.as_array(preprocess=True))
    np.testing.assert_array_equal(adata.X, int_fdata.as_array(preprocess=False))
    assert adata.X.dtype == np.uint16

    pd.testing.assert_frame_equal(adata.var, unprocessed.var)
    assert adata.uns["fcs"] == unprocessed.uns["fcs"]


def test_read_fcs_without_events(int_fcs_bytes: bytes) -> None:
    fdata = flowio.FlowData(io.BytesIO(int_fcs_bytes), only_text=True)
    with pytest.raises(ValueError, match="no event data"):
        read_fcs(fdata)


def test_read_fcs_duplicate_channels() -> None:
    handle = io.BytesIO()
    flowio.create_fcs(handle, np.zeros(4, dtype=np.float32), ["A", "A"])
    handle.seek(0)
    with pytest.warns(FCSConversionWarning, match="not unique"):
        adata = read_fcs(flowio.FlowData(handle))
    assert adata.var_names.is_unique


def test_duplicate_channels_keep_keywords() -> None:
    """Duplicate $PnN are made unique on read and each duplicate keeps its per-channel keywords on write."""
    handle = io.BytesIO()
    flowio.create_fcs(handle, np.zeros(6, dtype=np.float32), ["A", "A", "B"], opt_channel_names=["CD3", "CD4", "CD8"])
    handle.seek(0)
    with pytest.warns(FCSConversionWarning, match="not unique"):
        adata = read_fcs(flowio.FlowData(handle))
    assert list(adata.var_names) == ["A", "A-1", "B"]

    fdata = to_flowdata(adata, index_channel=None)
    assert fdata.pnn_labels == ["A", "A-1", "B"]
    assert fdata.pns_labels == ["CD3", "CD4", "CD8"]

    reordered = adata[:, ["B", "A-1"]].copy()
    assert to_flowdata(reordered, index_channel=None).pns_labels == ["CD8", "CD4"]


def _with_index_values(fdata: flowio.FlowData, values: np.ndarray) -> flowio.FlowData:
    """Copy of an FCS file written by write_fcs with modified values of the event index channel."""
    events = fdata.as_array(preprocess=False)[: len(values)]
    events[:, fdata.pnn_labels.index("barcode_rank")] = values
    metadata = {key: value for key, value in fdata.text.items() if key not in STRUCTURAL_KEYWORDS}
    handle = io.BytesIO()
    flowio.create_fcs(handle, events.ravel(), fdata.pnn_labels, metadata_dict=metadata)
    handle.seek(0)
    return flowio.FlowData(handle)


@pytest.mark.parametrize("values", [[0.5, 1.0, 2.0], [-1.0, 0.0, 1.0], [np.nan, 0.0, 1.0]])
def test_read_fcs_invalid_event_index(adata_annotated: AnnData, values: list[float]) -> None:
    """A modified index channel is not used, since truncating it would map events to wrong observations."""
    fdata = _with_index_values(to_flowdata(adata_annotated), np.array(values))
    with pytest.warns(FCSConversionWarning) as record:
        restored = read_fcs(fdata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "does not hold valid event positions" in messages
    assert "Observation metadata is not restored" in messages
    assert "barcode_rank" not in restored.var_names
    assert len(restored.obs.columns) == 0


def test_read_fcs_event_index_out_of_range(adata_annotated: AnnData) -> None:
    fdata = _with_index_values(to_flowdata(adata_annotated), np.array([0.0, 1.0, 5000.0]))
    with pytest.warns(FCSConversionWarning) as record:
        restored = read_fcs(fdata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "refers to more observations than stored (1000)" in messages
    assert "Observation metadata is not restored" in messages
    assert len(restored.obs.columns) == 0


def test_to_flowdata(adata_annotated: AnnData) -> None:
    fdata = to_flowdata(adata_annotated)

    assert isinstance(fdata, flowio.FlowData)
    assert fdata.channel_count == adata_annotated.n_vars + 1
    assert fdata.pnn_labels == [*adata_annotated.var_names, "barcode_rank"]

    events = fdata.as_array(preprocess=False)
    np.testing.assert_array_equal(events[:, :-1], adata_annotated.X.toarray())
    np.testing.assert_array_equal(events[:, -1], np.arange(adata_annotated.n_obs))

    assert fdata.text[KW_SCHEMA] == "1"
    assert KW_OBS in fdata.text


def test_write_fcs_does_not_modify_input(adata_annotated: AnnData) -> None:
    before = adata_annotated.copy()
    to_flowdata(adata_annotated)

    pd.testing.assert_frame_equal(adata_annotated.obs, before.obs)
    pd.testing.assert_frame_equal(adata_annotated.var, before.var)
    assert adata_annotated.uns.keys() == before.uns.keys()


def test_write_fcs_index_channel(adata: AnnData) -> None:
    fdata = to_flowdata(adata, index_channel=None)
    assert fdata.pnn_labels == list(adata.var_names)

    fdata = to_flowdata(adata, index_channel="event_id")
    assert fdata.pnn_labels[-1] == "event_id"

    with pytest.raises(ValueError, match="already exists"):
        to_flowdata(adata, index_channel=adata.var_names[0])


def test_write_fcs_pns(adata: AnnData) -> None:
    fdata = to_flowdata(adata, pns_key="gene_ids", index_channel=None)
    assert fdata.pns_labels == list(adata.var["gene_ids"])

    adata.uns["fcs"] = {"text": {"p1n": adata.var_names[0], "p1s": "label"}}
    fdata = to_flowdata(adata, index_channel=None)
    assert fdata.pns_labels == ["label"] + [""] * (adata.n_vars - 1)

    with pytest.raises(KeyError):
        to_flowdata(adata, pns_key="missing")


def test_write_fcs_range() -> None:
    adata = AnnData(np.array([[1.0, 500_000.5], [2.0, 3.0]], dtype=np.float32))
    fdata = to_flowdata(adata, index_channel=None)
    assert fdata.text["p1r"] == "262144"
    assert fdata.text["p2r"] == "500001"


def test_write_fcs_precision_loss() -> None:
    adata = AnnData(np.array([[0.1, 0.2]], dtype=np.float64))
    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        to_flowdata(adata)


def test_write_fcs_unsupported_slots(adata: AnnData) -> None:
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["X_umap"] = np.zeros((adata.n_obs, 2))
    with pytest.warns(FCSConversionWarning, match="layers, obsm"):
        to_flowdata(adata)


def test_write_fcs_unsupported_metadata(adata: AnnData) -> None:
    adata.obs["time"] = pd.Timestamp("2020-01-01")
    adata.uns["frame"] = pd.DataFrame({"a": [1]})

    with pytest.warns(FCSConversionWarning) as record:
        fdata = to_flowdata(adata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "time" in messages and "uns['frame']" in messages

    restored = read_fcs(fdata)
    assert "time" not in restored.obs.columns
    assert "frame" not in restored.uns


def test_write_fcs_analysis(int_fdata: flowio.FlowData) -> None:
    adata = read_fcs(int_fdata)
    # Linear scale keeps the integer data exact in float32, so only the ANALYSIS segment is lost
    adata.uns["fcs"]["text"]["p2e"] = "0,0"

    with pytest.warns(FCSConversionWarning, match="ANALYSIS"):
        to_flowdata(adata)


def test_write_fcs_invalid_input() -> None:
    with pytest.raises(TypeError, match="numeric"):
        to_flowdata(AnnData(np.array([["a"]], dtype=object)))

    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["A", "A"]
    with pytest.raises(ValueError, match="unique"):
        to_flowdata(adata)


def test_write_fcs_unwritten_keywords() -> None:
    """Keywords that are not written are reported like any other information loss."""
    adata = AnnData(np.zeros((2, 1), dtype=np.float32))
    adata.var_names = ["A"]
    adata.uns["fcs"] = {"text": {"com": ""}}
    with pytest.warns(FCSConversionWarning, match="empty value"):
        to_flowdata(adata)


@pytest.mark.parametrize(
    ("keyword", "value", "match"), [("p2e", "4", r"'FL1-H'.*Invalid \$PnE"), ("p2r", "max", r"'FL1-H'.*Invalid \$PnR")]
)
def test_write_fcs_invalid_channel_keywords(int_fdata: flowio.FlowData, keyword: str, value: str, match: str) -> None:
    adata = read_fcs(int_fdata)
    del adata.uns["fcs"]["analysis"]
    adata.uns["fcs"]["text"][keyword] = value
    with pytest.raises(ValueError, match=match):
        to_flowdata(adata)


def test_write_fcs_store_obs(adata_annotated: AnnData) -> None:
    fdata = to_flowdata(adata_annotated, store_obs=False)
    assert KW_OBS not in fdata.text

    restored = read_fcs(fdata)
    assert list(restored.obs_names) == [str(i) for i in range(adata_annotated.n_obs)]
    assert len(restored.obs.columns) == 0


def test_write_fcs_originality(int_fdata: flowio.FlowData) -> None:
    """Written files are derived copies, which FCS 3.1 marks with $ORIGINALITY and $LAST_MODIFIED."""
    adata = read_fcs(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    adata.uns["fcs"]["text"]["originality"] = "Original"
    adata.uns["fcs"]["text"]["last_modified"] = "01-JAN-2000 00:00:00.00"

    fdata = to_flowdata(adata)
    assert fdata.text["originality"] == "DataModified"
    assert fdata.text["last_modified"] != "01-JAN-2000 00:00:00.00"

    handle = io.BytesIO()
    write_fcs(adata, handle)
    assert b"$ORIGINALITY/DataModified/" in handle.getvalue()


def test_write_fcs_spillover(lsr_ii_fdata: flowio.FlowData) -> None:
    adata = read_fcs(lsr_ii_fdata)
    spillover = "2,FITC-A,AmCyan-A,1,0.02,0.1,1"
    adata.uns["fcs"]["text"]["spillover"] = spillover

    assert to_flowdata(adata).text["spillover"] == spillover

    # $SPILLOVER refers to $PnN values, so removed channels are removed from the matrix
    subset = adata[:, [name for name in adata.var_names if name != "AmCyan-A"]].copy()
    with pytest.warns(FCSConversionWarning, match="AmCyan-A"):
        fdata = to_flowdata(subset)
    assert fdata.text["spillover"] == "1,FITC-A,1"


def test_write_fcs_spill(lsr_ii_fdata: flowio.FlowData) -> None:
    """$SPILL (FCS 3.0, BD FACSDiva) is restricted to the written channels like $SPILLOVER."""
    adata = read_fcs(lsr_ii_fdata)
    assert adata.uns["fcs"]["text"]["spill"].startswith("4,FITC-A,PerCP-Cy5-5-A,AmCyan-A,PE-TxRed YG-A,1,0,")

    subset = adata[:, ["FSC-A", "SSC-A", "FITC-A"]].copy()
    with pytest.warns(FCSConversionWarning, match=r"\$SPILL refers to channels not present in var_names \(PerCP"):
        fdata = to_flowdata(subset, index_channel=None)
    assert fdata.text["spill"] == "1,FITC-A,1"


def test_write_fcs_comma_in_channel_name() -> None:
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["CD3,CD4", "B"]
    with pytest.warns(FCSConversionWarning, match="commas"):
        fdata = to_flowdata(adata)
    assert fdata.pnn_labels[0] == "CD3,CD4"


def test_write_fcs_channel_keywords_follow_channels(int_fdata: flowio.FlowData) -> None:
    """Per-channel keywords are assigned to the written channels by $PnN, so subsetting and reordering keeps them."""
    adata = read_fcs(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    subset = adata[:, ["FL2-H", "FL1-H"]].copy()

    fdata = to_flowdata(subset, index_channel=None)
    assert fdata.pnn_labels == ["FL2-H", "FL1-H"]
    assert fdata.pns_labels == ["CD8/PE", "CD4 FITC"]
    assert fdata.text["p1g"] == "2.0"
    assert fdata.text["p2v"] == "650"
    assert "p3n" not in fdata.text and "p4g" not in fdata.text


def test_write_fcs_unassigned_channel_keyword() -> None:
    """Per-channel keywords without a matching $PnN would be attributed to another channel and are not written."""
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["A", "B"]
    adata.uns["fcs"] = {"text": {"p1n": "A", "p1s": "CD3", "p2n": "C", "p2s": "CD4", "p3s": "orphan"}}

    with pytest.warns(FCSConversionWarning) as record:
        fdata = to_flowdata(adata)
    messages = " ".join(str(warning.message) for warning in record)
    assert "'p3s' refers to channel 3" in messages
    # Channel C is replaced by B, so the keywords of C are not assigned to B
    assert "without a matching $PnN keyword" in messages
    assert fdata.pns_labels == ["CD3", "", ""]
    assert "p3s" not in fdata.text


def test_write_fcs_renamed_channels(int_fdata: flowio.FlowData) -> None:
    """Renamed channels lose their per-channel keywords, which can mislabel log scaled event data."""
    adata = read_fcs(int_fdata)
    del adata.uns["fcs"]["analysis"]
    adata.var_names = ["FSC-A", "CD4", "FL2-H", "Time"]

    with pytest.warns(FCSConversionWarning, match=r"var_names without a matching \$PnN keyword .*: CD4\."):
        fdata = to_flowdata(adata, index_channel=None)
    assert fdata.pns_labels[1] == ""


def test_write_fcs_added_channel(int_fdata: flowio.FlowData) -> None:
    """Channels added to the AnnData object have no keywords to lose and do not trigger a warning."""
    adata = read_fcs(int_fdata)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]
    extended = AnnData(
        X=np.column_stack([adata.X, adata.X[:, 0]]),
        var=pd.DataFrame(index=[*adata.var_names, "ratio"]),
        uns=adata.uns,
    )

    fdata = to_flowdata(extended, index_channel=None)
    assert fdata.pnn_labels == ["FSC-A", "FL1-H", "FL2-H", "Time", "ratio"]
    assert fdata.pns_labels[1] == "CD4 FITC"


def test_write_fcs_var_columns(adata: AnnData) -> None:
    """var columns are stored as payload, also if they look like keywords."""
    adata.var["pns"] = "label"
    fdata = to_flowdata(adata, index_channel=None)
    assert not any(fdata.pns_labels)

    restored = read_fcs(fdata)
    assert list(restored.var["pns"]) == ["label"] * adata.n_vars


def test_write_fcs_raw(lsr_ii_fdata: flowio.FlowData) -> None:
    """Only X is written. raw (e.g. counts of a scanpy object) is reported and can be written with raw.to_adata()."""
    adata = read_fcs(lsr_ii_fdata)
    adata.raw = adata.copy()
    adata.X = adata.X * 2

    with pytest.warns(FCSConversionWarning, match="raw cannot be stored"):
        fdata = to_flowdata(adata, index_channel=None)
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), adata.X)

    fdata = to_flowdata(adata.raw.to_adata(), index_channel=None)
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), lsr_ii_fdata.as_array(preprocess=False))


def test_write_fcs_subset_channels(lsr_ii_fdata: flowio.FlowData) -> None:
    adata = read_fcs(lsr_ii_fdata)
    subset = adata[:, ["FSC-A", "SSC-A"]].copy()

    # $SPILL only refers to fluorescence channels, which are not written
    with pytest.warns(FCSConversionWarning, match=r"\$SPILL refers to channels"):
        fdata = to_flowdata(subset, index_channel=None)
    assert fdata.pnn_labels == ["FSC-A", "SSC-A"]
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), subset.X)


def test_write_fcs_preprocessed_layer(int_fdata: flowio.FlowData) -> None:
    """The preprocessed layer is derived from X and the keywords, so it is not written and not reported."""
    adata = read_fcs(int_fdata, preprocess=True)
    adata.uns["fcs"]["text"]["p2e"] = "0,0"
    del adata.uns["fcs"]["analysis"]

    fdata = to_flowdata(adata, index_channel=None)
    np.testing.assert_array_equal(fdata.as_array(preprocess=False), adata.X)

    adata.layers["counts"] = adata.X.copy()
    with pytest.warns(FCSConversionWarning, match="layers cannot be stored"):
        to_flowdata(adata, index_channel=None)


@pytest.mark.parametrize("chunk_rows", [7, 65_536])
@pytest.mark.parametrize("layout", ["dense", "csr", "csc"])
def test_write_fcs_event_blocks(
    int_fdata: flowio.FlowData, monkeypatch: pytest.MonkeyPatch, chunk_rows: int, layout: str
) -> None:
    """Events are converted block-wise: results do not depend on the block size or the matrix layout."""
    monkeypatch.setattr(_convert, "_CHUNK_ROWS", chunk_rows)
    adata = read_fcs(int_fdata)
    del adata.uns["fcs"]["analysis"]
    expected = int_fdata.as_array(preprocess=True)
    if layout != "dense":
        adata.X = getattr(sparse, f"{layout}_matrix")(adata.X)

    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        fdata = to_flowdata(adata, index_channel=None)
    # The log scaled channel FL1-H is converted to linear scale
    np.testing.assert_allclose(fdata.as_array(preprocess=True), expected, rtol=1e-6)


def test_write_fcs_precision_by_dtype() -> None:
    """Only dtypes which are not exactly representable as float32 are checked and reported."""
    for dtype in (np.int16, np.uint16, np.float16, np.float32, np.bool_):
        to_flowdata(AnnData(np.ones((3, 2), dtype=dtype)))
    to_flowdata(AnnData(np.full((3, 2), 2**24, dtype=np.int64)))
    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        to_flowdata(AnnData(np.full((3, 2), 2**24 + 1, dtype=np.int64)))
    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        to_flowdata(AnnData(np.array([[np.nan, 0.1]], dtype=np.float64)))
    # NaN and inf are exactly representable
    to_flowdata(AnnData(np.array([[np.nan, np.inf]], dtype=np.float64)))


def test_read_fcs_does_not_share_memory(lsr_ii_fdata: flowio.FlowData) -> None:
    before = np.array(lsr_ii_fdata.events)
    adata = read_fcs(lsr_ii_fdata)
    adata.X[:] = -1
    np.testing.assert_array_equal(np.array(lsr_ii_fdata.events), before)


@pytest.mark.parametrize("name", ["/A", "A/"])
def test_write_fcs_delimiter_in_channel_name(name: str) -> None:
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = [name, "B"]
    with pytest.raises(ValueError, match="must not start or end with '/'"):
        to_flowdata(adata)
    with pytest.raises(ValueError, match="must not start or end with '/'"):
        to_flowdata(AnnData(np.zeros((2, 1), dtype=np.float32)), index_channel=name)


def test_write_fcs_dollar_in_channel_name() -> None:
    adata = AnnData(np.zeros((2, 2), dtype=np.float32))
    adata.var_names = ["CD3$", "B"]
    with pytest.warns(FCSConversionWarning, match=r"contain '\$'.*CD3\$"):
        to_flowdata(adata, index_channel=None)


def test_write_fcs_text_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The size of the whole TEXT segment is checked, not only of the AnnData payload."""
    adata = AnnData(np.zeros((2, 1), dtype=np.float32))
    adata.uns["fcs"] = {"text": {"com": "x" * 5000}}
    size = len(to_flowdata(adata).text["com"])
    assert size == 5000

    monkeypatch.setattr(_convert, "TEXT_SIZE_WARNING_BYTES", 4000)
    with pytest.warns(FCSConversionWarning, match="TEXT segment is large"):
        to_flowdata(adata)

    monkeypatch.setattr(_convert, "MAX_TEXT_BYTES", 4000)
    with pytest.raises(ValueError, match="exceeds the maximum size"):
        to_flowdata(adata)


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
