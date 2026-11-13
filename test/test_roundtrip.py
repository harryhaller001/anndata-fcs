"""Round trip tests for the conversion contract.

1. FCS -> AnnData -> FCS keeps event data, per-channel keywords and non-structural global keywords.
2. AnnData -> FCS -> AnnData keeps X, var_names, var, obs_names, obs and JSON serialisable uns entries.
3. FCS -> AnnData -> h5ad -> AnnData -> FCS gives the same result as 1.
"""

import io
import os
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import flowio
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

from anndata_fcs import FCSConversionWarning, anndata_to_fcs, fcs_to_anndata, read_fcs, write_fcs
from anndata_fcs._schema import KEYWORD_PREFIX, STRUCTURAL_KEYWORDS


def _comparable_text(fdata: flowio.FlowData) -> dict[str, str]:
    """Keywords expected to be identical after a round trip ($PnB is regenerated)."""
    return {
        key: value
        for key, value in fdata.text.items()
        if key not in STRUCTURAL_KEYWORDS
        and not key.startswith(KEYWORD_PREFIX)
        and not (key.startswith("p") and key.endswith("b") and key[1:-1].isdigit())
    }


@contextmanager
def _expect_conversion_warnings(*patterns: str) -> Iterator[None]:
    """Assert that exactly the given FCSConversionWarnings (regex patterns) are emitted, no more and no less."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", FCSConversionWarning)
        yield

    messages = [str(warning.message) for warning in record if issubclass(warning.category, FCSConversionWarning)]
    for pattern in patterns:
        assert any(re.search(pattern, message) for message in messages), f"Missing warning matching '{pattern}'"
    for message in messages:
        assert any(re.search(pattern, message) for pattern in patterns), f"Unexpected warning: {message}"


def _write_read(adata: AnnData, **kwargs) -> flowio.FlowData:
    handle = io.BytesIO()
    write_fcs(adata, handle, **kwargs)
    handle.seek(0)
    return flowio.FlowData(handle)


def _assert_same_anndata(restored: AnnData, original: AnnData) -> None:
    X = original.X.toarray() if hasattr(original.X, "toarray") else original.X
    np.testing.assert_array_equal(restored.X, X)
    assert list(restored.obs_names) == list(original.obs_names)
    assert list(restored.var_names) == list(original.var_names)
    pd.testing.assert_frame_equal(restored.obs, original.obs, check_index_type=False)
    pd.testing.assert_frame_equal(restored.var[original.var.columns], original.var, check_index_type=False)


# ---------------------------------------------------------------------------------------------------------------------
# 1. FCS -> AnnData -> FCS
# ---------------------------------------------------------------------------------------------------------------------


def test_fcs_roundtrip_float(lsr_ii_fdata: flowio.FlowData) -> None:
    fdata = _write_read(fcs_to_anndata(lsr_ii_fdata), index_channel=None)

    np.testing.assert_array_equal(fdata.as_array(preprocess=False), lsr_ii_fdata.as_array(preprocess=False))
    assert fdata.pnn_labels == lsr_ii_fdata.pnn_labels
    assert _comparable_text(fdata) == _comparable_text(lsr_ii_fdata)


def test_fcs_roundtrip_int(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata)
    del adata.uns["fcs"]["analysis"]

    # Log scaled channel cannot be represented exactly as float32 after linearisation
    with pytest.warns(FCSConversionWarning, match="32-bit float"):
        fdata = _write_read(adata, index_channel=None)

    assert fdata.data_type == "F"
    np.testing.assert_allclose(fdata.as_array(preprocess=True), int_fdata.as_array(preprocess=True), rtol=1e-6)
    assert fdata.text["p2e"] == "0,0"
    assert fdata.text["p2r"] == "10000"
    assert fdata.text["p3g"] == "2.0"
    assert fdata.pns_labels == int_fdata.pns_labels
    assert fdata.text["custom/key"] == "custom value"
    assert fdata.text["timestep"] == "0.01"


def test_fcs_roundtrip_preprocessed(int_fdata: flowio.FlowData) -> None:
    adata = fcs_to_anndata(int_fdata, preprocess=True)
    del adata.uns["fcs"]["analysis"]

    # Preprocessed values (log scale, gain) are float64 and not exactly representable as float32
    with _expect_conversion_warnings("32-bit float"):
        fdata = _write_read(adata, index_channel=None)

    np.testing.assert_allclose(fdata.as_array(preprocess=True), int_fdata.as_array(preprocess=True), rtol=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# 2. AnnData -> FCS -> AnnData
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("index_channel", ["barcode_rank", None])
def test_anndata_roundtrip(adata_annotated: AnnData, index_channel: str | None) -> None:
    restored = fcs_to_anndata(anndata_to_fcs(adata_annotated, index_channel=index_channel))

    _assert_same_anndata(restored, adata_annotated)
    assert restored.uns["params"]["resolution"] == 0.5
    assert restored.uns["params"]["markers"] == ["CD3", "CD19"]
    np.testing.assert_array_equal(restored.uns["params"]["matrix"], np.eye(2))


def test_anndata_roundtrip_twice(adata_annotated: AnnData) -> None:
    once = fcs_to_anndata(anndata_to_fcs(adata_annotated))
    twice = fcs_to_anndata(anndata_to_fcs(once))

    _assert_same_anndata(twice, once)
    assert twice.uns["fcs"]["text"] == once.uns["fcs"]["text"]


def test_anndata_roundtrip_file(adata_annotated: AnnData, tmp_path: os.PathLike) -> None:
    path = os.path.join(tmp_path, "citeseq.fcs")
    write_fcs(adata_annotated, path)

    _assert_same_anndata(read_fcs(path), adata_annotated)


def test_anndata_roundtrip_gated_subset(adata_annotated: AnnData) -> None:
    """Events exported from a gate are mapped back to their observations using the index channel."""
    fdata = anndata_to_fcs(adata_annotated)
    subset = np.array([5, 1, 999, 42])

    # Simulate an exported gate: same keywords, subset of events
    metadata = {key: value for key, value in fdata.text.items() if key not in STRUCTURAL_KEYWORDS}
    events = fdata.as_array(preprocess=False)[subset]
    handle = io.BytesIO()
    flowio.create_fcs(
        handle,
        events.ravel(),
        fdata.pnn_labels,
        opt_channel_names=fdata.pns_labels,
        metadata_dict=metadata,
    )
    handle.seek(0)

    restored = fcs_to_anndata(flowio.FlowData(handle))

    assert list(restored.obs_names) == list(adata_annotated.obs_names[subset])
    pd.testing.assert_frame_equal(restored.obs, adata_annotated.obs.iloc[subset], check_index_type=False)
    pd.testing.assert_frame_equal(
        restored.var[adata_annotated.var.columns], adata_annotated.var, check_index_type=False
    )


# ---------------------------------------------------------------------------------------------------------------------
# 3. FCS -> AnnData -> h5ad -> AnnData -> FCS
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "expected_warnings"),
    [
        ("lsr_ii_fdata", ()),
        # Log scaled channel is linearised (not exact in float32), ANALYSIS segment cannot be written
        ("int_fdata", ("32-bit float", "ANALYSIS")),
    ],
)
def test_h5ad_roundtrip(
    fixture: str,
    expected_warnings: tuple[str, ...],
    request: pytest.FixtureRequest,
    tmp_path: os.PathLike,
) -> None:
    source: flowio.FlowData = request.getfixturevalue(fixture)
    adata = fcs_to_anndata(source)

    path = os.path.join(tmp_path, "data.h5ad")
    adata.write_h5ad(path)
    from_h5ad = read_h5ad(path)

    assert from_h5ad.uns["fcs"]["text"] == adata.uns["fcs"]["text"]

    with _expect_conversion_warnings(*expected_warnings):
        direct = _write_read(adata, index_channel=None)
    with _expect_conversion_warnings(*expected_warnings):
        via_h5ad = _write_read(from_h5ad, index_channel=None)

    np.testing.assert_array_equal(via_h5ad.as_array(preprocess=False), direct.as_array(preprocess=False))
    assert via_h5ad.text == direct.text


def test_h5ad_roundtrip_anndata(adata_annotated: AnnData, tmp_path: os.PathLike) -> None:
    restored = fcs_to_anndata(anndata_to_fcs(adata_annotated))
    path = os.path.join(tmp_path, "data.h5ad")
    restored.write_h5ad(path)

    from_h5ad = read_h5ad(path)
    again = fcs_to_anndata(anndata_to_fcs(from_h5ad))

    np.testing.assert_array_equal(again.X, restored.X)
    assert list(again.obs_names) == list(restored.obs_names)
    pd.testing.assert_frame_equal(again.obs, from_h5ad.obs, check_index_type=False)
