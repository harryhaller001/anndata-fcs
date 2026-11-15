import io
import os
import warnings
from datetime import datetime

import flowio
import numpy as np
import pytest
from anndata import AnnData

from anndata_fcs import FCSConversionWarning, _convert, read_fcs, write_fcs
from test.conftest import LSR_II_FILENAME


@pytest.mark.parametrize("as_pathlike", [False, True])
def test_read_fcs_path(as_pathlike: bool) -> None:
    from pathlib import Path

    path = Path(LSR_II_FILENAME) if as_pathlike else LSR_II_FILENAME
    adata = read_fcs(path)
    assert adata.shape == (14945, 11)


def test_read_fcs_handle_and_flowdata(int_fcs_bytes: bytes, int_fdata: flowio.FlowData) -> None:
    from_handle = read_fcs(io.BytesIO(int_fcs_bytes))
    from_fdata = read_fcs(int_fdata)
    np.testing.assert_array_equal(from_handle.X, from_fdata.X)

    with pytest.raises(TypeError):
        read_fcs(int_fdata, ignore_offset_error=True)


def test_read_fcs_flowio_kwargs(int_fcs_bytes: bytes) -> None:
    adata = read_fcs(io.BytesIO(int_fcs_bytes), ignore_offset_discrepancy=True, preprocess=True)
    assert "preprocessed" in adata.layers


def test_write_fcs(adata: AnnData, tmp_path: os.PathLike, monkeypatch: pytest.MonkeyPatch) -> None:
    # $LAST_MODIFIED is the only non-deterministic keyword
    monkeypatch.setattr(_convert, "_now", lambda: datetime(2026, 1, 2, 3, 4, 5))
    path = os.path.join(tmp_path, "out.fcs")
    write_fcs(adata, path)

    handle = io.BytesIO()
    write_fcs(adata, handle)

    with open(path, "rb") as file:
        assert file.read() == handle.getvalue()


def test_warnings_point_to_caller(adata: AnnData) -> None:
    """Warnings are attributed to the calling code, independent of the internal call depth."""
    handle = io.BytesIO()
    flowio.create_fcs(handle, np.zeros(4, dtype=np.float32), ["A", "A"])
    for read in (
        lambda: read_fcs(io.BytesIO(handle.getvalue())),
        lambda: read_fcs(flowio.FlowData(io.BytesIO(handle.getvalue()))),
    ):
        with pytest.warns(FCSConversionWarning, match="not unique") as record:
            read()
        assert record[0].filename == __file__

    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 2))
    adata.uns["unsupported"] = object()
    adata.uns["fcs"] = {"text": {"com": ""}}
    with pytest.warns(FCSConversionWarning) as record:
        write_fcs(adata, io.BytesIO())
    # Warnings from different internal call depths: slots, payload and keywords
    assert len(record) == 3
    assert {warning.filename for warning in record} == {__file__}


def test_write_fcs_handle_position(adata: AnnData, monkeypatch: pytest.MonkeyPatch) -> None:
    """Handles not at position 0 (e.g. appending) get the file after their current content."""
    # $LAST_MODIFIED is the only non-deterministic keyword
    monkeypatch.setattr(_convert, "_now", lambda: datetime(2026, 1, 2, 3, 4, 5))
    expected = io.BytesIO()
    write_fcs(adata, expected)

    handle = io.BytesIO()
    handle.write(b"prefix")
    write_fcs(adata, handle)
    assert handle.getvalue() == b"prefix" + expected.getvalue()


def test_write_fcs_does_not_truncate_on_error(adata: AnnData, tmp_path: os.PathLike) -> None:
    path = os.path.join(tmp_path, "out.fcs")
    with open(path, "wb") as file:
        file.write(b"existing")
    adata.obsm["X_pca"] = np.zeros((adata.n_obs, 2))
    # Warnings turned into errors are raised before the file is opened
    with warnings.catch_warnings():
        warnings.simplefilter("error", FCSConversionWarning)
        with pytest.raises(FCSConversionWarning, match="obsm"):
            write_fcs(adata, path)
    with open(path, "rb") as file:
        assert file.read() == b"existing"
