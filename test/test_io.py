import io
import os

import flowio
import numpy as np
import pytest
from anndata import AnnData

from anndata_fcs import read_fcs, write_fcs
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
    assert adata.uns["fcs"]["preprocessed"] is True


def test_write_fcs(adata: AnnData, tmp_path: os.PathLike) -> None:
    path = os.path.join(tmp_path, "out.fcs")
    write_fcs(adata, path)

    handle = io.BytesIO()
    write_fcs(adata, handle)

    with open(path, "rb") as file:
        assert file.read() == handle.getvalue()
