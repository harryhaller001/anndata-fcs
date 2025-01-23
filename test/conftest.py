import os
from collections.abc import Generator

import pytest
from anndata import AnnData, read_h5ad

BASEDIR = os.path.dirname(__file__)
TESTING_FILENAME = os.path.join(BASEDIR, "testing.h5ad")


@pytest.fixture(scope="session")
def adata() -> Generator[AnnData, None, None]:
    """Load anndata testing data."""
    assert os.path.isfile(TESTING_FILENAME)
    adata = read_h5ad(TESTING_FILENAME)
    yield adata
