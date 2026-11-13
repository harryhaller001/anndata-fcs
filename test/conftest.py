import io
import os
from collections.abc import Mapping

import flowio
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

BASEDIR = os.path.dirname(__file__)
TESTING_FILENAME = os.path.join(BASEDIR, "testing.h5ad")

# BD LSR-II file from the fcsparser test suite (https://github.com/eyurtsev/fcsparser, MIT license)
LSR_II_FILENAME = os.path.join(BASEDIR, "data", "HTS_BD_LSR_II_Mixed_Specimen_001_D6_D06.fcs")


@pytest.fixture(scope="session")
def _adata() -> AnnData:
    assert os.path.isfile(TESTING_FILENAME)
    return read_h5ad(TESTING_FILENAME)


@pytest.fixture
def adata(_adata: AnnData) -> AnnData:
    """CITE-seq testing data (1000 cells x 32 antibodies). A fresh copy per test."""
    return _adata.copy()


@pytest.fixture
def adata_annotated(adata: AnnData) -> AnnData:
    """CITE-seq data with obs columns of several dtypes and uns entries."""
    rng = np.random.default_rng(0)
    adata.obs["cluster"] = pd.Categorical(rng.choice(["T/NK", "B$", "Mono"], adata.n_obs))
    adata.obs["score"] = rng.random(adata.n_obs)
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.int64)
    adata.obs["is_doublet"] = rng.random(adata.n_obs) > 0.9
    adata.uns["params"] = {"resolution": 0.5, "markers": ["CD3", "CD19"], "matrix": np.eye(2)}
    return adata


@pytest.fixture
def lsr_ii_fdata() -> flowio.FlowData:
    """Real float FCS 3.0 file from a BD LSR-II."""
    assert os.path.isfile(LSR_II_FILENAME)
    return flowio.FlowData(LSR_II_FILENAME)


def write_int_fcs(
    events: np.ndarray,
    channels: list[Mapping[str, str]],
    keywords: Mapping[str, str],
    analysis: Mapping[str, str] | None = None,
) -> bytes:
    """Write a minimal FCS 3.0 file with 16-bit integer data (not supported by flowio.create_fcs).

    Args:
        events: Event matrix (events x channels), stored as big endian uint16.
        channels: Per-channel keywords as suffix -> value (``n`` is required, ``b`` is set to 16).
        keywords: Additional global keywords.
        analysis: Optional ANALYSIS segment keywords.
    """
    delimiter = "/"
    header_size = 58
    width = 10  # fixed width offsets avoid iterating over offset lengths

    data = np.ascontiguousarray(events, dtype=">u2").tobytes()

    def escape(value: str) -> str:
        return value.replace(delimiter, delimiter * 2)

    def segment(pairs: Mapping[str, str]) -> bytes:
        return (delimiter + "".join(f"{escape(k)}{delimiter}{escape(v)}{delimiter}" for k, v in pairs.items())).encode()

    analysis_bytes = segment(analysis) if analysis else b""

    def text_for(begin_data: int, begin_analysis: int) -> dict[str, str]:
        text = {
            "$BEGINANALYSIS": str(begin_analysis if analysis else 0).zfill(width),
            "$ENDANALYSIS": str(begin_analysis + len(analysis_bytes) - 1 if analysis else 0).zfill(width),
            "$BEGINSTEXT": "0",
            "$ENDSTEXT": "0",
            "$BEGINDATA": str(begin_data).zfill(width),
            "$ENDDATA": str(begin_data + len(data) - 1).zfill(width),
            "$BYTEORD": "4,3,2,1",
            "$DATATYPE": "I",
            "$MODE": "L",
            "$NEXTDATA": "0",
            "$PAR": str(len(channels)),
            "$TOT": str(events.shape[0]),
        }
        for number, channel in enumerate(channels, start=1):
            text[f"$P{number}B"] = "16"
            for suffix, value in channel.items():
                text[f"$P{number}{suffix.upper()}"] = value
        text.update(keywords)
        return text

    text_size = len(segment(text_for(0, 0)))
    begin_data = header_size + text_size
    begin_analysis = begin_data + len(data)
    text_bytes = segment(text_for(begin_data, begin_analysis))
    assert len(text_bytes) == text_size

    def field(value: int) -> str:
        return str(value).rjust(8)

    header = "FCS3.0    " + "".join(
        field(v)
        for v in (
            header_size,
            begin_data - 1,
            begin_data,
            begin_data + len(data) - 1,
            begin_analysis if analysis else 0,
            begin_analysis + len(analysis_bytes) - 1 if analysis else 0,
        )
    )
    assert len(header) == header_size
    return header.encode() + text_bytes + data + analysis_bytes


@pytest.fixture
def int_fcs_bytes() -> bytes:
    """Integer FCS file with a log scaled channel, a channel with gain, a time channel and an ANALYSIS segment."""
    rng = np.random.default_rng(1)
    events = rng.integers(0, 1024, size=(500, 4))
    channels = [
        {"n": "FSC-A", "r": "1024", "e": "0,0"},
        {"n": "FL1-H", "s": "CD4 FITC", "r": "1024", "e": "4,1", "v": "650"},
        {"n": "FL2-H", "s": "CD8/PE", "r": "1024", "e": "0,0", "g": "2.0"},
        {"n": "Time", "r": "1024", "e": "0,0", "g": "0.01"},
    ]
    keywords = {"$CYT": "FACSCalibur", "$TIMESTEP": "0.01", "$DATE": "01-JAN-2020", "CUSTOM/KEY": "custom value"}
    return write_int_fcs(events, channels, keywords, analysis={"GATE1": "lymphocytes"})


@pytest.fixture
def int_fdata(int_fcs_bytes: bytes) -> flowio.FlowData:
    return flowio.FlowData(io.BytesIO(int_fcs_bytes))
