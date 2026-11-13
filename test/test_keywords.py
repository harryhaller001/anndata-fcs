import pytest

from anndata_fcs import FCSConversionWarning
from anndata_fcs._keywords import build_metadata, escape_key, split_text, unescape_key


@pytest.mark.parametrize("key", ["cyt", "custom/key", "100%", "a%2Fb", "/%/"])
def test_escape_roundtrip(key: str) -> None:
    escaped = escape_key(key)
    assert "/" not in escaped
    assert unescape_key(escaped) == key


def test_split_text() -> None:
    text = {
        "begindata": "100  ",
        "tot": "5",
        "p1n": "FSC-A",
        "p1s": "Forward",
        "p10n": "CD4",
        "p10display": "LOG",
        "p99n": "out of range",
        "plate name": "plate",
        "anndata_fcs_schema": "1",
        "cyt": "LSRII",
    }
    split = split_text(text, n_channels=10)

    assert split.structural == {"begindata": "100", "tot": "5"}
    assert split.channels[1] == {"n": "FSC-A", "s": "Forward"}
    assert split.channels[10] == {"n": "CD4", "display": "LOG"}
    assert split.channels[2] == {}
    assert split.payload == {"anndata_fcs_schema": "1"}
    assert split.global_ == {"p99n": "out of range", "plate name": "plate", "cyt": "LSRII"}


def test_build_metadata() -> None:
    channels = [
        {"n": "A", "b": "16", "e": "0,0", "s": "CD3", "r": "1024", "v": "500"},
        {"n": "B", "r": "262144"},
    ]
    metadata, pns = build_metadata(channels, {"cyt": "LSRII", "$DATE": "today", "tot": "5", "a/b": "c"})

    assert pns == ["CD3", ""]
    assert metadata == {"p1r": "1024", "p1v": "500", "p2r": "262144", "cyt": "LSRII", "date": "today", "a//b": "c"}


def test_build_metadata_empty_value() -> None:
    with pytest.warns(FCSConversionWarning, match="empty value"):
        metadata, _ = build_metadata([{"n": "A", "s": ""}], {"com": ""})
    assert metadata == {}


@pytest.mark.parametrize("value", ["/leading", "trailing/", "dollar $ sign"])
def test_build_metadata_unparseable_value(value: str) -> None:
    with pytest.warns(FCSConversionWarning):
        metadata, _ = build_metadata([], {"com": value})
    assert metadata == {"com": value}
