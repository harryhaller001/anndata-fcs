from datetime import datetime

import pytest

from anndata_fcs import FCSConversionWarning
from anndata_fcs._keywords import (
    build_metadata,
    check_spillover,
    escape_key,
    format_timestamp,
    parse_pne,
    split_text,
    unescape_key,
)


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


def test_build_metadata_invalid_keyword() -> None:
    with pytest.warns(FCSConversionWarning, match="printable ASCII"):
        metadata, _ = build_metadata([], {"größe": "1", "cyt": "LSRII"})
    assert metadata == {"cyt": "LSRII"}


def test_build_metadata_channel_keyword_in_global_keywords() -> None:
    """$Pn keywords of channels beyond $PAR must not be attributed to channels of the written file."""
    with pytest.warns(FCSConversionWarning, match="refers to channel 2"):
        metadata, _ = build_metadata([{"r": "1"}, {"r": "1"}], {"p2foo": "x", "p3foo": "y"})
    assert metadata == {"p1r": "1", "p2r": "1", "p3foo": "y"}


def test_format_timestamp() -> None:
    assert format_timestamp(datetime(2026, 3, 4, 5, 6, 7, 890_000)) == "04-MAR-2026 05:06:07.89"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0,0", (0.0, 0.0)), ("4,1", (4.0, 1.0)), ("4,0", (4.0, 1.0)), ("3.5,0.1", (3.5, 0.1))],
)
def test_parse_pne(value: str, expected: tuple[float, float]) -> None:
    assert parse_pne(value) == expected


SPILLOVER = "3,FL1-A,FL2-A,FL3-A,1,0.1,0.04,0.2,1,0.13,0.15,0.08,1"


def test_check_spillover_valid() -> None:
    assert check_spillover(SPILLOVER, ["FSC-A", "FL1-A", "FL2-A", "FL3-A"]) == (SPILLOVER, [])


def test_check_spillover_subset() -> None:
    value, messages = check_spillover(SPILLOVER, ["FL1-A", "FL3-A"])
    assert value == "2,FL1-A,FL3-A,1,0.04,0.15,1"
    assert len(messages) == 1 and "FL2-A" in messages[0]


def test_check_spillover_no_channels() -> None:
    value, messages = check_spillover(SPILLOVER, ["FSC-A"])
    assert value is None
    assert "dropped" in messages[0]


@pytest.mark.parametrize("value", ["", "x,A,1", "2,A,B,1,0,0", "2,A,A,1,0,0,1", "2,A,B,1,0,0,x"])
def test_check_spillover_malformed(value: str) -> None:
    assert check_spillover(value, ["A", "B"]) == (None, ["$SPILLOVER is malformed and is dropped."])


def test_check_spillover_singular() -> None:
    with pytest.warns(FCSConversionWarning, match="not invertible"):
        value, messages = check_spillover("2,A,B,1,1,1,1", ["A", "B"])
    assert value == "2,A,B,1,1,1,1"
    assert messages == []
