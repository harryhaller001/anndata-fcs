from datetime import datetime

import pytest

from anndata_fcs import FCSConversionWarning
from anndata_fcs._keywords import (
    assign_channels,
    build_metadata,
    check_spillover,
    escape_key,
    format_timestamp,
    parse_pne,
    remove_channel,
    split_text,
    unescape_key,
    unique_channel_names,
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
    text["p1b"] = "32"
    text["p99b"] = "8"
    split = split_text(text, n_channels=10)

    assert split.structural == {"begindata": "100", "tot": "5", "p1b": "32"}
    assert split.payload == {"anndata_fcs_schema": "1"}
    assert split.text == {
        "p1n": "FSC-A",
        "p1s": "Forward",
        "p10n": "CD4",
        "p10display": "LOG",
        "p99n": "out of range",
        "plate name": "plate",
        "cyt": "LSRII",
        "p99b": "8",
    }


def test_remove_channel() -> None:
    keywords = {"p1n": "A", "p2n": "B", "p2s": "x", "p3n": "C", "p3r": "1", "p9n": "out of range", "cyt": "LSRII"}
    assert remove_channel(keywords, 2, n_channels=3) == {
        "p1n": "A",
        "p2n": "C",
        "p2r": "1",
        "p9n": "out of range",
        "cyt": "LSRII",
    }


def test_assign_channels() -> None:
    keywords = {"p1n": "A", "p1s": "CD3", "p2n": "B", "p2r": "1024", "p3n": "C", "p9n": "out of range", "cyt": "LSRII"}

    channels, global_keywords = assign_channels(keywords, ["B", "A", "D"], n_channels=3)
    assert channels == [{"n": "B", "r": "1024"}, {"n": "A", "s": "CD3"}, {}]
    assert global_keywords == {"p9n": "out of range", "cyt": "LSRII"}

    channels, global_keywords = assign_channels(keywords, ["out of range"])
    assert channels == [{"n": "out of range"}]
    assert global_keywords == {"cyt": "LSRII"}


def test_assign_channels_duplicate_names() -> None:
    """Duplicate $PnN are made unique like var_names when reading, so each duplicate keeps its keywords."""
    keywords = {"p1n": "A", "p1s": "CD3", "p2n": "A", "p2s": "CD4", "p3n": "A-1", "p3s": "CD8"}
    names = unique_channel_names(["A", "A", "A-1"])
    assert len(set(names)) == 3 and names[0] == "A"

    channels, _ = assign_channels(keywords, list(reversed(names)), n_channels=3)
    assert [channel["s"] for channel in channels] == ["CD8", "CD4", "CD3"]


def test_build_metadata() -> None:
    channels = [
        {"n": "A", "b": "16", "e": "0,0", "s": "CD3", "r": "1024", "v": "500"},
        {"n": "B", "r": "262144"},
    ]
    metadata, pns, messages = build_metadata(channels, {"cyt": "LSRII", "$DATE": "today", "tot": "5", "a/b": "c"})

    assert pns == ["CD3", ""]
    assert metadata == {"p1r": "1024", "p1v": "500", "p2r": "262144", "cyt": "LSRII", "date": "today", "a//b": "c"}
    assert messages == []


def test_build_metadata_empty_value() -> None:
    metadata, _, messages = build_metadata([{"n": "A", "s": ""}], {"com": ""})
    assert metadata == {}
    assert len(messages) == 2 and all("empty value" in message for message in messages)


@pytest.mark.parametrize("value", ["/leading", "trailing/", "/"])
def test_build_metadata_delimiter_value(value: str) -> None:
    """FlowIO cannot parse values starting or ending with the delimiter, which also corrupts following keywords."""
    metadata, _, messages = build_metadata([{"s": value}], {"com": value, "cyt": "LSRII"})
    assert metadata == {"cyt": "LSRII"}
    assert len(messages) == 2 and all("It is not written" in message for message in messages)


@pytest.mark.parametrize("key", ["/leading", "trailing/"])
def test_build_metadata_delimiter_keyword(key: str) -> None:
    metadata, _, messages = build_metadata([], {key: "value", "cyt": "LSRII"})
    assert metadata == {"cyt": "LSRII"}
    assert len(messages) == 1 and "delimiter" in messages[0]


def test_build_metadata_dollar_value() -> None:
    """Values containing '$' are written, although FlowIO removes '$' when parsing."""
    metadata, _, messages = build_metadata([], {"com": "dollar $ sign"})
    assert metadata == {"com": "dollar $ sign"}
    assert len(messages) == 1 and "'$'" in messages[0]


def test_build_metadata_invalid_keyword() -> None:
    metadata, _, messages = build_metadata([], {"größe": "1", "cyt": "LSRII"})
    assert metadata == {"cyt": "LSRII"}
    assert len(messages) == 1 and "printable ASCII" in messages[0]


def test_build_metadata_channel_keyword_in_global_keywords() -> None:
    """$Pn keywords of channels beyond $PAR must not be attributed to channels of the written file."""
    metadata, _, messages = build_metadata([{"r": "1"}, {"r": "1"}], {"p2foo": "x", "p3foo": "y"})
    assert metadata == {"p1r": "1", "p2r": "1", "p3foo": "y"}
    assert len(messages) == 1 and "'p2foo' refers to channel 2" in messages[0]


def test_format_timestamp() -> None:
    assert format_timestamp(datetime(2026, 3, 4, 5, 6, 7, 890_000)) == "04-MAR-2026 05:06:07.89"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0,0", (0.0, 0.0)), ("4,1", (4.0, 1.0)), ("4,0", (4.0, 1.0)), ("3.5,0.1", (3.5, 0.1))],
)
def test_parse_pne(value: str, expected: tuple[float, float]) -> None:
    assert parse_pne(value) == expected


@pytest.mark.parametrize("value", ["", "4", "4,1,0", "a,b"])
def test_parse_pne_invalid(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid \\$PnE value"):
        parse_pne(value)


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
