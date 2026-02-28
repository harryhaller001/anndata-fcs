import io
from collections.abc import Sequence

import flowio

from ._convert import fcs_to_dataframe
from ._types import NumberType


def point_in_polygon(point: Sequence[NumberType], polygon: Sequence[Sequence[NumberType]]) -> bool:
    """Checking if a point is inside a polygon.

    Args:
        point (collections.abc.Sequence[typing.Union[int, float]]): Point data with structure `[x, y]`
        polygon (collections.abc.Sequence[collections.abc.Sequence[typing.Union[int, float]]]): Points of polygon with structure `[[x1, y1], [xn, yn], ...]`.

    Returns:
        bool: `True` is point is inside of polygon.
    """
    num_vertices = len(polygon)
    x, y = point[0], point[1]
    inside = False
    p1 = polygon[0]

    for i in range(1, num_vertices + 1):
        p2 = polygon[i % num_vertices]
        if y > min(p1[1], p2[1]):
            if y <= max(p1[1], p2[1]):
                if x <= max(p1[0], p2[0]):
                    x_intersection = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]

                    if p1[0] == p2[0] or x <= x_intersection:
                        inside = not inside
        p1 = p2

    return inside


def gate_polygon(fdata: flowio.FlowData, x: str, y: str, polygon: Sequence[Sequence[NumberType]]) -> Sequence[bool]:
    """Gate channel `x` and `y` of FlowData instance with polygon.

    Args:
        fdata (flowio.FlowData): FlowData instance to gate.
        x (str): Channel name for x coordinate.
        y (str): Channel name for y coordinate.
        polygon (collections.abc.Sequence[collections.abc.Sequence[typing.Union[int, float]]]): Points of polygon with structure `[[x1, y1], [xn, yn], ...]`.

    Returns:
        collections.abc.Sequence[bool]: List of boolean with length of events per channel, where `True` indicated that point is inside the polygon.
    """
    df = fcs_to_dataframe(fdata)

    in_polygon = []
    for point in df[[x, y]].to_numpy():
        in_polygon.append(point_in_polygon(tuple(point), polygon))

    return in_polygon


def gate_polygon_subset(
    fdata: flowio.FlowData, x: str, y: str, polygon: Sequence[Sequence[NumberType]]
) -> flowio.FlowData:
    """Gating polygon on flow data object and returning subset.

    Args:
        fdata (flowio.FlowData): FlowData instance to gate.
        x (str): Channel name for x coordinate.
        y (str): Channel name for y coordinate.
        polygon (collections.abc.Sequence[collections.abc.Sequence[typing.Union[int, float]]]): Points of polygon with structure `[[x1, y1], [xn, yn], ...]`.

    Returns:
        flowio.FlowData: Subset on `fdata` object, only containing events inside of polygon.
    """
    df = fcs_to_dataframe(fdata)
    in_polygon = gate_polygon(fdata=fdata, x=x, y=y, polygon=polygon)

    assert len(df) == len(in_polygon)

    # Filter out in polygon
    df = df[in_polygon]

    # convert to FlowData object
    fcs_obj = flowio.create_fcs(
        file_handle=io.BytesIO(),
        event_data=df.to_numpy().flatten(),
        channel_names=df.columns,
    )
    return flowio.FlowData(fcs_obj)
