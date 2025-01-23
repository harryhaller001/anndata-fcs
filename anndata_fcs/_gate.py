from typing import List, Union

import flowio

from ._convert import fcs_to_dataframe


def point_in_polygon(point: List[Union[int, float]], polygon: List[List[Union[int, float]]]) -> bool:
    """Checking if a point is inside a polygon."""
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


def gate_polygon(fdata: flowio.FlowData, x: str, y: str, polygon: List[List[Union[int, float]]]) -> List[bool]:
    df = fcs_to_dataframe(fdata)

    in_polygon = []
    for point in df[[x, y]].to_numpy():
        in_polygon.append(point_in_polygon(point, polygon))

    return in_polygon
