from collections.abc import Sequence
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from flowio import FlowData
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from ._convert import fcs_to_dataframe
from ._types import NumberType, ScaleOptions


def scatter(
    data: Union[pd.DataFrame, FlowData, AnnData],
    x: str,
    y: str,
    xscale: ScaleOptions = "log",
    yscale: ScaleOptions = "log",
    density: bool = False,
    gates: Optional[Dict[str, Sequence[Sequence[NumberType]]]] = None,
    gate_color: str = "black",
    highlight: Optional[Sequence[bool]] = None,
    highlight_color: str = "red",
    color: str = "black",
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (5, 5),
) -> Axes:
    """Plot scatter from FCS data.

    Args:
        data (typing.Union[pandas.DataFrame, flowio.FlowData, anndata.AnnData]): Event data. Argument can be DataFrame, AnnData or FlowData instance.
        x (str): Channel name for x coordinate.
        y (str): Channel name for y coordinate.
        xscale (typing.Literal["linear", "log", "symlog", "logit"], optional): Scale of x axis. Defaults to `log`.
        yscale (typing.Literal["linear", "log", "symlog", "logit"], optional): Scale of y axis. Defaults to `log`.
        density (bool, optional): Show denity as color. Defaults to `False`. `scipy` is required for this option.
        gates (typing.Optional[typing.Dict[str, collections.abc[collections.abc[typing.Union[int, float]]]]], optional): Dict of gates, where the key indicated to name of the gate and the value describes to polygon. Defaults to `None`.
        gate_color (str, optional): Color of gate. Defaults to `"black"`.
        highlight (typing.Optional[List[bool]], optional): List of events to highlight. Defaults to `None`.
        highlight_color (str, optional): Color of events. Defaults to `"red"`,
        color (str): Color of default event. Defaults to `"black"`.
        ax (typing.Optional[matplotlib.axes.Axes], optional): Matplotlib axes. Default to `None`.
        figsize (typing.Tuple[int, int], optional): Figure size. Defaults to `(5, 5)`.

    Returns:
        matplotlib.axes.Axes: Axes instance.

    Raises:
        NotImplementedError: If `data` if other type than DataFrame, AnnData oder FlowData.
        ImportError: If `denity=True` and `scipy` is not installed.
        AssertionError: If length of `highlight` does not match with length of `data`.
        AssertionError: If x and y gate boundaries could not be calculated.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    formatted_data: pd.DataFrame

    if isinstance(data, pd.DataFrame):
        formatted_data = data
    elif isinstance(data, AnnData):
        formatted_data = pd.DataFrame(
            data=data.X.toarray(),
            columns=data.var.index,
            index=data.obs.index,
        )
    elif isinstance(data, FlowData):
        formatted_data = fcs_to_dataframe(data)
    else:
        raise NotImplementedError(f"Type '{type(data).__name__}' is not supported for 'data' argument.")

    assert x in formatted_data.columns and y in formatted_data.columns

    if density is True:
        values = np.vstack(
            [
                formatted_data[x].sample(n=1000, random_state=1),
                formatted_data[y].sample(n=1000, random_state=1),
            ]
        )

        try:
            from scipy import stats
        except ImportError as import_err:
            raise ImportError(
                "For density plot 'gaussian_kde' from scipy is required. Install package with 'pip install scipy'"
            ) from import_err

        kernel = stats.gaussian_kde(values)
        density_color = kernel(np.vstack([formatted_data[x], formatted_data[y]]))
        ax.scatter(x=formatted_data[x], y=formatted_data[y], c=density_color, s=1, cmap="jet")
    else:
        if highlight is None:
            ax.scatter(x=formatted_data[x], y=formatted_data[y], c=color, s=1)
        else:
            assert len(highlight) == len(formatted_data)
            ax.scatter(
                x=formatted_data[x],
                y=formatted_data[y],
                c=([highlight_color if x is True else color for x in highlight]),
                s=1,
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if gates is not None:
        for gate_name, gate_edges in gates.items():
            poly = Polygon(
                gate_edges,
                closed=True,
                edgecolor=gate_color,
                facecolor="none",
            )
            ax.add_patch(poly)

            # TODO: add different legend positions (center, top, bottom)
            # TODO: add transparrent background for annotate

            xmin: Optional[Union[float, int]] = None
            xmax: Optional[Union[float, int]] = None
            ymax: Optional[Union[float, int]] = None
            for x_coord, y_coord in gate_edges:
                if xmin is None:
                    xmin = x_coord
                elif xmin > x_coord:
                    xmin = x_coord

                if xmax is None:
                    xmax = x_coord
                elif xmax < x_coord:
                    xmax = x_coord

                if ymax is None:
                    ymax = y_coord
                elif ymax < y_coord:
                    ymax = y_coord

            assert xmax is not None and xmin is not None and ymax is not None

            annotate_x = ((xmax - xmin) / 2) + xmin
            annotate_y = ymax * 1.1

            if xscale == "symlog" or xscale == "log":
                annotate_x = np.exp(((np.log(xmax) - np.log(xmin)) / 2) + np.log(xmin))

            ax.annotate(
                text=gate_name,
                xy=(annotate_x, annotate_y),
                ha="center",
                # bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

    return ax
