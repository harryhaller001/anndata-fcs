from typing import Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon


def scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    xscale: Literal["linear", "log", "symlog", "logit"] = "log",
    yscale: Literal["linear", "log", "symlog", "logit"] = "log",
    density: bool = False,
    gates: Optional[Dict[str, List[List[Union[int, float]]]]] = None,
    gate_color: str = "black",
    highlight: Optional[List[bool]] = None,
    highlight_color: str = "red",
    color: str = "black",
):
    _fig, _ax = plt.subplots(figsize=(5, 5))

    if density is True:
        values = np.vstack(
            [
                data[x].sample(n=1000, random_state=1),
                data[y].sample(n=1000, random_state=1),
            ]
        )

        try:
            from scipy import stats
        except ImportError as import_err:
            raise ImportError(
                "For density plot 'gaussian_kde' from scipy is required. Install package with 'pip install scipy'"
            ) from import_err

        kernel = stats.gaussian_kde(values)
        density_color = kernel(np.vstack([data[x], data[y]]))
        _ax.scatter(x=data[x], y=data[y], c=density_color, s=1, cmap="jet")
    else:
        if highlight is None:
            _ax.scatter(x=data[x], y=data[y], c=color, s=1)
        else:
            assert len(highlight) == len(data)
            _ax.scatter(
                x=data[x],
                y=data[y],
                c=([highlight_color if x is True else color for x in highlight]),
                s=1,
            )

    _ax.set_xscale(xscale)
    _ax.set_yscale(yscale)
    _ax.set_xlabel(x)
    _ax.set_ylabel(y)

    if gates is not None:
        for gate_name, gate_edges in gates.items():
            poly = Polygon(
                gate_edges,
                closed=True,
                edgecolor=gate_color,
                facecolor="none",
            )
            _ax.add_patch(poly)

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

            _ax.annotate(
                text=gate_name,
                xy=(annotate_x, annotate_y),
                ha="center",
                # bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
            )

    return _fig, _ax
