import warnings

import flowio
import numpy as np
import pandas as pd


def fcs_to_dataframe(fdata: flowio.FlowData, legacy_flowio: bool = False) -> pd.DataFrame:
    """[deprecated] Converts FlowData instance to pandas DataFrame.

    Args:
        fdata (flowio.FlowData): FlowData instance to convert.
        legacy_flowio (bool, optional): Support for legacy `flowio<1.4.0`. Defaults to `false`.

    Returns:
        pandas.DataFrame: Pandas dataframe.
    """
    warnings.warn(
        message="This method is deprecated and will be remove in the next version of `anndata_fcs`.",
        category=DeprecationWarning,
        stacklevel=1,
    )
    events_colname: str = "pnn"
    if legacy_flowio is True:
        events_colname = "PnN"

    return pd.DataFrame(
        np.reshape(np.asarray(fdata.events), (-1, fdata.channel_count)),  # type: ignore[call-overload]
        columns=[v[events_colname] for v in fdata.channels.values()],
    )
