import os
import sys
import warnings

_PACKAGE_PREFIX = os.path.dirname(os.path.abspath(__file__)) + os.sep


class FCSConversionWarning(UserWarning):
    """Warning emitted when information cannot be converted between AnnData and FCS without loss."""


def warn(message: str) -> None:
    """Emit a :class:`FCSConversionWarning` attributed to the first caller outside of anndata-fcs.

    Internal call depths differ between code paths (e.g. :func:`anndata_fcs.read_fcs` vs.
    :func:`anndata_fcs.fcs_to_anndata`), so a fixed ``stacklevel`` would point into the package for some of them.
    """
    stacklevel = 2
    frame = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_PREFIX):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, FCSConversionWarning, stacklevel=stacklevel)
