from numpy import double
from typing import NamedTuple

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("zigzag-dt")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from zigzag_dt.core import (
    PEAK,
    VALLEY,
    SIDEMOVE,
    EPS,
    peak_valley_pivots,
    pivots_to_modes,
    compute_segment_returns,
    compute_performance,
    compute_performance_nd,
    max_drawdown,
    zigzag,
)

# Aliases to avoid name collisions with other zigzag packages
zz_pivots = peak_valley_pivots
zz_modes = pivots_to_modes
zz_segment_returns = compute_segment_returns
zz_performance = compute_performance
zz_performance_nd = compute_performance_nd
zz_max_drawdown = max_drawdown
zz_line = zigzag

__all__ = [
    '__version__',
    'PEAK',
    'VALLEY',
    'max_drawdown',
    'peak_valley_pivots',
    'pivots_to_modes',
    'compute_segment_returns',
    'compute_performance',
    'compute_performance_nd',
    'zigzag',
    'EPS',
    # zz aliases
    'zz_pivots',
    'zz_modes',
    'zz_segment_returns',
    'zz_performance',
    'zz_performance_nd',
    'zz_max_drawdown',
    'zz_line',
]
