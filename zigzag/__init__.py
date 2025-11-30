#from zigzag.core import *
from numpy import double
from typing import NamedTuple

PEAK = 1
VALLEY = -1
SIDEMOVE = 2 # future development 

__all__ = [
  'PEAK',
  'VALLEY',
  'identify_initial_pivot',
  'max_drawdown',
  'peak_valley_pivots',
  'pivots_to_modes',
  'compute_segment_returns',
  'compute_performance',
  'compute_performance_nd',
  ]
