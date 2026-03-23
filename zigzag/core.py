'''
Created on 5 Sept 2025

@author: alex
conversion to plain python and anc code changes that you should care about correct params when YOU call 
 identify_initial_pivot which I treat as pure internal function.
 this optimisation economy you one nanosecond in lap year 
  
'''
from zigzag.__init__ import  PEAK, VALLEY, EPS 

import pandas as pd
import numpy as np
from numpy import double, array
from numpy import ndarray, copy 

def zigzag(X: pd.Series | np.ndarray, pivots: pd.Series | np.ndarray|None = None, 
           up_rate:float=0.02, down_rate=None)->pd.Series:
    """
    Translate pivots into trend lines.
    :param pivots: the result of calling ``peak_valley_pivots``
    :return: numpy array of trend lines.
    X[Valley]+i*(dX[Valley,PEAK])
    """
    if not isinstance(pivots ,(pd.Series,np.ndarray)):
      # calc pivots first 
      if down_rate is None:
        down_rate=-up_rate
      pivots = peak_valley_pivots(X=X, up_thresh=up_rate, down_thresh= down_rate)
    mask = pivots != 0 
    # filled linear
    if isinstance(X,np.ndarray) or isinstance(X.index,pd.RangeIndex):
      filled = pd.Series(X).where(pivots != 0).interpolate(method='linear', limit_direction='both')
      #filled = np.interp(dt, dt[mask], signal[mask])
    elif isinstance(X.index,pd.DatetimeIndex):
      filled =  pd.Series(X).where(pivots != 0).interpolate(method='time', limit_direction='both')
    else:
      raise ValueError(f"Unsupported type {type(X.index)} of index 0")
    return filled

    

def identify_initial_pivot(X:[double],
                          up_thresh:double ,
                          down_thresh:double 
                          )->int :
  """
  X = np.array of double values ( All positive  ) 
  up_thresh  =  minimal relatively last VALLEY  raising value which shall be recognized as new PEAK  
  down_thresh  =  minimal relatively last PEAK  decline value which shall be recognized as new VALEY
  divider = - miminal value near 0 to escape dividing by zero  
  """
  # simple function replace 0 to min recognizible value   
#  min_value = lambda x : x if np.abs(x) > min_divider else np.sign(x)*min_divider 
#  x_0:double  = min_value(X[0])
  x_0:double  = X[0]

  max_x:double = x_0
  min_x:double = x_0
  x_t:double   = x_0

  max_t:int = 0
  min_t:int = 0

#  if up_thresh   < 1:  up_thresh   += 1.0
#  if down_thresh < 0:  down_thresh += 1.0

  for t in range(1, len(X)):
#      x_t = min_value(X[t])
      x_t = X[t]

      if x_t  >= (1 + up_thresh  ) * min_x:
          return VALLEY if min_t == 0 else PEAK

      if x_t  <= (1 + down_thresh) * max_x: # down is negatigve 
          return PEAK   if max_t == 0 else VALLEY

      if x_t > max_x:
          max_x = x_t
          max_t = t

      if x_t < min_x:
          min_x = x_t
          min_t = t
  pivot = VALLEY if X[0] < X[-1] else PEAK         
  return pivot

def _to_ndarray(X:pd.Series | list | tuple)->ndarray:
    # The type signature in peak_valley_pivots_detailed does not work for
    # pandas series because as of 0.13.0 it no longer sub-classes ndarray.
    # The workaround everyone used was to call `.values` directly before
    # calling the function. Which is fine but a little annoying.
    t = type(X)
    if t.__name__ == 'ndarray':
        pass  # Check for ndarray first for historical reasons
    elif 'pandas' in t.__module__ and 'Series'in t.__name__:
        X = X.values
    elif isinstance(X, (list, tuple)):
        X = np.array(X)
    else:
      raise ValueError(f"zigzag expecting ndarray, pandas.Series or list type, but received type '{t}' ")

    return X


def peak_valley_pivots(
    X:ndarray| pd.Series, up_thresh, down_thresh, 
                       limit_to_finalized_segments=True,
                       use_eager_switching_for_non_final=False
                       )-> ndarray[int]:
  
    X = _to_ndarray(X) 
    # Ensure float for correct signature
    if not str(X.dtype).startswith('float'):
        X = X.astype(np.float64)
    # Algorithm does not work proper on X has negative values.
    # so we find min in array and substract from array shifting X down 
    # ( or up for negative minimum ) to the zero and add EPS to make global min 
    # positive
    X  = X.copy() # DEEP COPY as we modify it 
    X -= X.min()-EPS # - and - give us positive offset        


    return peak_valley_pivots_detailed(X, up_thresh, down_thresh, 
                                       limit_to_finalized_segments, 
                                       use_eager_switching_for_non_final )


def peak_valley_pivots_detailed(
    X:ndarray[float],
    up_thresh:double,
    down_thresh:double,
    limit_to_finalized_segments:bool ,
    use_eager_switching_for_non_final:bool) -> ndarray[int]:
    """
    Find the peaks and valleys of a series.

    :param X: the series to analyze
    :param up_thresh: minimum relative change necessary to define a peak
    :param down_thesh: minimum relative change necessary to define a valley
    :return: an array with 0 indicating no pivot and -1 and 1 indicating
        valley and peak


    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias
    analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')
    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    # use fraction now 
    
    # X alredy positive 
    initial_pivot = identify_initial_pivot(
      X,  up_thresh,      down_thresh
      )
    up_thresh += 1
    down_thresh += 1
    t_n:int = len(X)
    pivots:ndarray[int]  = np.zeros(t_n, dtype=int)
    trend:int = -initial_pivot
    last_pivot_t:int = 0
    last_pivot_x:double = X[0]
    x:double
    r:double

    pivots[0] = initial_pivot


    for t in range(1, t_n):
        x = X[t]
        r = x / last_pivot_x

        if trend == VALLEY:
            if r >= up_thresh:
                pivots[last_pivot_t] = trend
                trend = PEAK
                last_pivot_x = x
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else: # trend == PEAK 
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = VALLEY
                last_pivot_x = x
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t


    if limit_to_finalized_segments:
        if use_eager_switching_for_non_final:
            if last_pivot_t > 0 and last_pivot_t < t_n-1:
                pivots[last_pivot_t] = trend
                pivots[t_n-1] = -trend
            else:
                pivots[t_n-1] = trend
        else:
            if last_pivot_t == t_n-1:
                pivots[last_pivot_t] = trend
            elif pivots[t_n-1] == 0:
                pivots[t_n-1] = -trend

    return pivots


def max_drawdown(X) -> float:
    X = _to_ndarray(X)

    # Ensure float for correct signature
    if not str(X.dtype).startswith('float'):
        X = X.astype(np.float64)

    return max_drawdown_c(X)


def max_drawdown_c(X:ndarray[double] )->double:
    """
    Compute the maximum drawdown of some sequence.

    :return: 0 if the sequence is strictly increasing.
        otherwise the abs value of the maximum drawdown
        of sequence X
    """
    mdd:double = 0
    peak:double = X[0]
    x:double
    dd:double

    for x in X:
        if x > peak:
            peak = x

        dd = (peak - x) / peak

        if dd > mdd:
            mdd = dd

    return mdd if mdd != 0.0 else 0.0


def pivots_to_modes(pivots:list[int])->ndarray[int]:
    """
    Translate pivots into trend modes.

    :param pivots: the result of calling ``peak_valley_pivots``
    :return: numpy array of trend modes. That is, between (VALLEY, PEAK] it
    is 1 and between (PEAK, VALLEY] it is -1.
    """

    x:int 
    t:int 
    modes:ndarray[int]  = np.zeros(len(pivots), dtype=int)
    mode:int  = -pivots[0] #????

    modes[0] = pivots[0]

    for t in range(1, len(pivots)):
        x = pivots[t]
        if x != 0:
            modes[t] = mode
            mode = -x
        else:
            modes[t] = mode

    return modes


def compute_segment_returns(X, pivots):
    """
    :return: numpy array of the pivot-to-pivot returns for each segment."""
    X = _to_ndarray(X)
    pivot_points = X[pivots != 0]
    return pivot_points[1:] / pivot_points[:-1] - 1.0

def compute_performance(X:pd.Series, pivots:pd.Series )->(list,list,list) :
#  we need to use time in index to correctly calc velocity of changes so here new algorithm
  (drawdowns,gains,periods) =  compute_performance_nd(_to_ndarray(X),_to_ndarray(pivots))
  # convert to pd.Series
  return (drawdowns,gains,periods)
  
def compute_performance_nd(X:ndarray, pivots:ndarray[int])->(list,list,list) :
  # extract indices of pivots and respective X values 
  # we need to use time in index to correctly calc velocity of changes
  idx:ndarray =np.flatnonzero((pivots == VALLEY) | (pivots == PEAK )) # pivot indices

  drawdowns=np.zeros(len(pivots), dtype=double)
  gains    =np.zeros(len(pivots), dtype=double)
  periods  =np.zeros(len(pivots), dtype=int)
    
  for ix in range(0,len(X)-1): # current_point
    while (idx[0] <= ix):
      # we passed this pivot delete it 
      idx=idx[1:]
    # we need two point either [peak, valley] or [valley, peak]
    # we break loop after first peak
    for j in range(0,min(2,len(idx))): # pivot indexes next to current_point
      ij=idx[j] # next pivot
      if pivots[ij] == PEAK:
        gains[ix]    = (X[ij] - X[ix])/X[ix] # ++ looks correct
        periods[ix]  = ij-ix                 # ++ looks correct
        break # we count only valleys before peaks
      else:   # pivots[ij] == VALLEY
        drawdowns[ix] = (X[ix] - X[ij])/X[ix] # ++ looks correct
    # completed one item
  return (drawdowns,gains,periods)
