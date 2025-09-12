'''
Created on 12 Sept 2025

@author: alex

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zigzag as zzg
import plotting as pl
from core import zigzag




def main():
    # This is not nessessary to use zigzag. It's only here so that
    # this example is reproducible.
    np.random.seed(1997)
    X = np.cumprod(1 + np.random.randn(200) * 0.1)
    change = 0.015
    pivots = zzg.peak_valley_pivots(X, change, -change)
    ax=plt.new_figure_manager(num=1)
    pl.plot_pivots(X, pivots)
#    dX=(X[1:] - X[:-1])/X[1:]
#    pl.plot_dX(dX,limit=change)
    modes = zzg.pivots_to_modes(pivots)
    xx=pd.Series(X).pct_change().groupby(modes).describe()
    
    cs=zzg.compute_segment_returns(X, pivots)
    pl.plot_cs(cs)
    
    md= zzg.max_drawdown(X)
    zz=zigzag(X, min_rate=change)
    pl.plot_signal(X,zz)
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    main()