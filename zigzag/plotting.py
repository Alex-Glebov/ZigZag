"""
The package requirements do not enforce matplotlib as a requirement so this
package is optional. However, it's useful.
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_pivots(X, pivots, ax=None):
    if ax is None:
        plt.figure("Pivots")
        ax = plt.subplots(1, 1)[1]

    ax.set_xlim(0, len(X))
    ax.set_ylim(X.min()*0.99, X.max()*1.01)
    ax.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    ax.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    ax.scatter(np.arange(len(X))[pivots ==  1], X[pivots ==  1], color='g')
    ax.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
    ax.scatter(np.arange(len(X))[pivots ==  0], X[pivots ==  0], color='gray')

def plot_signal(X, signal, ax=None):
    if ax is None:
        plt.figure("Signal")
        ax = plt.subplots(1, 1)[1]

    ax.set_xlim(0, len(X))
    ax.set_ylim(X.min()*0.99, X.max()*1.01)
    ax.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    ax.scatter(np.arange(len(X)), signal, color='r')

def plot_cs(cs, ax=None):
    if ax is None:
        plt.figure("CS")
        ax = plt.subplots(1, 1)[1]
    plt.xlim(0, len(cs))
    plt.ylim(cs.min()*0.99, cs.max()*1.01)
    plt.plot(np.arange(len(cs))[cs >  0], cs[cs >  0], 'g-')
    plt.plot(np.arange(len(cs))[cs <  0], cs[cs <  0], 'k-')
#    plt.scatter(np.arange(len(cs))[pivots == 1], cs[pivots == 1], color='g')
#    plt.scatter(np.arange(len(cs))[pivots == -1], cs[pivots == -1], color='r')
#    plt.scatter(np.arange(len(cs))[pivots ==  0], cs[pivots ==  0], color='gray')
def plot_dX(dX,limit=0.01, ax=None):
    if ax is None:
        plt.figure("dX")
        ax = plt.subplots(1, 1)[1]
    ldx = len(dX)
    plt.figure(1)
    plt.xlim(0, ldx )
    plt.ylim(dX.min()*0.99, dX.max()*1.01)
    plt.plot(np.arange(ldx), dX, 'k:', alpha=0.5)
    pivots=np.zeros(ldx)
    inx =0
    for nx in range(0,ldx):
        if abs(dX[nx]-dX[inx]) > limit :
            inx = nx
            pivots[inx]=1
            
    plt.scatter(np.arange(ldx)[pivots == 1], dX[pivots == 1], color='g')
    plt.scatter(np.arange(ldx)[pivots == 0], dX[pivots == 0], color='r')
#    plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
