"""Basic plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_prices_returns(prices: pd.DataFrame, returns: pd.DataFrame):
    """Normalised prices (top) and daily log returns (bottom)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    (prices / prices.iloc[0]).plot(ax=ax1, logy=True, lw=1)
    ax1.set_title("Adjusted prices (rebased to 1)")
    returns.plot(ax=ax2, lw=0.5, legend=False)
    ax2.set_title("Daily log returns")
    fig.tight_layout()
    return fig


def plot_uniformity(u: pd.DataFrame):
    """Histogram of each pseudo-observation column; should be flat on (0, 1)."""
    n = u.shape[1]
    ncols = min(4, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows), squeeze=False)
    for ax, c in zip(axes.ravel(), u.columns):
        ax.hist(u[c], bins=20, range=(0, 1), density=True, color="0.6")
        ax.axhline(1.0, color="C3", lw=1)
        ax.set_title(c)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Pseudo-observations (should be ~Uniform(0,1))")
    fig.tight_layout()
    return fig
