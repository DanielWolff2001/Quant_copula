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
