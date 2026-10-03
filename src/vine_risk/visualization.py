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


def plot_vine_tree(result, tree: int = 1, ax=None):
    """Draw one tree of the fitted vine.

    Tree 1 nodes are assets. For ``tree > 1`` the nodes are the edges of the previous
    tree (labelled ``conditioned|conditioning``), and each edge is a pair-copula,
    labelled with its family and Kendall's tau.
    """
    import networkx as nx

    pcs = [p for p in result.pair_copulas if p.tree == tree]
    if not pcs:
        raise ValueError(f"Vine has no tree {tree}.")

    def label(cond, given):
        return f"{cond[0]},{cond[1]}" + (f"|{','.join(given)}" if given else "")

    g = nx.Graph()
    if tree == 1:
        g.add_nodes_from(result.assets)
        for p in pcs:
            g.add_edge(*p.conditioned, text=f"{p.family}\n{p.tau:.2f}")
    else:
        prev = [p for p in result.pair_copulas if p.tree == tree - 1]
        members = {label(q.conditioned, q.conditioning): set(q.conditioned) | set(q.conditioning)
                   for q in prev}
        g.add_nodes_from(members)
        for p in pcs:
            a, b = p.conditioned
            c = set(p.conditioning)
            ends = [n for n, m in members.items() if m in ({a} | c, {b} | c)]
            if len(ends) == 2:
                g.add_edge(*ends, text=f"{p.family}\n{p.tau:.2f}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    pos = nx.circular_layout(g)
    nx.draw_networkx(g, pos, ax=ax, node_color="#cfe3f5", node_size=1100, font_size=8)
    nx.draw_networkx_edge_labels(g, pos, ax=ax, font_size=7,
                                 edge_labels={(u, v): d["text"] for u, v, d in g.edges(data=True)})
    ax.set_title(f"Tree {tree}")
    ax.axis("off")
    return ax.figure
