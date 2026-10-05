"""Figures for the README and the documentation, from a finished run.

Usage: python scripts/make_doc_figures.py [--run data/results/w250] [--out docs/assets]

Writes ``dependence-evolution.png`` and ``portfolio-es.png``. (``detection.png`` comes from
``scripts/plot_validation.py``.) Colours are the validated categorical palette, in fixed order.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898883"
EPISODES = [("2008-09-15", "2009-06-30", "2008-09 to 2009-06"), ("2020-02-20", "2020-06-30", "2020-02 to 2020-06")]


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=9)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def _end_labels(ax, series: dict[str, tuple], gap: float) -> None:
    """Direct labels in the right margin, pushed apart vertically so they never overlap."""
    prev = float("-inf")
    for name, (x, y) in sorted(series.items(), key=lambda kv: kv[1][1]):
        prev = max(prev + gap, y)
        ax.annotate(name, xy=(x, prev), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK, annotation_clip=False)


def _legend_above(ax) -> None:
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, handlelength=1.6,
              borderaxespad=0.2)


def dependence_figure(run: Path, out: Path) -> None:
    m = _read(run / "dependence_metrics.parquet")
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(m.index, m.d_t, color=BLUE, lw=1.6, label="D_t, average |Kendall tau|")
    ax.plot(m.index, m.mean_lower_tail_q, color=ORANGE, lw=1.3, label="average lower-tail dependence (5 %)")
    for a, b, _ in EPISODES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=GREY, alpha=0.18, lw=0)
    lo = ax.get_ylim()[0]
    ax.text(pd.Timestamp("2008-12-01"), lo + 0.004, "2008 crisis", fontsize=8, color=MUTED, ha="center", va="bottom")
    ax.text(pd.Timestamp("2020-04-15"), lo + 0.004, "2020 crash", fontsize=8, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(m.index[0], m.index[-1])
    _end_labels(ax, {"D_t": (m.index[-1], m.d_t.iloc[-1]), "lower tail": (m.index[-1], m.mean_lower_tail_q.iloc[-1])}, 0.025)
    ax.set_ylabel("dependence", color=MUTED, fontsize=9)
    ax.set_title("Dependence between eight assets on a rolling 250-day window", loc="left", fontsize=11, color=INK, pad=26)
    _legend_above(ax)
    _style(ax)
    fig.text(0.01, 0.01, "Shaded: the 2008 crisis and the 2020 crash. An association in this sample, not a causal claim.",
             fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.84, bottom=0.12)
    fig.savefig(out / "dependence-evolution.png", dpi=130, facecolor="white")


def risk_figure(run: Path, out: Path) -> None:
    r = 100 * _read(run / "portfolio_risk.parquet")[["es_vine", "es_gauss", "es_indep", "es_hist"]]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    names = {"es_vine": "vine copula", "es_gauss": "Gaussian copula", "es_indep": "independence", "es_hist": "historical"}
    for col, color, ls in (("es_vine", BLUE, "-"), ("es_gauss", ORANGE, "-"), ("es_indep", AQUA, "-"), ("es_hist", GREY, "--")):
        ax.plot(r.index, r[col], color=color, lw=1.5, ls=ls, label=names[col])
    ax.set_xlim(r.index[0], r.index[-1])
    _end_labels(ax, {names[c]: (r.index[-1], r[c].iloc[-1]) for c in r}, 0.5)
    ax.set_ylabel("99 % Expected Shortfall, % of portfolio value", color=MUTED, fontsize=9)
    ax.set_title("Tail risk of an equal-weight portfolio under four dependence models (same marginals)",
                 loc="left", fontsize=11, color=INK, pad=26)
    _legend_above(ax)
    _style(ax)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.84, bottom=0.1)
    fig.savefig(out / "portfolio-es.png", dpi=130, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="data/results/w250")
    ap.add_argument("--out", default="docs/assets")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dependence_figure(Path(a.run), out)
    risk_figure(Path(a.run), out)
    print(f"-> {out}/dependence-evolution.png, {out}/portfolio-es.png")


if __name__ == "__main__":
    main()
