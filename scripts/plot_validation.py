"""Figure for the synthetic validation study (reads the raw tables of run_validation.py).

Usage: python scripts/plot_validation.py [--raw data/results/validation]
       [--out reports/figures/phase8_detection.png] [--window 250]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from vine_risk.validation import standard_experiments

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
# categorical slots 1-3 of the reference palette (validated), fixed order; the null is neutral gray
SERIES = [("B_corr", "correlation 0.3 → 0.7", "#2a78d6"),
          ("C_tail_moderate", "tails only: Gaussian → t(3)", "#eb6834"),
          ("C_tail_strong", "tails only: Gaussian → t(1)", "#1baf7a")]
NULL = ("A_vol", "no change (clustered volatility)", "#898883")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/results/validation")
    ap.add_argument("--out", default="reports/figures/phase8_detection.png")
    ap.add_argument("--window", type=int, default=250)
    a = ap.parse_args()
    exps = standard_experiments()
    c, w = exps["B_corr"].change_at, a.window

    panels = [("stat_tau", "null_mean_tau", "Kendall-tau matrix"),
              ("stat_lower", "null_mean_lower", "Lower-tail dependence")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (stat, null, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for name, label, color in [NULL] + SERIES:
            df = pd.read_parquet(Path(a.raw) / f"scan_{name}.parquet")
            m = (df[stat] / df[null]).groupby(df["pos"]).mean()
            x = m.index - c
            is_null = name == NULL[0]
            ax.plot(x, m.values, color=color, lw=1.4 if is_null else 2.0, zorder=2 if is_null else 3)
        ax.axhline(1.0, color=GRID, lw=1, zorder=1)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.axvline(w, color=GRID, lw=1, ls=(0, (1, 3)), zorder=1)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("days since the change", color=MUTED, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("distance between windows ÷ typical no-change distance\n(mean over 20 simulated histories)",
                       color=MUTED, fontsize=9)
    axes[0].set_ylim(0.4, 5.0)
    for ax in axes:
        ax.text(6, 0.45, "change", color=MUTED, fontsize=8, va="bottom")
        ax.text(w + 6, 0.45, "windows split at the change", color=MUTED, fontsize=8, va="bottom")
    handles = [plt.Line2D([], [], color=col, lw=2) for _, _, col in SERIES + [NULL]]
    fig.legend(handles, [lab for _, lab, _ in SERIES + [NULL]], loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Permutation scan: correlation jumps are detected clearly; tail-only changes barely register",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor=SURFACE)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
