"""Plotly figure builders for the dashboard (one function per panel).

Colours follow the validated reference palette: categorical slots in a fixed order, a
blue-grey-red diverging scale for tau, a neutral grey for benchmarks, and separate steps
for the dark theme. Backgrounds are transparent so charts sit on the Streamlit page in
either theme. Every function returns a ``plotly.graph_objects.Figure``.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from vine_risk.dashboard_data import alert_periods
from vine_risk.visualization import vine_tree_graph

THEMES: dict[str, dict] = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], neutral="#898781",
                  div_lo="#2a78d6", div_mid="#f0efec", div_hi="#e34948", node="#cde2fb"),
    "dark": dict(ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500"], neutral="#898781",
                 div_lo="#3987e5", div_mid="#383835", div_hi="#e66767", node="#184f95"),
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def tokens(theme: str) -> dict:
    return THEMES.get(theme, THEMES["light"])


def _style(fig: go.Figure, theme: str, ytitle: str | None = None, height: int = 330,
           right_margin: int = 20, legend: bool = True) -> go.Figure:
    t = tokens(theme)
    fig.update_layout(
        height=height, margin=dict(l=55, r=right_margin, t=30, b=40), hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=t["ink2"]),
        showlegend=legend, legend=dict(orientation="h", y=1.12, x=0, font=dict(color=t["ink"])),
        hoverlabel=dict(font=dict(family=FONT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=t["base"], tickcolor=t["base"], tickfont=dict(color=t["muted"]))
    fig.update_yaxes(gridcolor=t["grid"], gridwidth=1, zeroline=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=t["muted"]), title=dict(text=ytitle, font=dict(color=t["ink2"], size=11)))
    return fig


def _end_labels(fig: go.Figure, theme: str, last: Mapping[str, tuple], x_start=None, log: bool = False) -> None:
    """Direct labels at the right end of each line, in ink (not series colour).

    Labels that would overlap are pushed apart vertically, and the x axis is pinned to the
    data so the labels live in the right margin instead of stretching the axis. On a log
    axis (``log=True``) annotations are placed in log10 units, as plotly requires.
    """
    if not last:
        return
    f = (lambda v: math.log10(v)) if log else float
    last = {k: (x, f(y)) for k, (x, y) in last.items()}
    ys = [float(y) for _, y in last.values()]
    every = [f(v) for tr in fig.data if tr.y is not None for v in tr.y
             if v is not None and np.isfinite(v) and (v > 0 or not log)]
    span = (max(every) - min(every)) if every else (max(ys) - min(ys))
    gap = 0.05 * (span or abs(max(ys)) or 1.0)
    placed: dict[str, float] = {}
    prev = -math.inf
    for name, (_, y) in sorted(last.items(), key=lambda kv: float(kv[1][1])):
        prev = max(float(y), prev + gap)
        placed[name] = prev
    shift = (np.mean(list(placed.values())) - np.mean(ys)) if placed else 0.0
    for name, (x, _) in last.items():
        fig.add_annotation(x=x, y=placed[name] - shift, text=name, showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=tokens(theme)["ink"]))
    x_end = max(x for x, _ in last.values())
    if x_start is not None:
        fig.update_xaxes(range=[x_start, x_end])


def _line(x, y, name: str, color: str, dash: str | None = None, width: float = 2.0, fmt: str = ".3f",
          unit: str = "") -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash),
                      hovertemplate=f"%{{y:{fmt}}}{unit}", connectgaps=False)


# --- panel 1 -----------------------------------------------------------------------
def prices_figure(frame: pd.DataFrame, theme: str, ytitle: str, log: bool = False) -> go.Figure:
    """Selected assets over time: overlaid for up to 3, otherwise one small chart each.
    ``log=True`` uses a logarithmic axis for the overlaid version (rebased prices can differ by
    orders of magnitude); the small multiples have a scale of their own each."""
    t = tokens(theme)
    cols = list(frame.columns)
    if len(cols) <= 3:
        fig = go.Figure([_line(frame.index, frame[c], c, t["series"][i]) for i, c in enumerate(cols)])
        _style(fig, theme, ytitle, right_margin=60)
        _end_labels(fig, theme, {c: (frame.index[-1], frame[c].iloc[-1]) for c in cols}, frame.index[0], log)
        if log:
            fig.update_yaxes(type="log", tickformat="~g")
        return fig
    rows = math.ceil(len(cols) / 2)
    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols, shared_xaxes=True, vertical_spacing=0.08)
    for i, c in enumerate(cols):
        fig.add_trace(_line(frame.index, frame[c], c, t["series"][0], width=1.4), row=i // 2 + 1, col=i % 2 + 1)
    _style(fig, theme, None, height=max(330, 150 * rows), legend=False)  # one chart per asset: own scale
    fig.update_annotations(font=dict(size=11, color=t["ink"]))
    return fig


# --- panel 2 -----------------------------------------------------------------------
def dependence_figure(metrics: pd.DataFrame, theme: str, extras: bool = False) -> go.Figure:
    """Average absolute Kendall tau ``D_t`` (optionally with mean Pearson and lower tail)."""
    t = tokens(theme)
    series = {"D_t (mean |tau|)": metrics["d_t"]}
    if extras:
        series["mean Pearson"] = metrics["mean_pearson"]
        series["mean lower tail (5%)"] = metrics["mean_lower_tail_q"]
    fig = go.Figure([_line(s.index, s, n, t["series"][i]) for i, (n, s) in enumerate(series.items())])
    _style(fig, theme, "dependence", right_margin=150 if extras else 20, legend=extras or False)
    if extras:
        _end_labels(fig, theme, {n: (s.dropna().index[-1], s.dropna().iloc[-1]) for n, s in series.items()},
                    metrics.index[0])
    return fig


# --- panel 3 -----------------------------------------------------------------------
def tau_heatmap(matrix: pd.DataFrame, date, theme: str, vmax: float = 0.8) -> go.Figure:
    """Kendall-tau matrix on one date; diverging blue-grey-red scale centred on 0."""
    t = tokens(theme)
    scale = [[0.0, t["div_lo"]], [0.5, t["div_mid"]], [1.0, t["div_hi"]]]
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index), zmin=-vmax, zmax=vmax,
        colorscale=scale, texttemplate="%{z:.2f}", textfont=dict(size=11, color=t["ink"]),
        hovertemplate="%{y} – %{x}: tau = %{z:.3f}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(tickfont=dict(color=t["muted"]), thickness=12),
    ))
    _style(fig, theme, None, height=420, legend=False)
    fig.update_layout(hovermode="closest", title=dict(text=f"Kendall tau on {pd.Timestamp(date).date()}",
                                                      font=dict(size=12, color=t["ink2"]), x=0, y=0.98))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=t["ink2"]))
    fig.update_xaxes(tickfont=dict(color=t["ink2"]), side="bottom")
    return fig


# --- panels 4 and 5 ----------------------------------------------------------------
def pair_figure(series: pd.Series, label: str, theme: str) -> go.Figure:
    """Kendall's tau of one pair through time."""
    t = tokens(theme)
    fig = go.Figure(_line(series.index, series, f"tau {label}", t["series"][0]))
    fig.add_hline(y=0, line=dict(color=t["base"], width=1))
    return _style(fig, theme, "Kendall tau", legend=False)


def tail_figure(lower: pd.Series, upper: pd.Series, label: str, theme: str, q: float = 0.05) -> go.Figure:
    """Model-implied lower and upper tail dependence through time (finite level ``q``)."""
    t = tokens(theme)
    fig = go.Figure([_line(lower.index, lower, "lower tail", t["series"][0]),
                     _line(upper.index, upper, "upper tail", t["series"][1])])
    _style(fig, theme, f"P(both in {q:.0%} tail | one is)", right_margin=95)
    _end_labels(fig, theme, {"lower tail": (lower.dropna().index[-1], lower.dropna().iloc[-1]),
                             "upper tail": (upper.dropna().index[-1], upper.dropna().iloc[-1])}, lower.index[0])
    return fig


# --- panel 6 -----------------------------------------------------------------------
def _tree_layout(nodes: list[str], edges: list) -> dict[str, tuple[float, float]]:
    """Deterministic force-directed layout (hubs end up in the centre), scaled to [-1, 1]."""
    import networkx as nx

    n = len(nodes)
    if n <= 3:
        return {nd: (math.cos(2 * math.pi * k / n + math.pi / 2), math.sin(2 * math.pi * k / n + math.pi / 2))
                for k, nd in enumerate(nodes)}
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from((u, v) for u, v, _ in edges)
    raw = nx.kamada_kawai_layout(g)
    xs = np.array([raw[nd][0] for nd in nodes])
    ys = np.array([raw[nd][1] for nd in nodes])
    scale = max(np.abs(xs).max(), np.abs(ys).max()) or 1.0
    return {nd: (float(x / scale), float(y / scale)) for nd, x, y in zip(nodes, xs, ys)}


def vine_tree_figure(result, tree: int, theme: str) -> go.Figure:
    """One tree of the fitted vine: nodes on a circle, edges labelled by family on hover."""
    t = tokens(theme)
    nodes, edges = vine_tree_graph(result, tree)
    pos = _tree_layout(nodes, edges)
    ex, ey = [], []
    for u, v, _ in edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=t["ink2"], width=1.5), hoverinfo="skip"))
    mid = [((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2) for u, v, _ in edges]
    fig.add_trace(go.Scatter(
        x=[m[0] for m in mid], y=[m[1] for m in mid], mode="markers+text",
        text=[f"{p.family}<br>{p.tau:.2f}" for _, _, p in edges] if len(edges) <= 5 else None,
        textfont=dict(size=10, color=t["ink2"]),
        textposition="top center", marker=dict(size=14, color="rgba(0,0,0,0)"),
        hovertext=[f"<b>{u} – {v}</b><br>{p.family} (rotation {p.rotation})<br>tau {p.tau:.3f}"
                   f"<br>lower tail {p.lower_tail:.3f}, upper tail {p.upper_tail:.3f}"
                   + (f"<br>given {', '.join(p.conditioning)}" if p.conditioning else "")
                   for u, v, p in edges], hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=[pos[k][0] for k in nodes], y=[pos[k][1] for k in nodes], mode="markers+text", text=nodes,
        textfont=dict(size=11, color=t["ink"]), textposition="middle center",
        marker=dict(size=46 if tree == 1 else 72, color=t["node"], line=dict(color=t["ink2"], width=1)),
        hoverinfo="skip"))
    _style(fig, theme, None, height=430, legend=False)
    fig.update_layout(hovermode="closest", title=dict(text=f"Tree {tree} on {result.timestamp}",
                                                      font=dict(size=12, color=t["ink2"]), x=0, y=0.98))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor="x")
    return fig


# --- panel 7 -----------------------------------------------------------------------
def change_score_figure(score: pd.Series, threshold: float, label: str, theme: str) -> go.Figure:
    """Structural-change score with the periods above ``threshold`` shaded."""
    t = tokens(theme)
    fig = go.Figure(_line(score.index, score, label, t["series"][0]))
    periods = alert_periods(score, threshold)
    for a, b in periods:
        fig.add_vrect(x0=a, x1=b, fillcolor=t["series"][1], opacity=0.18, line_width=0, layer="below")
    fig.add_hline(y=threshold, line=dict(color=t["ink2"], width=1, dash="dash"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="threshold",
                             line=dict(color=t["ink2"], width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="above threshold",
                             marker=dict(size=10, color=t["series"][1], symbol="square", opacity=0.4)))
    _style(fig, theme, label)
    return fig


# --- panel 8 -----------------------------------------------------------------------
RISK_MODELS = [("vine", "vine copula"), ("gauss", "Gaussian copula"), ("indep", "independence"),
               ("hist", "historical")]


def risk_figure(risk: pd.DataFrame, measure: str, theme: str, models: Sequence[str] | None = None) -> go.Figure:
    """Rolling portfolio VaR or Expected Shortfall (in % of portfolio value) per model."""
    if measure not in ("var", "es"):
        raise ValueError("measure must be 'var' or 'es'.")
    t = tokens(theme)
    chosen = [(k, n) for k, n in RISK_MODELS if models is None or k in models]
    traces, last = [], {}
    for i, (k, name) in enumerate(chosen):
        s = 100 * risk[f"{measure}_{k}"]
        grey = k == "hist"
        traces.append(_line(s.index, s, name, t["neutral"] if grey else t["series"][min(i, 2)],
                            dash="dash" if grey else None, width=1.5 if grey else 2.0, fmt=".2f", unit="%"))
        last[name] = (s.index[-1], s.iloc[-1])
    fig = go.Figure(traces)
    _style(fig, theme, f"99% {'Expected Shortfall' if measure == 'es' else 'VaR'} (% of value)", right_margin=125)
    _end_labels(fig, theme, last, risk.index[0])
    return fig
