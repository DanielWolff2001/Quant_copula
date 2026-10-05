"""Streamlit dashboard for a finished rolling vine-copula run.

Run with:   streamlit run dashboard/app.py

Environment variables (optional): VINE_RISK_RUN_DIR (default data/results/w250) and
VINE_RISK_CONFIG (default configs/default.yaml). The dashboard only reads files written
by the scripts in ``scripts/``; panels whose file is missing say which script to run.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from vine_risk import dashboard_figures as figs
from vine_risk.config import load_config
from vine_risk.dashboard_data import (
    CheckpointIndex, alert_periods, load_run, nearest_fit_date, pair_column, prices_from_returns, tau_matrix_at,
)
from vine_risk.pipeline import load_prices_and_returns

CONFIG = os.environ.get("VINE_RISK_CONFIG", "configs/default.yaml")
RUN_DIR = os.environ.get("VINE_RISK_RUN_DIR", "data/results/w250")

st.set_page_config(page_title="Vine copula risk monitor", layout="wide")


def chart(fig) -> None:
    # theme=None keeps the validated palette instead of Streamlit's own plotly colours
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


@st.cache_data(show_spinner="Loading run ...")
def get_run(run_dir: str, config_path: str):
    cfg = load_config(config_path)
    try:  # prices come from the local cache written by the rolling run; optional for the dashboard
        prices, returns = load_prices_and_returns(cfg)
    except Exception:
        prices = returns = None
    return load_run(run_dir, prices, returns)


@st.cache_resource(show_spinner=False)
def get_index(run_dir: str) -> CheckpointIndex:
    return CheckpointIndex(f"{run_dir}/checkpoint.jsonl")


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except Exception:
        return "light"


def panel(title: str, caption: str) -> None:
    st.subheader(title)
    st.caption(caption)


def need(script: str) -> None:
    st.info(f"Not available for this run. Create the data with `{script}`.")


# ---------------------------------------------------------------- sidebar / loading
st.sidebar.header("Run")
run_dir = st.sidebar.text_input("Results folder", RUN_DIR)
try:
    run = get_run(run_dir, CONFIG)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

TH = theme()
dates = run.metrics.index
d0, d1 = dates[0].date(), dates[-1].date()
start, end = st.sidebar.slider("Date range", d0, d1, (d0, d1), format="YYYY-MM-DD")
start, end = pd.Timestamp(start), pd.Timestamp(end)
tables = st.sidebar.checkbox("Show data tables", value=False,
                             help="Every chart can be read as a table, too.")


def within(df: pd.DataFrame | None) -> pd.DataFrame | None:
    return None if df is None else df.loc[start:end]


metrics, tau = within(run.metrics), within(run.tau)
lower, upper = within(run.lower), within(run.upper)

st.title("Dynamic vine copula risk monitor")
st.caption(f"{len(run.assets)} assets, {run.window}-day rolling window, {dates[0].date()} to {dates[-1].date()}. "
           "Associations between dependence and risk, not causal claims; changes flagged here are "
           "statistical, not necessarily economic regime changes.")

# ---------------------------------------------------------------- key figures
k1, k2, k3, k4 = st.columns(4)
last = metrics.iloc[-1]
k1.metric("Latest fit", str(metrics.index[-1].date()))
k2.metric("Average dependence D_t", f"{last['d_t']:.3f}",
          None if len(metrics) < 250 else f"{last['d_t'] - metrics['d_t'].iloc[-250]:+.3f} vs 250 fits ago")
k3.metric("Mean lower tail (5%)", f"{last['mean_lower_tail_q']:.3f}")
risk = within(run.risk)
k4.metric("99% ES, vine copula", "n/a" if risk is None or risk.empty else f"{100 * risk['es_vine'].iloc[-1]:.2f}%")

# ---------------------------------------------------------------- panels 1 and 2
c1, c2 = st.columns(2)
with c1:
    panel("1. Asset prices and returns", "Selected assets over the chosen dates.")
    chosen = st.multiselect("Assets", run.assets, default=run.assets[:3])
    mode = st.radio("Show", ["Rebased prices", "Daily log returns"], horizontal=True)
    if run.returns is None:
        need("python scripts/run_rolling.py (prices are read from the cache in data/cache)")
    elif chosen:
        r = run.returns.loc[start:end, chosen]
        frame = prices_from_returns(r) if mode == "Rebased prices" else r
        rebased = mode == "Rebased prices"
        chart(figs.prices_figure(frame, TH, "start = 1 (log scale)" if rebased else "log return", log=rebased))
        if tables:
            st.dataframe(frame)
with c2:
    panel("2. Rolling dependence", "Average absolute Kendall tau across all pairs, D_t, estimated on each window.")
    extras = st.checkbox("Also show average Pearson correlation and lower-tail dependence")
    chart(figs.dependence_figure(metrics, TH, extras))
    if tables:
        st.dataframe(metrics[["d_t", "mean_pearson", "mean_lower_tail_q"]])

# ---------------------------------------------------------------- panels 3 and 4
c3, c4 = st.columns(2)
with c3:
    panel("3. Pairwise dependence on a date", "Kendall tau between all pairs, from the window ending on the chosen date.")
    sel = st.select_slider("Date", options=list(tau.index.date), value=tau.index.date[-1])
    mat = tau_matrix_at(run.tau, pd.Timestamp(sel), run.assets)
    chart(figs.tau_heatmap(mat, sel, TH))
    if tables:
        st.dataframe(mat)
with c4:
    panel("4. Dependence evolution of a pair", "Kendall tau of two assets through time.")
    ca, cb = st.columns(2)
    a = ca.selectbox("Asset A", run.assets, index=0)
    b = cb.selectbox("Asset B", run.assets, index=min(1, len(run.assets) - 1))
    if a == b:
        st.warning("Choose two different assets.")
    else:
        col = pair_column(tau, a, b, run.assets)
        chart(figs.pair_figure(tau[col], f"{a}-{b}", TH))
        if tables:
            st.dataframe(tau[[col]])

# ---------------------------------------------------------------- panels 5 and 6
c5, c6 = st.columns(2)
with c5:
    panel("5. Tail dependence", "Fitted-vine probability that both assets are in their worst (best) 5% days "
          "given that one is, for the pair chosen in panel 4. Finite-level coefficients, so positive even for "
          "Gaussian pairs: compare dates, do not read as absolute truth. Sudden jumps usually mean the "
          "fitted copula family changed.")
    if lower is None or upper is None:
        need("python scripts/compute_metrics.py")
    elif a != b:
        col = pair_column(lower, a, b, run.assets)
        chart(figs.tail_figure(lower[col], upper[col], f"{a}-{b}", TH))
        if tables:
            st.dataframe(pd.DataFrame({"lower": lower[col], "upper": upper[col]}))
with c6:
    panel("6. Vine structure", "The fitted vine on the date chosen in panel 3. Hover an edge for the "
          "pair-copula family, tau and tail dependence; tree 2 and up are conditional dependencies.")
    try:
        index = get_index(run_dir)
        fit_date = nearest_fit_date(index.timestamps, pd.Timestamp(sel))
        fit = index.get(fit_date)
        n_trees = max(p.tree for p in fit.pair_copulas)
        tree = st.number_input("Tree", 1, n_trees, 1)
        chart(figs.vine_tree_figure(fit, int(tree), TH))
        if tables:
            st.dataframe(fit.pair_copula_frame())
    except (FileNotFoundError, KeyError, ValueError) as e:
        st.info(f"Vine structure not available: {e}")

# ---------------------------------------------------------------- panels 7 and 8
c7, c8 = st.columns(2)
with c7:
    panel("7. Structural-change score", "How different the dependence is from one window earlier. "
          "Shaded periods exceed the threshold, a visual aid and not a significance level "
          "(p-values with multiple-testing control are in change_scan.parquet).")
    scores, scan = within(run.scores), within(run.scan)
    options: dict[str, pd.Series] = {}
    if scores is not None:
        options["Tau-matrix distance to one window ago (S_t)"] = scores["structural_change_score"]
        options["Model-implied distance (tau and tails)"] = scores["dist_model"]
    if scan is not None:
        options["Permutation test, tau: distance ÷ no-change distance"] = scan["stat_tau"] / scan["null_mean_tau"]
        options["Permutation test, lower tail: distance ÷ no-change distance"] = (
            scan["stat_mean_lower"] / scan["null_mean_mean_lower"])
    if not options:
        need("python scripts/detect_changes.py")
    else:
        label = st.selectbox("Score", list(options))
        s = options[label].dropna()
        default = 2.0 if "÷" in label else float(s.quantile(0.9))
        thr = st.number_input("Threshold", value=round(default, 3), key=f"thr_{label}", format="%.3f")
        chart(figs.change_score_figure(s, thr, label, TH))
        periods = alert_periods(s, thr)
        st.caption(f"{len(periods)} period(s) above the threshold.")
        if scan is not None and "p_tau_fdr0.01" in scan:
            st.caption(f"Permutation test: tau matrix differs significantly (1% false-discovery rate) on "
                       f"{scan['p_tau_fdr0.01'].mean():.0%} of scan dates in this range.")
        if tables:
            st.dataframe(s.to_frame("score"))
with c8:
    panel("8. Portfolio risk", "Equal-weight portfolio, 99% level. The vine copula is compared with a Gaussian "
          "copula, independence and the window's own history, all with the same marginals.")
    if risk is None:
        need("python scripts/compute_risk.py")
    else:
        measure = st.radio("Measure", ["Expected Shortfall", "Value at Risk"], horizontal=True)
        keys = [k for k, _ in figs.RISK_MODELS]
        names = dict(figs.RISK_MODELS)
        models = st.multiselect("Models", keys, default=keys, format_func=names.get)
        if models:
            chart(figs.risk_figure(risk, "es" if measure == "Expected Shortfall" else "var", TH, models))
        if tables:
            st.dataframe(risk)
        if run.backtest is not None:
            with st.expander("VaR backtest (does next day's loss exceed the VaR?)"):
                st.dataframe(run.backtest)

# ---------------------------------------------------------------- panel 9: live replay
st.divider()
panel("9. Live monitor replay", "A replay of recent history one observation at a time, as a live system would "
      "see it (scripts/run_live.py). It reproduces the batch results exactly; alerts start when the "
      "permutation test is significant and the effect is large, and end when the effect fades.")
live = within(run.live)
if live is None or live.empty:
    need("python scripts/run_live.py")
else:
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Observations replayed", len(live))
    l2.metric("Refits", int(live["refit"].sum()))
    l3.metric("New alerts", int(live["new_alert"].sum()))
    l4.metric("State at the end", str(live["alert_state"].iloc[-1]))
    if live["scan_ratio_tau"].notna().any():
        chart(figs.live_figure(live, TH))
    alerts = live[live["new_alert"]][["alert_state", "message"]]
    if len(alerts):
        st.dataframe(alerts)
    if tables:
        st.dataframe(live)
