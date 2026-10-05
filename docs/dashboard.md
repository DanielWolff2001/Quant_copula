# Dashboard

```bash
pip install -e ".[dashboard]"            # adds streamlit and plotly (once)
vine-risk dashboard                      # opens http://localhost:8501   (same as: streamlit run dashboard/app.py)
```

It reads the files that the pipeline (`vine-risk run`) wrote into `data/results/w250/` (set another
folder in the sidebar, or with the environment variable `VINE_RISK_RUN_DIR`). It needs at
least the rolling fits and the metrics step; a panel whose file is missing
says which script to run instead of failing.

| Panel | Shows | Needs |
|-------|-------|-------|
| 1 Asset prices / returns | selected assets, rebased prices on a log scale or daily log returns | `vine-risk run` (prices come from the local cache) |
| 2 Rolling dependence | average absolute Kendall tau D_t, optionally with Pearson and lower tail | `vine-risk run` |
| 3 Pairwise dependence | heatmap of Kendall tau between all pairs on a chosen date | `vine-risk run` |
| 4 Dependence evolution | Kendall tau of two chosen assets through time | `vine-risk run` |
| 5 Tail dependence | the fitted vine's lower and upper tail dependence for the same pair | `vine-risk run` |
| 6 Vine structure | the fitted vine on the chosen date, tree by tree; hover an edge for family, tau and tail dependence | `vine-risk run` |
| 7 Structural-change score | one of four scores, with the periods above an adjustable threshold shaded | `vine-risk run` |
| 8 Portfolio risk | rolling 99% Expected Shortfall or VaR for the vine copula and three benchmarks, plus the VaR backtest | `vine-risk run` |
| 9 Live monitor replay | the alert history of a simulated live run | `vine-risk replay` or `vine-risk update` |

Charts are interactive (hover for exact values) and follow the light or dark theme of
Streamlit. "Show data tables" in the sidebar lists the numbers behind every chart. The threshold
in panel 7 only shades periods on screen: it is a visual aid and not a significance
level, so use the p-values in `change_scan.parquet` ([structural change detection](methodology.md)) for decisions.
