# Notebooks

The quickest way to see what the package does is to run the notebooks in `notebooks/`. A *notebook* is a document in which
text and code alternate; you run the code cell by cell and see the tables and plots appear below it.

```bash
pip install -e ".[notebooks]"      # once, adds JupyterLab (not needed if you installed all extras)
jupyter lab                        # opens a browser; open a notebook from the notebooks/ folder
```

Run them in order. Each starts with what it teaches and what it needs, and every code cell has been executed, so you can also
read them on GitHub without running anything.

| Notebook | You learn | Needs | Time |
|----------|-----------|-------|------|
| `01_data_exploration` | settings, prices, cleaning, log returns, why returns are not normal; using your own tickers | internet once (prices are cached) | 1 min |
| `02_static_vine` | marginal transform, fitting one vine, reading families, tau and tail dependence, tree plots, simulation, model options | notebook 1 | 1 min |
| `03_rolling_vine` | the rolling window, what is stored per date, stepwise versus batch, window-length sensitivity; **creates a small demo run** in `data/results/demo/` | notebook 2 | 2-3 min |
| `04_dependence_changes` | dependence over time, which pairs moved, structure churn, why one-day comparisons fail, the permutation test, alerts, simulated experiments with a known answer | demo run or full run | 2 min |
| `05_portfolio_risk` | VaR and Expected Shortfall, vine versus simpler models, your own weights, the backtest, dependence versus volatility | demo run or full run | 1 min |
| `06_live_monitor` | feeding prices one day at a time, resuming from saved state, checking against the batch run, the alert rule | demo run or full run | 1 min |

Notebooks 4 to 6 use the full 20-year run (`data/results/w250/`) if it exists and the small demo run otherwise, so they work
either way; with the full run the plots cover 2006-2026. To make the full run: `vine-risk run` (about an hour and a quarter; see [Command line](cli.md)).

---

## Using the package from Python (short version)


```python
from vine_risk.config import load_config
from vine_risk.data import download_prices
from vine_risk.returns import build_return_matrix
from vine_risk.marginals import EmpiricalMarginal
from vine_risk.copula import VineCopula

cfg = load_config("configs/default.yaml")
prices = download_prices(cfg.assets, cfg.data.start, cfg.data.end, cfg.data.cache_dir)
returns = build_return_matrix(prices)                 # T x d table of log returns

window = returns.iloc[-250:]                          # the last 250 days
u = EmpiricalMarginal().fit_transform(window)         # uniform data

result = VineCopula(truncation_level=3).fit(u, window).summary()
print(result.pair_copula_frame())                     # families, parameters, tail dependence
print(result.tau_matrix())                            # Kendall's tau between all assets
```

Rolling version (use inside an `if __name__ == "__main__":` block when `n_jobs > 1`):

```python
from vine_risk.rolling import RollingVineModel, save_results

model = RollingVineModel.from_config(cfg)
results = model.run(returns, n_jobs=4, checkpoint="data/results/run.jsonl")
save_results(results, "data/results/run")
```
