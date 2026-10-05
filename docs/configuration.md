# Configuration

All settings live in `configs/default.yaml` (YAML: indented `key: value` lines). Use another file with `--config`. Keys that are
left out take the defaults shown here.

```yaml
assets: [AAPL, MSFT, NVDA, JPM, XOM, JNJ, SPY, TLT]

data:
  frequency: daily            # only daily data is supported
  start: "2005-01-01"
  end: null                   # null = up to the latest day
  cache_dir: data/cache       # local copy of downloaded prices
  max_missing_frac: 0.05      # drop a ticker with more than this share of missing prices
  max_ffill_days: 3           # carry the last price forward over gaps up to this length
  source: yahoo               # yahoo | csv
  csv_path: null              # file or folder when source is csv

rolling:
  window: 250                 # trading days per fit
  refit_frequency: 1          # 1 = refit every day
  truncation_level: 3         # fit only the first 3 trees (null = full vine)
  selection_criterion: bic    # aic | bic | loglik, for choosing pair-copula families
  n_jobs: 4                   # worker processes for the batch run
  tail_simulations: 16384     # simulated points for model-implied tail dependence
  tail_level: 0.05            # q in "probability that both are in the worst q, given that one is"
  marginal: empirical         # empirical (ranks) | garch_t | garch_empirical  (GARCH(1,1)-filtered)
  marginal_lookback: null     # history the marginal is fitted on (null: window; GARCH: 1000)

risk:
  confidence_level: 0.99
  simulations: 16384          # scenarios per window (a power of 2 works best)
  seed: 42                    # seeds every simulation
  weights: null               # null = equal weights; or a list with one weight per asset
```

## Which settings change the fits?

These define the *fits*, and a run folder can only be extended with the same values (see [Reproducibility](reproducibility.md)):
`assets`, `window`, `refit_frequency`, `truncation_level`, `selection_criterion`, `tail_simulations`, `tail_level`, `seed`,
`marginal`, `marginal_lookback`, `max_missing_frac`, `max_ffill_days`. Everything else (`confidence_level`, `weights`, risk `simulations`, `n_jobs`, the scan options)
only affects what is computed *from* the fits.

## Marginal models

The copula describes dependence between *uniform* numbers, so each asset's returns first go through a marginal model.

| `marginal` | What it does |
|------------|--------------|
| `empirical` (default) | the rank transform of the window; no model, robust |
| `garch_t` | a GARCH(1,1) with Student-t innovations per asset; the copula is fitted to the PIT of the standardised residuals |
| `garch_empirical` | the same GARCH(1,1), with the empirical distribution of its residuals instead of a Student-t |

GARCH filtering removes volatility clustering, so the copula describes dependence between *shocks* rather than a mixture of
dependence and market volatility. GARCH parameters are poorly determined by 250 observations, so the marginal is fitted on a longer
history (`marginal_lookback`, default 1000 days) while the vine is still fitted to the last `window` days. The first fit is then
possible only after `marginal_lookback` days. If a GARCH fit fails for an asset, that asset falls back to a RiskMetrics (EWMA)
volatility and the fit is recorded as such.

Scenarios drawn through a GARCH marginal are forecasts for **tomorrow given today's volatility**, while rank marginals describe the
window's unconditional distribution; this matters for risk numbers (see [Risk model comparison](benchmarks.md)).

## Environment variables

| Variable | Used by | Meaning |
|----------|---------|---------|
| `VINE_RISK_RUN_DIR` | dashboard | the run folder to show (default `data/results/w250`) |
| `VINE_RISK_CONFIG` | dashboard | the config file (default `configs/default.yaml`) |

`vine-risk dashboard` sets both for you.

## Monitor and alert settings

The alert rule has three settings, set with `vine-risk update --alpha / --enter-ratio / --exit-ratio` or in `MonitorConfig`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `alpha` | 0.01 | an alert needs the permutation test's p-value to be at most this |
| `enter_ratio` | 2.0 | ... and the tau-matrix distance to be at least this multiple of the typical no-change distance |
| `exit_ratio` | 1.5 | the alert ends when the distance falls below this multiple |
| `scan_step` | 5 | run the test every this many days |
| `n_perm` | 499 | permutations per test |
| `block` | 10 | shuffle blocks of this many days (keeps volatility clustering) |

The two ratios are conventions, not estimated quantities (see [Live monitor](live-monitor.md)).
