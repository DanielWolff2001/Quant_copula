# Risk model comparison

Does the vine copula forecast portfolio risk better than standard models? `vine-risk benchmark` answers this by letting ten models
forecast the next day's **Value at Risk (VaR)** and **Expected Shortfall (ES)** on the same days, for the same portfolios, using only
the data available on each day, and then checking the forecasts against what actually happened.

!!! note "Status"
    The method, the code and the tests are complete. The results on the 20-year real-data run are being produced and will be added to
    this page. The known-truth experiment (below) already shows what each model can and cannot do.

## The ten models

| Model | What it is | Reacts to today's volatility? |
|-------|------------|-------------------------------|
| `hist` | historical simulation: empirical quantiles of the last 250 portfolio returns | slowly |
| `ewma_n` | RiskMetrics: exponentially weighted covariance (lambda 0.94), normal returns | yes, but thin tails |
| `ewma_t` | the same covariance with Student-t returns, tail thickness estimated | yes |
| `fhs` | filtered historical simulation: GARCH(1,1) volatility per asset, residual rows used as joint scenarios | yes |
| `vine_emp`, `gauss_emp`, `indep_emp` | vine, Gaussian and independence copulas with rank marginals (the window's own distribution) | no (unconditional) |
| `vine_garch`, `gauss_garch`, `indep_garch` | the same copulas on GARCH-filtered marginals; scenarios are forecasts for tomorrow given today's volatility | yes |

The three copula variants of each family share the same random numbers and the same marginals, so any difference between `vine`,
`gauss` and `indep` comes from dependence alone: independence switches it off, the Gaussian copula keeps correlation but has no tail
dependence, the vine adds tail dependence where the data call for it.

## How the forecasts are made (no look-ahead)

For every forecast date, each model sees only the returns up to and including that date. The vine models use the vine fitted on the
window ending that day (from the run folders), and the GARCH models fit on the preceding 500 days. The realised loss is the portfolio's
loss on the *next* day, computed from the raw returns. Models are compared only on dates where all of them have a forecast.

Portfolios are named in the configuration (`risk.portfolios`; see [Configuration](configuration.md)): by default all assets equally,
the six stocks equally, and 60 % SPY / 40 % TLT. Levels are 97.5 % (the Basel standard for ES) and 99 %.

## The backtests

| Test | Question | How to read it |
|------|----------|----------------|
| Kupiec | is the share of days with a loss above the VaR close to the expected 2.5 % / 1 %? | small p-value: wrong number of exceedances |
| Christoffersen | do exceedances cluster in time, and is coverage right? | small p-value: the model reacts too slowly (clusters) |
| Acerbi-Székely Z1, Z2 | when VaR is exceeded, is the loss as large as the ES forecast says? | positive Z and small p-value: risk underestimated |
| FZ0 score | one number that judges VaR and ES forecasts together (lower is better) | rank models by it |
| Diebold-Mariano | is one model's FZ0 score significantly lower than the reference's? | `dm_stat` < 0 and small `dm_p`: better than the reference |

The ES test p-values are simulated under the hypothesis that each day's forecast distribution is right, using a 20-point summary of
each day's tail; their size and power were checked by simulation (they reject a correct forecast about 5 % of the time at the 5 % level
and a forecast that is 35 % too narrow more than 90 % of the time).

## Running it

```bash
vine-risk run                                                                          # rank-marginal fits (once)
vine-risk run --marginal garch_t --marginal-lookback 500 --run-dir data/results/w250_garch --steps rolling,metrics,risk
vine-risk benchmark --garch-run-dir data/results/w250_garch                            # forecasts + report
```

The GARCH run takes about as long as the first (roughly 70 to 90 minutes); the benchmark itself about 25 minutes. Both are incremental
and can be interrupted. Output in `data/results/benchmark/`: `forecasts.parquet` (one row per date, model, portfolio and level),
`realized.parquet`, and `report.csv` with every test statistic and p-value.

## What the known-truth experiment shows

Real data cannot say which model is *right*. In the [validation study](validation.md) the truth is a GARCH process with a Student-t
copula, so the exact one-day-ahead VaR and ES are known. Over 80 forecast dates (8 independent series), at the 99 % level:

| Model | VaR error (RMSE) | ES error (RMSE) | ES bias |
|-------|-----------------:|----------------:|--------:|
| vine on GARCH marginals | 7.9 % | 8.7 % | +2.1 % |
| Gaussian copula on GARCH marginals | 7.5 % | 10.4 % | -6.6 % |
| EWMA, Student-t | 12 % | 13 % | 0 % |
| filtered historical simulation | 12 % | 20 % | +4.5 % |
| EWMA, normal | 17 % | 25 % | -23 % |
| historical simulation | 41 % | 48 % | +11 % |
| vine / Gaussian copula, rank marginals | 51 % / 43 % | 52 % / 40 % | +18 % / +5 % |

(Preliminary figures from a development run; the final table will come from `scripts/run_validation.py`.) The reading: models that
ignore today's volatility cannot follow changing risk; among those that do, the Gaussian copula underestimates the tail (it has no
tail dependence) and the vine does not.

## Limitations

* A comparison on one data set and one period says nothing about other markets; robustness to the window, the assets and the model
  settings has not been studied.
* Rank marginals cap simulated losses at the window's worst day, which matters beyond about the 99.5 % level.
* A GARCH fit fails (no convergence, or near-integrated volatility) for about 5-7 % of asset-windows; those assets fall back to an EWMA
  volatility.
* The 20-point tail summary behind the ES test p-values is an approximation.
* DCC-GARCH and extreme-value models are not included.
