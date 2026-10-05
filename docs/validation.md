# Validation study

Before trusting any result on real stocks, the detectors and the risk estimates were run on simulated data where the truth is known.

Before trusting any result on real stocks, the detectors were run on simulated data
where the truth is known (`src/vine_risk/validation.py`, `python scripts/run_validation.py`,
about 30 minutes; every number below is in `reports/validation/*.csv`). Four assets, 250-day
windows, 20 simulated histories per experiment, change at day 900 of 1,700. An alert is
`p <= 0.01`. *False alarm* is the share of dates without any change that raised an alert;
*power* is the share of dates raising an alert in the stretch where the two compared
windows straddle the change.

![Effect size around a known change](assets/detection.png)

| Experiment | False alarms | Power (tau matrix) | Power (best tail statistic) | Histories with a detection |
|------------|-------------:|-------------------:|----------------------------:|---------------------------:|
| A: constant dependence | 0-1% | n/a | n/a | n/a |
| A': constant dependence, clustered volatility | 1% | n/a | n/a | n/a |
| B: correlation jumps 0.3 → 0.7 | 1-2% | **94%** | 46% | 100% |
| C1: Gaussian → Student-t(3), same Kendall tau | 1% | 0% | 3% | 5% (tau) to 30% (tail) |
| C2: Gaussian → Student-t(1), same Kendall tau | 1% | 2% | 21% | 10% (tau) to 60% (tail) |

What this means:

- **The permutation test keeps its promise**: about 1% false alarms at the 1% level,
  also when volatility clusters.
- **Correlation changes are found reliably**, typically about 115 days (about half a
  window) after they happen.
- **Tail-only changes are mostly invisible** to every statistic here. Only a very
  strong tail change (C2, tail coefficient 0 to 0.5 at unchanged Kendall tau) is picked
  up, and then only in about half of the histories. The moderate one (C1) is not
  detectable with 250 daily observations: each window holds about 25 tail observations
  per asset, and the gap in tail dependence (0.33 vs 0.40 at the 10% level) is about the
  size of the sampling noise.
- **Distance scores from the rolling vine fits** (`structural_change_scores`) have no
  built-in null distribution, so their thresholds were calibrated by simulation (99th
  percentile of the no-change experiment). Calibrated on the harder volatility-clustering
  null, the Kendall-tau distance has 100% power in B (but 3% in C2), and the
  model-implied distance, which also uses the fitted tail dependence, has 74% in B and
  32% in C2 with about 0% false alarms. It is the more sensitive to tail changes, at
  the price of needing this calibration. Calibrated only on data *without* volatility
  clustering, the thresholds are too low and false alarms rise to 3-10% when volatility
  clusters.
- **Comparing consecutive days is useless**: the one-step distance (the literal "S_t"
  of the brief) has only 1-6% power in B because neighbouring windows share 249 of 250
  observations. Comparing with the window one year earlier is what works.
- **The z-score** of the disjoint-window distance (`z > 3`) is well calibrated (about
  1% false alarms) but has only 24% power in B: the baseline absorbs the shift. The
  **CUSUM** of the average dependence detects B (82% power) but keeps alarming after
  the change, because it never resets (31% "false alarms" are mostly that), and finds
  nothing in the tail experiments.

How well does the fitted vine recover the truth? (`reports/validation/estimation_accuracy.csv`,
40 histories per window length; true copula Student-t(3) with the same marginals)

| Window | Lower-tail coefficient (RMSE) | 99% ES: vine | 99% ES: Gaussian copula | 99% ES: historical |
|-------:|------------------------------:|-------------:|------------------------:|-------------------:|
| 125 | 25% | bias -6%, RMSE 13% | bias -12%, RMSE 15% | bias -14%, RMSE 18% |
| 250 | 14% | bias +3%, RMSE 11% | bias -7%, RMSE 10% | bias -2%, RMSE 12% |
| 500 | 7% | bias +2%, RMSE 7% | bias -8%, RMSE 9% | bias -1%, RMSE 9% |

A Gaussian copula underestimates the expected shortfall by 7-8% when the truth has tail
dependence, which is real but smaller than the estimation noise of a single 250-day
window (about 10%). With a Gaussian truth the vine costs a little accuracy
(ES RMSE 9.6% vs 7.6% at 250 days). Historical simulation from a short window
underestimates the 99% tail (bias -2% to -14% for windows up to 250 days).
