# Changelog

All notable changes to this project. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- GARCH-filtered marginals (`GarchMarginal`, config `rolling.marginal: garch_t | garch_empirical`): a GARCH(1,1) per asset fitted on a longer
  lookback than the copula window, with the copula fitted to the PIT of the standardised residuals; scenarios through it are
  one-day-ahead conditional forecasts. Run manifests include the marginal, and runs made before it existed remain compatible.
- `simulate_garch`: synthetic returns with GARCH volatility and copula-linked Student-t innovations (known truth).
- `vine-risk` command line (`run`, `update`, `schedule`, `replay`, `info`, `dashboard`) and `python -m vine_risk`.
- Daily update (`vine-risk update`): fetches new prices, checks them, processes each new day through the live monitor, extends the
  run folder incrementally; `--dry-run`; scheduler entries for cron, launchd and systemd via `vine-risk schedule`.
- Price sources: Yahoo Finance and local CSV files (`data.source`), a refreshable price cache, data-quality checks on new prices.
- Every pipeline step is incremental (new fits, scan dates and risk dates are appended); extending a run equals running it at once.
- Run manifests (`manifest.json`): versions, git commit, configuration and data fingerprints, steps; safe-resume checks that refuse
  mismatched fit settings or revised data; a lock against concurrent jobs; `vine-risk info`.
- `requirements-lock.txt` with the exact package versions; tests that keep it consistent with `pyproject.toml`.
- GitHub Actions: tests on Python 3.11-3.13, a job with the pinned lock, a package build/install check, a documentation build.
- Documentation site (MkDocs Material) with an API reference generated from the docstrings, and this changelog.

### Changed
- Python 3.11 or newer is required (the pinned numpy 2.5 and pandas 3 need it).
- The scripts in `scripts/` are thin wrappers around the shared pipeline functions.
- The README is short; details moved to the documentation.

### Fixed
- Test isolation: Streamlit's `AppTest` left the dashboard script as `__main__`, which broke later tests that start worker processes.
- Deprecated `license` table in `pyproject.toml`.

## [0.1.0] - 2026-10

The ten phases of the project plan.

### Added
- **Data and returns:** adjusted prices with a local cache, cleaning, log returns, missing-data handling.
- **Marginals:** empirical rank transform behind an interface for parametric marginals.
- **Vine copulas:** `VineCopula` (pyvinecopulib) with structured, serialisable results; rolling estimation with checkpoints, parallel fits and a stepwise API.
- **Dependence monitoring:** average and pairwise Kendall's tau, tail dependence implied by the fitted vine, structure-change metrics.
- **Change detection:** distance scores, rolling z-score and CUSUM, and a calibrated permutation test with effect sizes and false-discovery control.
- **Portfolio risk:** VaR and Expected Shortfall from copula simulation with benchmark models, rolling risk and a VaR backtest.
- **Validation study:** synthetic experiments for false alarms, power and delay; accuracy of tail and risk estimates.
- **Dashboard:** Streamlit app with nine interactive panels in light and dark themes.
- **Live monitor:** sequential monitoring with alert hysteresis; replay verified against the batch results.
- **Notebooks:** six executed tutorials.
