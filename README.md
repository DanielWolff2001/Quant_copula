# Dynamic Vine Copula Risk Monitoring

A quantitative risk framework for estimating and monitoring time-varying dependence
structures in financial markets using rolling vine copula models.

The project investigates how pairwise and higher-order dependence, particularly tail
dependence, evolves over time and whether statistically meaningful changes in the
dependence structure can be detected. The resulting dependence models are subsequently
used to investigate implications for portfolio tail risk.

This is a research project, not a trading strategy.

> **Status: work in progress.** Phases 1-5 are implemented (data, marginals, static
> vine, rolling estimation, dependence monitoring). Change detection, portfolio risk,
> the dashboard and the live simulation are still to come. See [Status](#status).

---

## 1. The idea in plain language

Stocks do not move independently. A **copula** is a way to describe *how* several
assets move together, separately from how each asset moves on its own. This matters
most in crashes, where assets often fall together far more often than ordinary
correlation suggests (this is called **tail dependence**).

With many assets, one big copula is hard to estimate, so we use a **vine copula**: it
builds the joint dependence out of many simple two-asset pieces ("pair-copulas")
arranged in a sequence of trees.

The project does this:

1. Take daily prices of a few liquid assets and turn them into returns.
2. Look at a **rolling window** (for example the last 250 trading days) and fit a vine
   copula to it.
3. Slide the window forward one day and fit again, over the whole history.
4. Track how the fitted dependence changes, decide which changes are real and which are
   just estimation noise, and see how portfolio risk (VaR, Expected Shortfall) reacts.

Central question: *how does the dependence structure of a multi-asset portfolio evolve
over time, and can meaningful changes in dependence and tail dependence be detected?*

---

## 2. Setup

You need **Python 3.10 or newer**. (Check with `python3 --version`; on macOS the
default `python3` can be older, so you may need to name a newer one, e.g. `python3.12`.)

```bash
# 1. Go to the project folder
cd Quant_copula

# 2. Create a virtual environment: a private folder (.venv) holding this project's
#    Python packages, so they don't clash with other projects on your computer.
python3.12 -m venv .venv

# 3. Activate it. Your terminal prompt should now show (.venv).
#    You must do this in every new terminal window.
source .venv/bin/activate

# 4. Install this project and its dependencies (explained in section 4)
pip install -e ".[dev]"

# 5. Check that everything works: runs the automated tests
pytest
```

Without activating, you can call the environment's Python directly, e.g.
`.venv/bin/python scripts/run_rolling.py` and `.venv/bin/pytest`.

### Running the rolling estimation

```bash
python scripts/run_rolling.py                       # settings from configs/default.yaml
python scripts/run_rolling.py --window 125          # a different window size
python scripts/run_rolling.py --last 350            # quick test on the last 350 days
```

The first run downloads prices from Yahoo Finance and caches them in `data/cache/`.
Fitting about 5,000 windows takes roughly an hour on 4 CPU cores. Progress is saved
as it goes, so if you stop the script and start it again it **resumes** where it left
off. Results end up in `data/results/w<window>/` as `.parquet` files (see
[Glossary](#6-glossary)).

After a run finishes, compute the monitoring metrics:

```bash
python scripts/compute_metrics.py                   # reads data/results/w250/checkpoint.jsonl
```

This writes `dependence_metrics.parquet` (one row per date: average absolute Kendall tau
`d_t`, Spearman/Pearson, model-implied lower/upper tail dependence, how much the vine
structure changed since the previous fit, AIC/BIC, ...) plus `pairwise_*.parquet`
tables with one column per asset pair.

**Tail dependence in this project.** `lower_tail_q` is the probability that asset *j*
is in its worst 5% of days given that asset *i* is, as implied by the fitted vine
(`q` = `tail_level` in the config). It is a finite-level version of the textbook tail
coefficient, so it is positive even for copulas with no asymptotic tail dependence; use
it to compare dates, not as an absolute truth. It is estimated by simulating from each
fitted vine with the same random numbers every time, so changes between dates come from
the model, not from simulation noise.

---

## 3. Project structure

If you are used to single scripts, the main difference here is that the reusable code
lives in a **package** (a folder of modules you `import`), and everything else
(scripts, tests, settings) sits around it and *uses* that package.

```
Quant_copula/
├── README.md              This file.
├── pyproject.toml         The project's "ID card": name, version, dependencies. (see below)
├── LICENSE                Terms under which others may use the code (MIT).
├── .gitignore             Files git should not track (the .venv, downloaded data, ...).
│
├── configs/
│   └── default.yaml       Settings: which tickers, window size, truncation level, ...
│
├── src/
│   └── vine_risk/         <-- THE PACKAGE: all the real code lives here.
│       ├── config.py          Reads default.yaml into typed Python objects.
│       ├── data.py            Downloads prices (with caching) and cleans them.
│       ├── returns.py         Prices -> log returns -> model-ready return matrix.
│       ├── marginals.py       Turns returns into uniform numbers in (0,1) via ranks.
│       ├── copula.py          VineCopula: fits a vine and summarises it.
│       ├── dependence.py      Pairwise tau/Spearman/Pearson and the monitoring metrics.
│       ├── rolling.py         RollingVineModel: the rolling-window machinery.
│       └── visualization.py   Plots (prices, uniformity check, vine trees).
│
├── scripts/
│   ├── run_rolling.py     Command-line entry point to run the rolling fit.
│   └── compute_metrics.py Turns a finished run into dependence_metrics.parquet.
│
├── tests/                 Automated checks of the maths and the code (run with pytest).
│
├── data/
│   ├── cache/             Downloaded prices (auto-created, not in git).
│   └── results/           Output of the rolling runs (auto-created, not in git).
│
├── reports/figures/       Saved figures and example outputs.
│
└── (planned)
    ├── notebooks/         Jupyter notebooks that demonstrate the package.
    └── dashboard/         Streamlit dashboard (planned).
```

Planned modules that do not exist yet: `change_detection.py` (structural change
scores), `portfolio.py` (VaR / Expected Shortfall) and `diagnostics.py`.

### Why is the code in `src/vine_risk/` and not next to the scripts?

Putting the code in a package means any script, test or notebook can simply write
`from vine_risk.rolling import RollingVineModel` without fiddling with file paths.
The extra `src/` level is a common convention that stops you from accidentally
importing the code from the wrong place.

### What is `pyproject.toml`?

A **TOML** file is just a plain-text settings file with simple `key = value` lines and
`[sections]`. `pyproject.toml` is the standard file where a Python project describes
itself. Ours says:

- the project's name (`vine-risk`) and version;
- which other packages it needs (`dependencies`: numpy, pandas, pyvinecopulib, ...);
- optional extras: `dev` (adds `pytest` for testing) and `dashboard` (adds Streamlit);
- where the code lives (`src/`), and where the tests are.

When you run `pip install -e ".[dev]"`, pip reads this file, installs all the listed
packages, and registers `vine_risk` so it can be imported from anywhere in this
environment.

- `.` means "the project in the current folder".
- `-e` ("editable") means pip links to your files instead of copying them, so edits
  you make to the code take effect immediately without reinstalling.
- `[dev]` selects the extra group of packages called `dev`.

### What is `configs/default.yaml`?

**YAML** is another plain-text settings format (indentation-based). We keep settings
here instead of hard-coding them in Python, so you can change the analysis without
touching code. For example:

```yaml
assets: [AAPL, MSFT, NVDA, JPM, XOM, JNJ, SPY, TLT]   # replace with any tickers
rolling:
  window: 250            # trading days per fit
  refit_frequency: 1     # 1 = refit every day, 5 = every 5th day
  truncation_level: 3    # only fit the first 3 trees of the vine (null = all)
  n_jobs: 4              # CPU cores to use
```

### What is `.gitignore`?

A list of files git should leave alone: the `.venv` folder (large and specific to your
machine), downloaded data and result files (they can be regenerated), and Python
temporary files such as `__pycache__`.

### What are `tests/`?

Small programs that check the code against cases where the right answer is known (for
example: log returns against hand-computed values, Kendall's tau against SciPy,
recovery of a known copula from simulated data). `pytest` finds and runs all of them.
If you change something and the tests still pass, you probably did not break anything.

---

## 4. How the pieces fit together

```
Yahoo Finance prices
        |   data.py         download + cache + clean
        v
  adjusted prices
        |   returns.py      log returns, handle missing data
        v
  return matrix R   (T days x d assets)
        |   marginals.py    rank / (T+1)   ->  numbers strictly between 0 and 1
        v
  uniform data U
        |   copula.py       fit a vine copula (pyvinecopulib)
        v
  VineFitResult  --  family, parameters, tail dependence, AIC/BIC, structure, ...
        |   rolling.py      repeat for every window, store one result per date
        v
  tables (.parquet): fits / pair_copulas / pairwise
```

Design choices worth knowing:

- **Log returns, not prices.** Copulas are fitted to returns, using adjusted prices so
  splits and dividends do not create fake jumps.
- **Ranks instead of a distribution model for each asset.** The first version uses
  the simple rank transform `u = rank(r) / (T + 1)`. `marginals.py` defines an
  interface (`Marginal`) so a Student-t or GARCH version can be dropped in later
  without touching the copula code.
- **A mature library for the vine itself.** We use
  [pyvinecopulib](https://vinecopulib.github.io/pyvinecopulib/) (a Python interface to
  a C++ library) instead of writing our own. Pair-copula families are chosen
  automatically using BIC (a score that penalises unnecessarily complex models).
- **No look-ahead.** The result stamped with date *t* only uses data up to and
  including *t*, and the rank transform is re-done inside each window. Tests check
  this.
- **Results are plain data, not model objects.** Each fit is summarised in a
  `VineFitResult` (a plain Python dataclass that converts to JSON), so results can be
  saved, reloaded and compared later.
- **Updating data is separate from refitting the model.** `RollingVineModel.push()`
  adds a new observation; `refit()` fits the model. This lets the same code handle
  daily refits, periodic refits, and later the "live" mode.

---

## 5. Using the package from Python

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

---

## 6. Glossary

| Term | Meaning |
|------|---------|
| Log return | `log(P_t / P_{t-1})`: the daily price change in a form that adds up over time. |
| Copula | A description of how variables depend on each other, independent of their individual distributions. |
| Vine copula | A joint copula assembled from many two-variable copulas arranged in trees. |
| Pair-copula | One two-variable building block of a vine, e.g. a "Clayton" or "Gaussian" copula. |
| Kendall's tau / Spearman | Rank-based measures of how strongly two assets move together (-1 to 1). |
| Tail dependence | Probability that two assets are extreme *together* (lower tail: crashes, upper tail: rallies). |
| AIC / BIC | Scores that balance how well a model fits against how complex it is; lower is better. |
| Truncation level | Fit only the first *k* trees of the vine; deeper trees are usually small and noisy. |
| VaR / ES | Value at Risk and Expected Shortfall: standard measures of portfolio tail loss. |
| Parquet | A compact file format for tables; read with `pandas.read_parquet`. |
| Virtual environment | A private folder of installed packages for one project (`.venv`). |
| Editable install | `pip install -e`: makes the package importable while still using your live source files. |

---

## 7. Status

| Phase | Content | State |
|------|---------|-------|
| 1 | Data download, cleaning, log returns | done |
| 2 | Marginals (rank transform) | done |
| 3 | Static vine copula, structured results, tree plot | done |
| 4 | Rolling estimation (checkpointed, parallel) | done |
| 5 | Dependence monitoring metrics (average tau, changes, tail dependence) | done |
| 6 | Structural change detection | next |
| 7 | Portfolio risk (VaR, ES) | planned |
| 8 | Synthetic validation | planned |
| 9 | Streamlit dashboard | planned |
| 10 | Live (replay) simulation | planned |

## 8. Limitations (to be extended)

- A rolling window describes *local* history; financial dependence is not stationary.
- Different vine structures can have nearly identical likelihoods, so structure changes
  can be noise.
- Looking at many pairs and many dates will produce some "unusual" changes by chance.
- In a vine, the tau and tail dependence of pair-copulas beyond the first tree are
  *conditional* on other assets; they are not the plain pairwise values.
- A detected statistical change is not automatically an economic regime change, and
  associations found here are not causal claims.
- Daily data only; no intraday dynamics.
