# How it fits together

## Project structure

If you are used to single scripts, the main difference here is that the reusable code lives in a **package** (a folder of modules
you `import`), and everything else (commands, scripts, tests, settings) sits around it and *uses* that package.

```
Quant_copula/
├── README.md              Short introduction.
├── CHANGELOG.md           What changed, version by version.
├── pyproject.toml         The project's "ID card": name, version, dependencies, the vine-risk command.
├── requirements-lock.txt  Exact versions of all packages used for the published results.
├── mkdocs.yml, docs/      This documentation.
├── LICENSE, .gitignore
├── .github/workflows/     Automatic tests (ci.yml) and documentation build (docs.yml).
│
├── configs/default.yaml   Settings: tickers, data source, window, truncation, risk, ...
│
├── src/vine_risk/         <-- THE PACKAGE: all the real code lives here.
│   │  data and models
│   ├── config.py          Reads the YAML into typed Python objects.
│   ├── sources.py         Where prices come from: Yahoo Finance or your own CSV files.
│   ├── data.py            Price download (cached), cleaning, sanity checks on new prices.
│   ├── returns.py         Prices -> log returns -> model-ready return matrix.
│   ├── marginals.py       Turns returns into uniform numbers in (0,1) via ranks; MarginalSpec picks the model.
│   ├── garch.py           GARCH(1,1)-filtered marginals (volatility removed before the copula).
│   ├── copula.py          VineCopula: fits a vine and summarises it.
│   ├── dependence.py      Pairwise tau/Spearman/Pearson and the monitoring metrics.
│   ├── rolling.py         RollingVineModel: the rolling-window machinery, checkpoints.
│   │  analysis
│   ├── change_detection.py Structural change scores and the calibrated permutation test.
│   ├── portfolio.py       Loss, VaR, Expected Shortfall from simulated scenarios (rank or GARCH marginals).
│   ├── riskmodels.py      Standard models to compare against: historical, EWMA normal/t, filtered historical simulation.
│   ├── backtesting.py     Kupiec, Christoffersen, Acerbi-Szekely ES tests, FZ0 score, Diebold-Mariano.
│   ├── benchmark.py       Ten risk models, same days, same portfolios: forecasts and a backtest report.
│   ├── synthetic.py, validation.py   Simulated data with known truth, and the validation study.
│   │  running it
│   ├── runner.py          The pipeline steps (incremental), shared by the command line and scripts.
│   ├── update.py          The daily update.
│   ├── monitor.py         LiveMonitor: one observation at a time, with alerts.
│   ├── replay.py          Replay history through the monitor and verify it.
│   ├── manifest.py        What produced a run folder; safe-resume checks.
│   ├── locking.py         One job at a time per run folder.
│   ├── pipeline.py        Small helpers chaining the data steps (used by the notebooks).
│   ├── cli.py, __main__.py  The vine-risk command.
│   │  looking at it
│   ├── dashboard_data.py, dashboard_figures.py   Data access and charts for the dashboard.
│   └── visualization.py   Plots (prices, uniformity check, vine trees).
│
├── dashboard/app.py       The Streamlit dashboard (thin: widgets and layout only).
├── scripts/               Thin wrappers around the pipeline steps, the validation study, figures.
├── notebooks/             Six Jupyter notebooks that teach the package step by step.
├── tests/                 Automated checks of the maths and the code (run with pytest).
├── data/                  cache/ (downloaded prices) and results/ (run folders); not in git.
└── reports/               Saved figures and the validation study's result tables.
```

## The layers

The code is organised so that each layer only uses the ones above it:

1. **Models** (`marginals`, `copula`, `dependence`, `rolling`): pure computation on data frames. They know nothing about files or schedules.
2. **Analysis** (`change_detection`, `portfolio`): questions asked of the fitted models.
3. **Operations** (`runner`, `update`, `monitor`, `manifest`, `cli`): read and write run folders, resume safely, run daily.
4. **Presentation** (`dashboard_*`, `visualization`, notebooks): read finished run folders.

A **run folder** (for example `data/results/w250/`) is the interface between them: parquet tables, a checkpoint of fits, a
manifest. The dashboard, notebooks and replay only read it; the pipeline and the daily update extend it.

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
- optional extras: `dev` (adds `pytest` for testing), `dashboard` (adds Streamlit and Plotly) and `notebooks` (adds JupyterLab);
- where the code lives (`src/`), and where the tests are.

When you run `pip install -e ".[dev,dashboard,notebooks]"`, pip reads this file, installs all the listed
packages, and registers `vine_risk` so it can be imported from anywhere in this
environment.

- `.` means "the project in the current folder".
- `-e` ("editable") means pip links to your files instead of copying them, so edits
  you make to the code take effect immediately without reinstalling.
- `[dev,dashboard,notebooks]` selects the extra groups of packages with these names (you can list any subset).

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

## How the pieces fit together

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
