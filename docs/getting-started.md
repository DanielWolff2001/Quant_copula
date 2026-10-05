# Getting started

## What you need

* **Python 3.11 or newer** (check with `python3 --version`; on macOS the default `python3` can be older, so you may have to name a newer one, such as `python3.12`).
* An internet connection for the first price download (Yahoo Finance). Later runs use a local copy.
* About 4 CPU cores. The full 20-year run takes about an hour and a quarter on 4 cores; a first test takes a minute or two.

## Install

```bash
cd Quant_copula                      # the project folder

python3.12 -m venv .venv             # a private folder with this project's packages
source .venv/bin/activate            # do this in every new terminal; the prompt then shows (.venv)

pip install -e ".[dev,dashboard,notebooks,docs]"
pytest                               # optional: about two minutes, checks that everything works
vine-risk --version
```

The brackets pick optional groups of packages: `dev` (tests), `dashboard` (Streamlit), `notebooks` (JupyterLab), `docs` (this
documentation). Leave out the ones you do not need.

!!! note "Exact versions"
    `pip install -e ...` takes the newest compatible versions. To install exactly the versions the results in this repository were made
    with, use `pip install -r requirements-lock.txt && pip install -e . --no-deps` (see [Reproducibility](reproducibility.md)).

## A first run in two minutes

A short run on the last 700 days, fitting every 5th day, written to its own folder:

```bash
vine-risk run --last 700 --refit-frequency 5 --run-dir data/results/demo
vine-risk info data/results/demo         # what was done, with which versions and settings
vine-risk dashboard --run-dir data/results/demo
```

The first command downloads prices (cached in `data/cache/`), fits about 90 vines in parallel, computes the dependence metrics,
the change scores and the portfolio risk, and prints the time each step took. Press Ctrl+C at any time: progress is saved and the
same command continues where it stopped.

## The full run

```bash
vine-risk run                            # window 250, a fit every day, 2005 to today
```

This is the run behind the numbers in this documentation. Results go to `data/results/w250/`.

## Where to go next

* [Notebooks](notebooks.md): six step-by-step tutorials, the best way to learn what the pieces do.
* [Command line](cli.md): every command and option.
* [Daily updates](daily-update.md): keep a run up to date with new prices, on a schedule.
* [Dashboard](dashboard.md): explore a run interactively.
* [Concepts](concepts.md): the ideas in plain language.
