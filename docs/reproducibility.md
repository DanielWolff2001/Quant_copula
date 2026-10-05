# Reproducibility

Three layers make a result traceable and repeatable.

## 1. Pinned versions: `requirements-lock.txt`

`requirements-lock.txt` lists the exact version of every package in the environment that produced the results here.

```bash
pip install -r requirements-lock.txt && pip install -e . --no-deps      # exactly those versions
```

The normal install takes the newest compatible versions instead, which is what you want for new work. The lock file is tested in a
fresh environment (the full test suite passes with it), and a test checks that it covers everything `pyproject.toml` asks for.
To update it deliberately: upgrade, re-run the tests, then `pip freeze --exclude-editable > requirements-lock.txt` and restore the header.

## 2. A manifest in every run folder: `manifest.json`

Every run folder records what produced it. `vine-risk info <run folder>` prints it.

| Recorded | Why |
|----------|-----|
| package version, git commit, and whether there were uncommitted changes | which code |
| Python, operating system, versions of the main libraries, a hash of the lock file | which environment |
| the full configuration and its hash | which settings |
| the **fit settings** (window, refit frequency, truncation, selection criterion, tail simulation, seed, assets, cleaning rules) and their hash | which settings change the fits |
| the data: assets, number of observations, date range, and a hash of the return matrix | which data |
| each step with its time and counts | what was done |

### What the manifest protects against

* **Mixing incompatible fits.** Extending a run made with a 250-day window using a 125-day window would silently produce a
  meaningless series. A run is only extended when its fit settings match; otherwise the command stops and lists the differences.
  Settings that do not change fits (the VaR level, the number of CPU cores) may change freely.
* **Stale fits after a data revision.** Before extending a run, the empirical Kendall's tau of the latest stored window is recomputed
  from the current prices and compared with the stored value. If they differ, the vendor changed history.
* **Two jobs at once.** A lock file in the run folder.

## 3. Determinism

* All random numbers are seeded (`risk.seed`): the simulation inside each fit and the VaR/ES scenarios use fixed quasi-random
  sequences, and the permutation tests use fixed seeds, so rerunning gives the same numbers.
* Results do not depend on how the work is split: fits are identical with 1 or 4 threads per fit, serial or on several processes,
  all at once or one day at a time (all tested; the live replay equals the stored batch results to rounding error).
* Tests run on every push (see below), on several Python versions.

## Continuous integration

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs the test suite on Python 3.11, 3.12 and 3.13 with freshly resolved
packages, once with the pinned lock file, and builds and installs the package in an empty environment. The documentation is built
with warnings treated as errors (`docs.yml`).
