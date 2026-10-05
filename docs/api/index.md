# API reference

The reference is generated from the docstrings of the package `vine_risk` (in `src/vine_risk/`). Modules are grouped by what
they do; the order follows the pipeline.

| Page | Modules |
|------|---------|
| [Data and marginals](data.md) | `config`, `sources`, `data`, `returns`, `marginals`, `pipeline` |
| [Models and rolling fits](models.md) | `copula`, `dependence`, `rolling` |
| [Change detection, risk, validation](analysis.md) | `change_detection`, `portfolio`, `synthetic`, `validation` |
| [Operations](operations.md) | `runner`, `update`, `monitor`, `replay`, `manifest`, `locking`, `cli` |
| [Dashboard and plots](dashboard.md) | `dashboard_data`, `dashboard_figures`, `visualization` |

Most users only need the command line (see [Command line](../cli.md)) and a few classes: `VineCopula`, `RollingVineModel`,
`LiveMonitor`. The [notebooks](../notebooks.md) show how they are used.
