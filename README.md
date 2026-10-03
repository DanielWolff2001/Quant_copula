# Dynamic Vine Copula Risk Monitoring

A quantitative risk framework for estimating and monitoring time-varying dependence
structures in financial markets using rolling vine copula models.

The project investigates how pairwise and higher-order dependence, particularly tail
dependence, evolves over time and whether statistically meaningful changes in the
dependence structure can be detected. The resulting dependence models are subsequently
used to investigate implications for portfolio tail risk.

> Work in progress. Phase 1 (data and returns) is implemented.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
