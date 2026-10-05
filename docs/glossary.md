# Glossary

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
| Permutation test | Shuffle observations between two windows many times to see how big a difference looks by chance alone. |
| Multiple testing / FDR | Testing many dates yields some "significant" results by luck; Benjamini-Hochberg limits the expected share of false discoveries. |
| Parquet | A compact file format for tables; read with `pandas.read_parquet`. |
| Virtual environment | A private folder of installed packages for one project (`.venv`). |
| Editable install | `pip install -e`: makes the package importable while still using your live source files. |
