# Limitations

The project is a research tool, and its results come with caveats that matter for how they are read.

- A rolling window describes *local* history; financial dependence is not stationary.
- Different vine structures can have nearly identical likelihoods, so structure changes
  can be noise.
- Looking at many pairs and many dates will produce some "unusual" changes by chance.
- In a vine, the tau and tail dependence of pair-copulas beyond the first tree are
  *conditional* on other assets; they are not the plain pairwise values.
- A detected statistical change is not automatically an economic regime change, and
  associations found here are not causal claims.
- The permutation test treats blocks of 10 days as exchangeable; longer-lasting serial
  dependence still makes it somewhat too liberal, and it only says that two windows
  differ, not when inside them the change happened.
- Tail-dependence changes of realistic size are hard to detect with one or two years of
  daily data; a non-significant tail test is weak evidence of "no change".
- Moving-window comparisons at neighbouring dates are not independent tests, so the
  multiple-testing correction is approximate.
- The live monitor is a replay of history. A real feed would bring late, missing or revised
  prices, which the monitor rejects (it raises an error) rather than repairs.
- Daily data only; no intraday dynamics.
