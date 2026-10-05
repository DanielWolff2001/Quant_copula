# Concepts

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

!!! tip "Words you may not know"
    The [glossary](glossary.md) explains terms such as Kendall's tau, tail dependence, BIC and VaR.
