# Sector allocation research

## Question and frozen protocol

Can the release-lag-aware macro forecasts already used by IndustryScope rank
sector ETF returns one month ahead? This is one fixed research question, not a
search for whichever portfolio happens to work.

- At every month end, each sector model is refit on an expanding window of at
  least 120 months. An observation enters only after its estimated source
  release date.
- Forecasts from multiple configured macro series for one sector are averaged.
- At least six eligible sectors are required. The portfolio is equal-weight
  long the top forecast quartile and short the bottom forecast quartile.
- The primary result charges 10 basis points per dollar traded. Five and 25
  basis points are fixed sensitivity checks.
- Baselines are the equal-weight eligible sector universe and a market-neutral
  12-1 momentum rank using only returns known at the decision date.
- The report includes annualized return and volatility, Sharpe, maximum
  drawdown, beta to the equal-weight universe, cross-sectional Spearman rank IC,
  one-way turnover, and all three cost scenarios.
- A six-month circular block bootstrap gives the primary annualized-mean
  interval. HAC standard errors cover serial dependence. Holm correction covers
  six planned superiority tests: model above zero and model above momentum at
  each cost.

The immutable values live in the `Protocol` dataclass. The run below used
protocol digest `e00805b4be8d`. Changing the question changes that digest.

## Current result

Run on September 7, 2026 against the locally exported provider histories through
August 2026:

```text
Frozen sector-allocation research protocol
Protocol digest: e00805b4be8d
Evaluation: 2009-01 through 2026-07; 211 months; 13 sectors and 19 macro pairings appear in the forecast panel.
Primary portfolio: equal-weight top/bottom forecast quartiles, market neutral, net of 10 bps per dollar traded.
Baselines: equal-weight eligible sectors and market-neutral 12-1 momentum.
The configured ETF universe is current, not a survivorship-free historical security master.

portfolio        ann ret   ann vol   Sharpe    max DD   beta EW
model            -11.11%    16.34%    -0.63   -89.07%     -0.19
momentum          -5.01%    18.22%    -0.19   -70.55%      0.05
equal_weight      15.91%    15.32%     1.05   -21.16%      1.00

Cross-sectional rank IC: -0.083; HAC t=-3.57, two-sided p=0.000.
Primary annualized arithmetic mean, 95% circular block-bootstrap interval: [-16.77%, -4.15%].
Average monthly turnover: model 1.67x gross, momentum 1.06x gross.

Cost sensitivity and Holm correction over all six planned superiority tests:
  cost     against    HAC t    raw p    Holm p
    5bp        zero    -2.80    0.997     1.000
    5bp    momentum    -1.27    0.897     1.000
   10bp        zero    -3.11    0.999     1.000
   10bp    momentum    -1.34    0.909     1.000
   25bp        zero    -4.04    1.000     1.000
   25bp    momentum    -1.55    0.939     1.000

0 of 6 planned superiority tests survive Holm correction.
```

The macro rank failed. It was negatively associated with next-month sector
returns, lost money before and after costs, and trailed both baselines. This is
a useful falsification of the product's intuitive macro pairings, but it is not
investable alpha and is not presented as one.

## Controls

`tests/test_sector_allocation.py` provides four load-bearing controls:

1. A planted cross-sectional signal is recovered with positive rank IC and a
   Holm-adjusted result, so the evaluator can detect a signal that exists.
2. A fixed-seed no-signal panel does not become significant.
3. Rewriting every future predictor and return leaves earlier forecasts exactly
   unchanged.
4. A deliberately leaked oracle that ranks on next month's realized return
   produces perfect IC and an implausible jump in return, demonstrating that the
   negative control would expose this class of look-ahead error.

The cost test also verifies that basis points are applied to actual traded
notional, including the opening trade.

## Evidence boundaries

- The 13-sector panel is large enough for this bounded falsification, not for a
  general claim about asset pricing.
- The current configured ETF universe is used retrospectively. It is not a
  survivorship-free, point-in-time security master.
- Release lags are estimated from current provider metadata and applied
  conservatively. The export does not carry every historical release vintage,
  so revisions after first publication cannot be reconstructed.
- Results change as provider histories and the exported universe update. The
  date, protocol digest, and exact output above identify this run.
- The equal-weight baseline is market-exposed while the forecast and momentum
  portfolios are market-neutral. Beta is reported to make that distinction
  explicit; relative Sharpe is descriptive rather than a matched-risk test.
