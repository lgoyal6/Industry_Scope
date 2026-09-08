"""Frozen, point-in-time sector allocation research protocol.

This turns the release-lag-aware forecasts in :mod:`analysis.macro_forecast`
into one deliberately narrow portfolio test.  It is research evidence, not an
investment strategy: the ETF universe is today's configured universe, the
sample is small, and the exported histories are not a survivorship-free point-
in-time security master.

Protocol
--------
At each month end, each eligible sector model is refit on an expanding window.
When a sector has several configured macro series, their one-month forecasts
are averaged.  The portfolio is equal-weight long the top quartile and short
the bottom quartile, with at least six sectors required.  The primary result is
net of 10 basis points per dollar traded.  Five and 25 basis points are fixed
sensitivity checks.  The two baselines are the contemporaneous equal-weight
sector universe and a market-neutral 12-1 momentum ranking.

The constants below are intentionally code, rather than command-line knobs.
Changing the research question requires a reviewed code change and produces a
new protocol digest in the report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import floor
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from analysis.macro_forecast import Pairing, build_panel, load_pairings, rolling_origin_forecasts


@dataclass(frozen=True)
class Protocol:
    min_train_months: int = 120
    min_assets: int = 6
    tail_fraction: float = 0.25
    momentum_lookback_months: int = 12
    momentum_skip_months: int = 1
    primary_cost_bps: int = 10
    cost_scenarios_bps: tuple[int, ...] = (5, 10, 25)
    hac_lags: int = 3
    bootstrap_block_months: int = 6
    bootstrap_samples: int = 2_000
    bootstrap_seed: int = 20260907

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Metrics:
    months: int
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    beta_to_equal_weight: float


@dataclass(frozen=True)
class ResearchResult:
    protocol: Protocol
    forecast_panel: pd.DataFrame
    monthly: pd.DataFrame
    metrics: dict[str, Metrics]
    mean_ic: float
    ic_hac_t: float
    ic_hac_p: float
    primary_annual_mean_ci: tuple[float, float]
    hypothesis_tests: pd.DataFrame


def _momentum_at(returns: pd.Series, origin: pd.Period, lookback: int, skip: int) -> float:
    history = returns.loc[returns.index <= origin]
    end = len(history) - skip
    start = end - lookback
    if start < 0 or end <= start:
        return float("nan")
    return float(history.iloc[start:end].sum())


def forecast_panel(
    pairings: Sequence[tuple[Pairing, pd.Series, pd.Series]],
    protocol: Protocol = Protocol(),
) -> pd.DataFrame:
    """Create one point-in-time forecast per sector and origin.

    Multiple macro series for one sector are averaged only after each model has
    made its own expanding-window forecast.  The next-month realized return is
    converted from log return to simple return before portfolio aggregation.
    """
    rows: list[dict[str, object]] = []
    for pairing, returns, growth in pairings:
        panel = build_panel(returns, growth)
        if len(panel) <= protocol.min_train_months:
            continue
        forecasts = rolling_origin_forecasts(panel, min_train=protocol.min_train_months)
        for origin, row in forecasts.iterrows():
            rows.append({
                "origin": origin,
                "sector": pairing.sector,
                "etf": pairing.etf,
                "series_id": pairing.series_id,
                "prediction": float(row["model"]),
                "actual": float(np.expm1(row["actual"])),
                "momentum": _momentum_at(
                    returns,
                    origin,
                    protocol.momentum_lookback_months,
                    protocol.momentum_skip_months,
                ),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "origin", "sector", "etf", "prediction", "actual", "momentum", "n_models"
        ])
    raw = pd.DataFrame(rows)
    # A sector's realized return and momentum are identical across its macro
    # models. Validate that before taking the first value.
    grouped = raw.groupby(["origin", "sector"], sort=True)
    if (grouped["actual"].nunique() > 1).any() or (grouped["momentum"].nunique() > 1).any():
        raise ValueError("models for one sector disagree on the return history")
    return grouped.agg(
        etf=("etf", "first"),
        prediction=("prediction", "mean"),
        actual=("actual", "first"),
        momentum=("momentum", "first"),
        n_models=("series_id", "nunique"),
    ).reset_index()


def _tail_weights(frame: pd.DataFrame, score: str, fraction: float) -> dict[str, float]:
    ordered = frame.sort_values([score, "sector"], kind="mergesort")
    count = max(1, floor(len(ordered) * fraction))
    short = ordered.head(count)["sector"].tolist()
    long = ordered.tail(count)["sector"].tolist()
    weights = {sector: -1.0 / count for sector in short}
    weights.update({sector: 1.0 / count for sector in long})
    return weights


def _turnover(current: dict[str, float], previous: dict[str, float]) -> float:
    names = set(current) | set(previous)
    return float(sum(abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names))


def portfolio_months(panel: pd.DataFrame, protocol: Protocol = Protocol()) -> pd.DataFrame:
    """Apply the frozen ranks and charge costs on one-way traded notional."""
    records: list[dict[str, object]] = []
    previous_model: dict[str, float] = {}
    previous_momentum: dict[str, float] = {}
    previous_equal: dict[str, float] = {}
    for origin, frame in panel.groupby("origin", sort=True):
        frame = frame.dropna(subset=["prediction", "actual", "momentum"])
        if len(frame) < protocol.min_assets:
            continue
        model_weights = _tail_weights(frame, "prediction", protocol.tail_fraction)
        momentum_weights = _tail_weights(frame, "momentum", protocol.tail_fraction)
        equal_weights = {sector: 1.0 / len(frame) for sector in frame["sector"]}
        realized = dict(zip(frame["sector"], frame["actual"]))
        model_gross = sum(weight * realized[name] for name, weight in model_weights.items())
        momentum_gross = sum(weight * realized[name] for name, weight in momentum_weights.items())
        equal_gross = sum(weight * realized[name] for name, weight in equal_weights.items())
        model_turnover = _turnover(model_weights, previous_model)
        momentum_turnover = _turnover(momentum_weights, previous_momentum)
        equal_turnover = _turnover(equal_weights, previous_equal)
        record: dict[str, object] = {
            "origin": origin,
            "assets": len(frame),
            "model_gross": model_gross,
            "momentum_gross": momentum_gross,
            "equal_weight_gross": equal_gross,
            "model_turnover": model_turnover,
            "momentum_turnover": momentum_turnover,
            "equal_weight_turnover": equal_turnover,
            "ic": frame["prediction"].corr(frame["actual"], method="spearman"),
        }
        for cost in protocol.cost_scenarios_bps:
            rate = cost / 10_000
            record[f"model_net_{cost}"] = model_gross - rate * model_turnover
            record[f"momentum_net_{cost}"] = momentum_gross - rate * momentum_turnover
            record[f"equal_weight_net_{cost}"] = equal_gross - rate * equal_turnover
        records.append(record)
        previous_model = model_weights
        previous_momentum = momentum_weights
        previous_equal = equal_weights
    return pd.DataFrame(records).set_index("origin") if records else pd.DataFrame()


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def summarize(returns: pd.Series, equal_weight: pd.Series) -> Metrics:
    n = len(returns)
    wealth = float((1.0 + returns).prod())
    annual_return = wealth ** (12.0 / n) - 1.0 if n and wealth > 0 else float("nan")
    volatility = float(returns.std(ddof=1) * np.sqrt(12))
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(12)) if volatility else float("nan")
    market_variance = float(equal_weight.var(ddof=1))
    beta = float(returns.cov(equal_weight) / market_variance) if market_variance else float("nan")
    return Metrics(n, annual_return, volatility, sharpe, _max_drawdown(returns), beta)


def _hac_mean_test(values: pd.Series, lags: int, *, one_sided: bool = False) -> tuple[float, float]:
    clean = values.dropna().to_numpy()
    fitted = sm.OLS(clean, np.ones(len(clean))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    statistic = float(fitted.tvalues[0])
    p_value = float(fitted.pvalues[0])
    if one_sided:
        p_value = p_value / 2 if statistic > 0 else 1 - p_value / 2
    return statistic, p_value


def _block_bootstrap_annual_mean(
    values: pd.Series, protocol: Protocol
) -> tuple[float, float]:
    clean = values.dropna().to_numpy()
    generator = np.random.default_rng(protocol.bootstrap_seed)
    starts = np.arange(len(clean))
    estimates = []
    blocks = int(np.ceil(len(clean) / protocol.bootstrap_block_months))
    for _ in range(protocol.bootstrap_samples):
        sample: list[float] = []
        for start in generator.choice(starts, size=blocks, replace=True):
            sample.extend(clean[(start + np.arange(protocol.bootstrap_block_months)) % len(clean)])
        estimates.append(float(np.mean(sample[:len(clean)]) * 12))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate_protocol(
    pairings: Sequence[tuple[Pairing, pd.Series, pd.Series]],
    protocol: Protocol = Protocol(),
) -> ResearchResult:
    panel = forecast_panel(pairings, protocol)
    monthly = portfolio_months(panel, protocol)
    if monthly.empty:
        raise ValueError("the exported histories do not produce six eligible sectors in any month")
    primary = protocol.primary_cost_bps
    equal = monthly[f"equal_weight_net_{primary}"]
    metrics: dict[str, Metrics] = {}
    for name in ("model", "momentum", "equal_weight"):
        metrics[name] = summarize(monthly[f"{name}_net_{primary}"], equal)

    ic_t, ic_p = _hac_mean_test(monthly["ic"], protocol.hac_lags)
    hypotheses = []
    raw_p = []
    for cost in protocol.cost_scenarios_bps:
        model = monthly[f"model_net_{cost}"]
        for comparison, difference in (
            ("zero", model),
            ("momentum", model - monthly[f"momentum_net_{cost}"]),
        ):
            statistic, p_value = _hac_mean_test(difference, protocol.hac_lags, one_sided=True)
            hypotheses.append({"cost_bps": cost, "comparison": comparison, "hac_t": statistic,
                               "raw_p": p_value})
            raw_p.append(p_value)
    adjusted = multipletests(raw_p, method="holm")[1]
    tests = pd.DataFrame(hypotheses)
    tests["holm_p"] = adjusted
    primary_returns = monthly[f"model_net_{primary}"]
    return ResearchResult(
        protocol=protocol,
        forecast_panel=panel,
        monthly=monthly,
        metrics=metrics,
        mean_ic=float(monthly["ic"].mean()),
        ic_hac_t=ic_t,
        ic_hac_p=ic_p,
        primary_annual_mean_ci=_block_bootstrap_annual_mean(primary_returns, protocol),
        hypothesis_tests=tests,
    )


def report(result: ResearchResult) -> str:
    first = str(result.monthly.index.min())
    last = str(result.monthly.index.max())
    sectors = result.forecast_panel["sector"].nunique()
    pairings = int(result.forecast_panel["n_models"].groupby(result.forecast_panel["sector"]).max().sum())
    lines = [
        "Frozen sector-allocation research protocol",
        f"Protocol digest: {result.protocol.digest}",
        f"Evaluation: {first} through {last}; {len(result.monthly)} months; "
        f"{sectors} sectors and {pairings} macro pairings appear in the forecast panel.",
        "Primary portfolio: equal-weight top/bottom forecast quartiles, market neutral, "
        f"net of {result.protocol.primary_cost_bps} bps per dollar traded.",
        "Baselines: equal-weight eligible sectors and market-neutral 12-1 momentum.",
        "The configured ETF universe is current, not a survivorship-free historical security master.",
        "",
        f"{'portfolio':<14}{'ann ret':>10}{'ann vol':>10}{'Sharpe':>9}{'max DD':>10}{'beta EW':>10}",
    ]
    for name in ("model", "momentum", "equal_weight"):
        metric = result.metrics[name]
        lines.append(
            f"{name:<14}{metric.annual_return:>10.2%}{metric.annual_volatility:>10.2%}"
            f"{metric.sharpe:>9.2f}{metric.max_drawdown:>10.2%}{metric.beta_to_equal_weight:>10.2f}"
        )
    low, high = result.primary_annual_mean_ci
    lines += [
        "",
        f"Cross-sectional rank IC: {result.mean_ic:+.3f}; HAC t={result.ic_hac_t:+.2f}, "
        f"two-sided p={result.ic_hac_p:.3f}.",
        f"Primary annualized arithmetic mean, 95% circular block-bootstrap interval: "
        f"[{low:+.2%}, {high:+.2%}].",
        f"Average monthly turnover: model {result.monthly['model_turnover'].mean():.2f}x gross, "
        f"momentum {result.monthly['momentum_turnover'].mean():.2f}x gross.",
        "",
        "Cost sensitivity and Holm correction over all six planned superiority tests:",
        f"{'cost':>6}{'against':>12}{'HAC t':>9}{'raw p':>9}{'Holm p':>10}",
    ]
    for row in result.hypothesis_tests.itertuples(index=False):
        lines.append(
            f"{row.cost_bps:>5}bp{row.comparison:>12}{row.hac_t:>9.2f}"
            f"{row.raw_p:>9.3f}{row.holm_p:>10.3f}"
        )
    survives = result.hypothesis_tests[result.hypothesis_tests["holm_p"] < 0.05]
    lines += [
        "",
        f"{len(survives)} of {len(result.hypothesis_tests)} planned superiority tests survive Holm correction.",
        "A non-significant or negative result is retained as the result. This report does not claim investable alpha.",
    ]
    return "\n".join(lines)


def write_report(path: Path, result: ResearchResult) -> None:
    path.write_text(report(result) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    print(report(evaluate_protocol(load_pairings())))
