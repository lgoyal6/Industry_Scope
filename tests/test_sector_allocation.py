from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analysis.macro_forecast import Pairing
from analysis.sector_allocation import (
    Protocol,
    evaluate_protocol,
    forecast_panel,
    portfolio_months,
)


def synthetic_pairings(*, slope: float, seed: int, sectors: int = 8, months: int = 220):
    generator = np.random.default_rng(seed)
    index = pd.period_range("2000-01", periods=months, freq="M")
    pairings = []
    for number in range(sectors):
        growth = pd.Series(generator.normal(0, 0.02, months), index=index)
        noise = generator.normal(0, 0.01, months)
        returns = pd.Series(slope * growth.shift(1).fillna(0).to_numpy() + noise, index=index)
        pairing = Pairing(
            sector=f"sector-{number}",
            etf=f"ETF{number}",
            series_id=f"SERIES{number}",
            label=f"Series {number}",
            lag_days=45,
        )
        pairings.append((pairing, returns, growth))
    return pairings


def fast_protocol(**changes) -> Protocol:
    base = Protocol(min_train_months=60, min_assets=6, bootstrap_samples=100)
    return replace(base, **changes)


def test_planted_cross_sectional_signal_is_recovered():
    result = evaluate_protocol(synthetic_pairings(slope=2.5, seed=31), fast_protocol())
    assert result.mean_ic > 0.65
    assert result.metrics["model"].sharpe > 2.0
    primary = result.hypothesis_tests[
        (result.hypothesis_tests["cost_bps"] == 10)
        & (result.hypothesis_tests["comparison"] == "zero")
    ].iloc[0]
    assert primary.holm_p < 0.05


def test_no_signal_does_not_become_a_positive_result():
    result = evaluate_protocol(synthetic_pairings(slope=0.0, seed=31), fast_protocol())
    assert abs(result.mean_ic) < 0.10
    assert (result.hypothesis_tests["holm_p"] >= 0.05).all()


def test_rewriting_the_future_cannot_move_an_earlier_forecast():
    protocol = fast_protocol()
    original = synthetic_pairings(slope=2.5, seed=7)
    honest = forecast_panel(original, protocol)
    cutoff = pd.Period("2014-01", freq="M")

    corrupted = []
    for pairing, returns, growth in original:
        altered_returns = returns.copy()
        altered_growth = growth.copy()
        altered_returns.loc[altered_returns.index >= cutoff] = 5.0
        altered_growth.loc[altered_growth.index >= cutoff] = -7.0
        corrupted.append((pairing, altered_returns, altered_growth))
    tampered = forecast_panel(corrupted, protocol)

    keys = ["origin", "sector"]
    before = honest[honest["origin"] < cutoff][keys + ["prediction"]].set_index(keys)
    after = tampered[tampered["origin"] < cutoff][keys + ["prediction"]].set_index(keys)
    pd.testing.assert_frame_equal(before, after)


def test_deliberate_future_return_ranking_is_caught_by_the_control():
    protocol = fast_protocol()
    panel = forecast_panel(synthetic_pairings(slope=0.0, seed=31), protocol)
    honest = portfolio_months(panel, protocol)
    leaked = panel.copy()
    leaked["prediction"] = leaked["actual"]
    oracle = portfolio_months(leaked, protocol)

    # The oracle creates an implausibly perfect ranking. This is a negative
    # control, not a candidate strategy, and the large separation proves the
    # control would expose a pipeline that ranked on future outcomes.
    assert oracle["ic"].min() == pytest.approx(1.0)
    assert oracle["model_gross"].mean() > honest["model_gross"].mean() + 0.02


def test_costs_are_charged_on_traded_notional():
    panel = pd.DataFrame([
        {"origin": pd.Period("2020-01", freq="M"), "sector": f"s{i}",
         "prediction": float(i), "actual": i / 100, "momentum": float(i), "etf": f"E{i}",
         "n_models": 1}
        for i in range(8)
    ])
    monthly = portfolio_months(panel, fast_protocol())
    row = monthly.iloc[0]
    assert row["model_turnover"] == pytest.approx(2.0)
    assert row["model_net_10"] == pytest.approx(row["model_gross"] - 0.002)
