"""Performance metrics: hand-computed values, edge cases and JSON-serialisable summaries."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from src import metrics as m

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2026-01-01", tz="UTC")
R4 = [0.01, -0.005, 0.02, 0.0]


def daily(values: list[float], start: pd.Timestamp = T0) -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def t_ns(ts: pd.Timestamp) -> int:
    return int(ts.value)


@dataclass(slots=True)
class Point:
    t_ns: int
    equity_liq: float
    equity_mid: float
    side: int


@dataclass(slots=True)
class Trade:
    pnl_net: float
    holding_s: float
    strategy: str


# --------------------------------------------------------------------------- Sharpe / Sortino

def test_sharpe_hand_value() -> None:
    # mean = 0.025/4 = 0.00625; deviations (.00375, -.01125, .01375, -.00625) square-sum to
    # 3.6875e-4, so sd(ddof=1) = sqrt(3.6875e-4/3) = 0.0110868 and SR = 0.00625/0.0110868*sqrt(365).
    sd = math.sqrt(3.6875e-4 / 3.0)
    assert sd == pytest.approx(0.011086779, abs=1e-9)
    expected = 0.00625 / sd * math.sqrt(365.0)
    sr = m.sharpe_ratio(R4, rf_annual=0.0, periods_per_year=365.0, min_periods=0)
    assert sr == pytest.approx(expected, rel=1e-12)
    assert sr == pytest.approx(10.770133, abs=1e-6)


def test_sharpe_matches_pandas_with_rf() -> None:
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.001, 0.01, 200))
    rf = (1.042) ** (1 / 365) - 1
    ex = r - rf
    expected = ex.mean() / ex.std(ddof=1) * math.sqrt(365)
    assert m.sharpe_ratio(r, rf_annual=0.042) == pytest.approx(expected, rel=1e-12)
    # A positive rf lowers the Sharpe of the same returns.
    assert m.sharpe_ratio(r, rf_annual=0.042) < m.sharpe_ratio(r, rf_annual=0.0)


def test_per_period_rf_compounds_back_to_annual() -> None:
    rf_d = m.per_period_rf(0.042, 365.0)
    assert rf_d == pytest.approx(1.042 ** (1 / 365) - 1, rel=1e-12)
    assert (1 + rf_d) ** 365 == pytest.approx(1.042, rel=1e-12)
    assert m.per_period_rf(0.042, 8760.0) == pytest.approx(1.042 ** (1 / 8760) - 1, rel=1e-12)
    assert m.per_period_rf(0.0, 365.0) == 0.0
    assert m.periods_per_year("1D") == 365.0
    assert m.periods_per_year("1h") == 8760.0
    assert m.periods_per_year("D") == 365.0


def test_sharpe_degenerate_inputs_are_nan() -> None:
    assert math.isnan(m.sharpe_ratio([0.001] * 50))                     # zero variance
    assert math.isnan(m.sharpe_ratio([0.001] * 50, rf_annual=0.0))
    assert math.isnan(m.sharpe_ratio([0.01], min_periods=0))             # n < 2
    assert math.isnan(m.sharpe_ratio([], min_periods=0))
    assert math.isnan(m.sortino_ratio([0.01], min_periods=0))
    # NaNs are dropped, not propagated.
    with_nan = [0.01, np.nan, -0.005, 0.02, 0.0]
    assert m.sharpe_ratio(with_nan, rf_annual=0.0, min_periods=0) == pytest.approx(10.770133, abs=1e-6)


def test_min_periods_behaviour() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, 30)
    assert math.isnan(m.sharpe_ratio(r[:29]))                            # default min_periods=30
    assert math.isfinite(m.sharpe_ratio(r[:29], min_periods=0))
    assert math.isfinite(m.sharpe_ratio(r))
    assert math.isnan(m.sortino_ratio(r[:29]))
    assert math.isfinite(m.sortino_ratio(r[:29], min_periods=0))


def test_sortino_hand_value() -> None:
    # Downside deviation over ALL four periods: sqrt(0.005^2 / 4) = 0.0025.
    s = m.sortino_ratio(R4, rf_annual=0.0, periods_per_year=365.0, min_periods=0)
    assert s == pytest.approx(0.00625 / 0.0025 * math.sqrt(365.0), rel=1e-12)
    assert s == pytest.approx(47.762433, abs=1e-6)
    assert m.sortino_ratio([0.01, 0.02, 0.0], rf_annual=0.0, min_periods=0) == math.inf
    assert math.isnan(m.sortino_ratio([0.0, 0.0, 0.0], rf_annual=0.0, min_periods=0))


def test_lo_correction_hand_formula() -> None:
    rng = np.random.default_rng(3)
    x = np.empty(400)
    x[0] = 0.0
    eps = rng.normal(0.0005, 0.01, 400)
    for i in range(1, 400):                                              # AR(1), phi = 0.5
        x[i] = 0.5 * x[i - 1] + eps[i]
    xc = x - x.mean()
    rho1 = float(np.sum(xc[:-1] * xc[1:]) / np.sum(xc * xc))
    sr_period = x.mean() / x.std(ddof=1)
    q = 365.0
    expected = sr_period * q / math.sqrt(q + 2 * (q - 1) * rho1)
    got = m.sharpe_ratio(x, rf_annual=0.0, periods_per_year=q, lo_correction=True, lo_max_lag=1)
    assert got == pytest.approx(expected, rel=1e-12)
    # Positive autocorrelation: the naive sqrt(P) scaling overstates the Sharpe.
    naive = m.sharpe_ratio(x, rf_annual=0.0, periods_per_year=q)
    assert 0 < m.sharpe_ratio(x, rf_annual=0.0, periods_per_year=q, lo_correction=True) < naive


def test_sharpe_se_analytic() -> None:
    assert m.sharpe_se(0.0, 100, 0.0, 3.0) == pytest.approx(0.1)
    assert m.sharpe_se(0.5, 100, 0.0, 3.0) == pytest.approx(math.sqrt(1.125 / 100))
    # Negative skew and fat tails widen the standard error.
    assert m.sharpe_se(0.5, 100, -1.0, 6.0) > m.sharpe_se(0.5, 100, 0.0, 3.0)
    assert math.isnan(m.sharpe_se(0.5, 0, 0.0, 3.0))


# --------------------------------------------------------------------------- drawdown

def test_drawdown_worked_example() -> None:
    eq = daily([100, 110, 99, 105, 112, 108])
    dd = m.drawdown_series(eq)
    np.testing.assert_allclose(dd.to_numpy(), [0, 0, -0.1, 105 / 110 - 1, 0, 108 / 112 - 1], atol=1e-15)
    assert dd.iloc[-1] == pytest.approx(-0.0357142857, abs=1e-10)

    st = m.drawdown_stats(eq)
    assert st.max_dd == pytest.approx(-0.10, abs=1e-12)
    assert st.peak_t == T0 + 1 * DAY
    assert st.trough_t == T0 + 2 * DAY
    assert st.recovery_t == T0 + 4 * DAY
    assert st.duration == 3 * DAY
    assert st.peak_to_trough == 1 * DAY
    assert st.censored is False
    # Episodes: day1->day4 (3 d, recovered) and day4->end (1 d, still open -> censored).
    assert st.longest_duration == 3 * DAY
    assert st.longest_censored is False
    tail = m.drawdown_stats(eq.iloc[4:])
    assert tail.max_dd == pytest.approx(108 / 112 - 1)
    assert tail.censored is True and tail.recovery_t is None and tail.duration == 1 * DAY


def test_drawdown_unrecovered_max_episode_is_censored() -> None:
    st = m.drawdown_stats(daily([100, 120, 90, 95]))
    assert st.max_dd == pytest.approx(-0.25)
    assert st.peak_t == T0 + DAY and st.trough_t == T0 + 2 * DAY
    assert st.recovery_t is None
    assert st.censored is True
    assert st.duration == 2 * DAY                   # T_end - t_peak, a lower bound
    assert st.longest_duration == 2 * DAY and st.longest_censored is True


def test_drawdown_longest_differs_from_deepest() -> None:
    # Long shallow episode (day0 -> day4) then a short deep one (day4 -> day6).
    st = m.drawdown_stats(daily([100, 99, 98, 99, 100.5, 80, 101]))
    assert st.max_dd == pytest.approx(80 / 100.5 - 1)
    assert st.peak_t == T0 + 4 * DAY and st.recovery_t == T0 + 6 * DAY
    assert st.duration == 2 * DAY
    assert st.longest_duration == 4 * DAY and st.longest_censored is False
    # Deepest episode recovered, but a longer later one is still open.
    st2 = m.drawdown_stats(daily([100, 50, 101, 100, 100, 100, 100]))
    assert st2.max_dd == pytest.approx(-0.5) and st2.censored is False and st2.duration == 2 * DAY
    assert st2.longest_duration == 4 * DAY and st2.longest_censored is True


def test_drawdown_edge_cases() -> None:
    st = m.drawdown_stats(daily([100, 101, 101, 102]))
    assert st.max_dd == 0.0 and st.peak_t is None and st.duration == pd.Timedelta(0)
    # A flat top: the peak is the LAST mark at the high-water mark.
    assert m.drawdown_stats(daily([100, 110, 110, 99, 111])).peak_t == T0 + 2 * DAY
    with pytest.raises(ValueError):
        m.drawdown_series(daily([100, 0, 50]))
    assert m.max_drawdown_from_returns([0.1, -0.1, -0.1, 0.5]) == pytest.approx(0.891 / 1.1 - 1)
    assert m.max_drawdown_from_returns([-0.2]) == pytest.approx(-0.2)   # a first loss counts


def test_cagr_and_calmar() -> None:
    eq = pd.Series([100.0, 90.0, 121.0], index=[T0, T0 + 100 * DAY, T0 + 730 * DAY])
    g = m.cagr(eq)
    assert g.value == pytest.approx(0.10, rel=1e-12)
    assert g.span_days == 730 and g.short_sample is False
    c = m.calmar(eq)
    assert c.value == pytest.approx(0.10 / 0.10, rel=1e-12)
    assert c.short_sample is False
    short = m.cagr(daily([100, 101]))
    assert short.short_sample is True and short.value == pytest.approx(1.01 ** 365 - 1, rel=1e-9)
    assert math.isnan(m.calmar(daily([100, 101, 102])).value)          # no drawdown
    assert math.isnan(m.cagr(daily([100])).value)


# --------------------------------------------------------------------------- inputs / resampling

def test_equity_from_points_dicts_objects_and_series() -> None:
    pts = [
        {"t_ns": t_ns(T0 + 2 * DAY), "equity_liq": 102.0},
        {"t_ns": t_ns(T0), "equity_liq": 100.0},
        {"t_ns": t_ns(T0 + DAY), "equity_liq": 101.0},
        {"t_ns": t_ns(T0 + DAY), "equity_liq": 101.5},                   # re-mark after a fill wins
    ]
    s = m.equity_from_points(pts)
    assert str(s.index.tz) == "UTC"
    assert list(s.index) == [T0, T0 + DAY, T0 + 2 * DAY]
    assert s.tolist() == [100.0, 101.5, 102.0]

    objs = [Point(t_ns(T0), 100.0, 100.5, 0), Point(t_ns(T0 + DAY), 99.0, 100.2, 1)]
    assert m.equity_from_points(objs).tolist() == [100.0, 99.0]
    assert m.equity_from_points(objs, which="equity_mid").tolist() == [100.5, 100.2]

    naive = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    assert m.equity_from_points(naive).index[0] == T0
    ny = pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp("2025-12-31 19:00", tz="America/New_York")]))
    assert m.equity_from_points(ny).index[0] == T0
    ints = pd.Series([1.0, 2.0], index=[t_ns(T0), t_ns(T0 + DAY)])
    assert list(m.equity_from_points(ints).index) == [T0, T0 + DAY]
    frame = pd.DataFrame(pts)
    assert m.equity_from_points(frame).tolist() == [100.0, 101.5, 102.0]


def test_period_returns_locf_irregular_timestamps() -> None:
    h = pd.Timedelta(hours=1)
    eq = pd.Series(
        [100.0, 105.0, 110.0, 99.0],
        index=[T0 + 10 * h, T0 + 20 * h, T0 + 2 * DAY + 3 * h, T0 + 2 * DAY + 5 * h],
    )
    r = m.period_returns(eq, "1D")
    # Bins (t-1D, t] labelled t: day1 -> 105 (last mark of day 0), day2 -> empty -> 105 carried
    # forward (zero return, never interpolated), day3 -> 99 (last mark of day 2).
    assert list(r.index) == [T0 + 2 * DAY, T0 + 3 * DAY]
    np.testing.assert_allclose(r.to_numpy(), [0.0, 99 / 105 - 1], rtol=1e-15)
    # A mark exactly on a bin edge belongs to the bin that ends there (closed='right').
    # (and creates no empty bin after it: pandas 3's resampler would add a zero return there).
    eq2 = pd.concat([eq, pd.Series([120.0], index=[T0 + 3 * DAY])])
    r2 = m.period_returns(eq2, "1D")
    assert list(r2.index) == [T0 + 2 * DAY, T0 + 3 * DAY]
    assert r2.iloc[-1] == pytest.approx(120 / 105 - 1)
    # Hourly grid: 10:00 -> 13:30 gives bins 10, 11, 12, 13, 14 and four returns.
    hourly = pd.Series([1.0, 1.1, 1.21], index=[T0 + 10 * h, T0 + 11.5 * h, T0 + 13.5 * h])
    np.testing.assert_allclose(m.period_returns(hourly, "1h").to_numpy(), [0.0, 0.1, 0.0, 0.1], rtol=1e-12)
    assert m.period_returns(eq.iloc[:1], "1D").empty
    assert m.period_returns(pd.Series(dtype=float), "1D").empty


# --------------------------------------------------------------------------- trades

def test_wilson_interval() -> None:
    lo, hi = m.wilson_interval(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    assert m.wilson_interval(0, 10)[0] == 0.0
    assert m.wilson_interval(10, 10)[1] == 1.0
    assert all(math.isnan(v) for v in m.wilson_interval(0, 0))
    with pytest.raises(ValueError):
        m.wilson_interval(3, 2)


def test_profit_factor() -> None:
    assert m.profit_factor([10, -5, 20, -5, 0]) == pytest.approx(3.0)
    assert m.profit_factor([1.0, 2.0]) == math.inf
    assert m.profit_factor([-1.0, -2.0]) == 0.0
    assert math.isnan(m.profit_factor([]))
    assert math.isnan(m.profit_factor([0.0, 0.0]))


def test_trade_stats_hand_values() -> None:
    trades = pd.DataFrame({
        "pnl_net": [10.0, -5.0, 20.0, -5.0, 0.0],
        "holding_s": [60.0, 120.0, 180.0, 240.0, 600.0],
        "entry_cost": [100.0, 100.0, 200.0, 50.0, 100.0],
    })
    st = m.trade_stats(trades)
    assert st["n_trades"] == 5 and st["n_wins"] == 2 and st["n_losses"] == 2
    assert st["hit_rate"] == pytest.approx(0.4)
    assert (st["hit_rate_ci_lo"], st["hit_rate_ci_hi"]) == pytest.approx(m.wilson_interval(2, 5))
    assert st["avg_win"] == pytest.approx(15.0) and st["avg_loss"] == pytest.approx(-5.0)
    assert st["profit_factor"] == pytest.approx(3.0)
    assert st["total_pnl"] == pytest.approx(20.0) and st["mean_pnl"] == pytest.approx(4.0)
    assert st["holding_mean_s"] == pytest.approx(240.0) and st["holding_median_s"] == pytest.approx(180.0)
    assert st["mean_return_on_cost"] == pytest.approx((0.1 - 0.05 + 0.1 - 0.1 + 0.0) / 5)
    assert st["anecdotal"] is True
    assert "by_entry" not in st


def test_trade_stats_outside_band_split_and_groups() -> None:
    trades = pd.DataFrame({
        "pnl_net": [3.0, -1.0, -2.0, 4.0] * 10,
        "holding_s": [10.0] * 40,
        "outside_band_at_entry": [True, False, False, True] * 10,
        "exit_reason": ["revert", "stop", "timeout", "revert"] * 10,
        "strategy": ["z_nogate", "z_nogate", "z_edge", "arb"] * 10,
    })
    st = m.trade_stats(trades)
    assert st["anecdotal"] is False
    conv, out = st["by_entry"]["convergence"], st["by_entry"]["outside_band"]
    assert conv["n_trades"] == 20 and conv["total_pnl"] == pytest.approx(-30.0) and conv["hit_rate"] == 0.0
    assert out["n_trades"] == 20 and out["total_pnl"] == pytest.approx(70.0) and out["hit_rate"] == 1.0
    assert st["exit_reasons"] == {"revert": 20, "stop": 10, "timeout": 10}
    assert st["by_strategy"]["arb"]["total_pnl"] == pytest.approx(40.0)
    assert st["by_strategy"]["z_nogate"]["n_trades"] == 20
    # Missing flags count as "entered inside the band".
    obj = pd.DataFrame({"pnl_net": [1.0, 2.0], "holding_s": [1.0, 1.0],
                        "outside_band_at_entry": pd.Series([True, None], dtype=object)})
    assert m.trade_stats(obj)["by_entry"]["convergence"]["n_trades"] == 1


def test_trade_stats_records_and_empty() -> None:
    st = m.trade_stats([Trade(1.0, 30.0, "a"), Trade(-2.0, 90.0, "b")])
    assert st["n_trades"] == 2 and st["holding_mean_s"] == pytest.approx(60.0)
    assert set(st["by_strategy"]) == {"a", "b"}
    empty = m.trade_stats(pd.DataFrame(columns=["pnl_net", "holding_s"]))
    assert empty["n_trades"] == 0 and empty["anecdotal"] is True
    assert math.isnan(empty["hit_rate"]) and math.isnan(empty["profit_factor"])


def test_exposure_time_weighted() -> None:
    h = pd.Timedelta(hours=1)
    side = pd.Series([1, 0, 0, 1, 0], index=[T0, T0 + h, T0 + 2 * h, T0 + 3 * h, T0 + 4 * h])
    assert m.exposure(side) == pytest.approx(0.5)                  # 1 h + 1 h long out of 4 h
    pts = [Point(t_ns(T0), 100, 100, -1), Point(t_ns(T0 + 3 * h), 100, 100, 0),
           Point(t_ns(T0 + 4 * h), 100, 100, 0)]
    assert m.exposure(pts) == pytest.approx(0.75)                  # short counts as exposed
    assert math.isnan(m.exposure(side.iloc[:1]))


def test_turnover() -> None:
    eq = pd.Series([1000.0, 1000.0], index=[T0, T0 + 73 * DAY])    # 0.2 years
    trades = pd.DataFrame({"pnl_net": [10.0, 10.0], "holding_s": [1.0, 1.0],
                           "entry_cost": [100.0, 100.0], "exit_proceeds": [110.0, 110.0]})
    assert m.turnover(trades, eq) == pytest.approx(420.0 / 1000.0 / 0.2)
    assert m.turnover(trades.drop(columns="exit_proceeds"), eq) == pytest.approx(200.0 / 1000.0 / 0.2)
    with pytest.raises(ValueError):
        m.turnover(trades.drop(columns="entry_cost"), eq)


# --------------------------------------------------------------------------- bootstrap

def test_stationary_bootstrap_reproducible() -> None:
    r = np.random.default_rng(11).normal(0.001, 0.01, 250)
    a = m.stationary_bootstrap(r, np.mean, n_boot=500, mean_block=5, seed=42)
    b = m.stationary_bootstrap(r, np.mean, n_boot=500, mean_block=5, seed=42)
    c = m.stationary_bootstrap(r, np.mean, n_boot=500, mean_block=5, seed=43)
    assert a == b
    assert a != c
    assert a[0] < r.mean() < a[1]
    with pytest.raises(ValueError):
        m.stationary_bootstrap(r, np.mean, mean_block=0.5)
    assert all(math.isnan(v) for v in m.stationary_bootstrap([0.1], np.mean, mean_block=2))


def test_stationary_bootstrap_block_limits() -> None:
    r = np.random.default_rng(5).normal(0.0, 0.01, 400)
    # An effectively infinite block makes every resample a circular rotation -> same mean.
    lo, hi = m.stationary_bootstrap(r, np.mean, n_boot=200, mean_block=1e15, seed=0)
    assert lo == pytest.approx(r.mean(), abs=1e-15) and hi == pytest.approx(r.mean(), abs=1e-15)
    # mean_block = 1 is the IID bootstrap: width ~ 2 * 1.96 * sd / sqrt(n).
    lo, hi = m.stationary_bootstrap(r, np.mean, n_boot=2000, mean_block=1.0, seed=0)
    assert hi - lo == pytest.approx(2 * 1.96 * r.std() / math.sqrt(r.size), rel=0.1)


def test_stationary_indices_geometric_blocks() -> None:
    n, p = 500, 0.2
    idx = m._stationary_indices(np.random.default_rng(0), n, 400, p)
    assert idx.shape == (400, n) and idx.min() >= 0 and idx.max() < n
    continues = idx[:, 1:] == (idx[:, :-1] + 1) % n
    # P(no continuation) = p * (1 - 1/n): a fresh start can land on the next index by chance.
    assert 1 - continues.mean() == pytest.approx(p * (1 - 1 / n), rel=0.03)


# --------------------------------------------------------------------------- summarize

def _random_walk_points(span: pd.Timedelta, step: pd.Timedelta, seed: int) -> list[Point]:
    rng = np.random.default_rng(seed)
    times = pd.date_range(T0, T0 + span, freq=step)
    eq = 10_000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, len(times))))
    side = (np.arange(len(times)) // 5) % 2
    return [Point(t_ns(t), float(e), float(e) * 1.001, int(s)) for t, e, s in zip(times, eq, side)]


def test_summarize_switches_to_hourly_for_short_spans() -> None:
    pts = _random_walk_points(pd.Timedelta(days=10), pd.Timedelta(minutes=30), seed=1)
    trades = pd.DataFrame({"pnl_net": [5.0, -2.0, 3.0], "holding_s": [3600.0] * 3,
                           "entry_cost": [500.0] * 3, "exit_proceeds": [505.0, 498.0, 503.0]})
    out = m.summarize(pts, trades, n_boot=300)
    json.dumps(out, allow_nan=False)
    assert out["sharpe_freq"] == "1h" and out["periods_per_year"] == 8760.0
    assert out["n_periods"] == 240 and out["too_short"] is False
    assert any("hourly" in w for w in out["warnings"])
    assert any("anecdotal" in w for w in out["warnings"])
    assert out["trades"]["anecdotal"] is True and out["trades"]["n_trades"] == 3
    assert out["short_sample"] is True
    eq = m.equity_from_points(pts)
    rets = m.period_returns(eq, "1h")
    assert out["sharpe"] == pytest.approx(m.sharpe_ratio(rets, periods_per_year=8760.0), rel=1e-12)
    assert out["sharpe_rf0"] == pytest.approx(m.sharpe_ratio(rets, rf_annual=0.0, periods_per_year=8760.0))
    assert out["sharpe_ci95_lo"] < out["sharpe"] < out["sharpe_ci95_hi"]
    assert out["max_dd_ci95_lo"] <= out["max_dd_ci95_hi"] <= 0.0
    assert out["sharpe_display"] == f"{out['sharpe']:.2f}"
    assert out["exposure"] == pytest.approx(m.exposure(pts))
    assert out["turnover"] == pytest.approx(m.turnover(trades, eq))
    assert out["total_return"] == pytest.approx(eq.iloc[-1] / eq.iloc[0] - 1)
    assert out["max_dd"] == pytest.approx(m.drawdown_stats(eq).max_dd)
    assert out["start"] == T0.isoformat()


def test_summarize_daily_for_long_spans_and_series_input() -> None:
    pts = _random_walk_points(pd.Timedelta(days=60), pd.Timedelta(hours=6), seed=2)
    out = m.summarize(pts, n_boot=200)
    json.dumps(out, allow_nan=False)
    assert out["sharpe_freq"] == "1D" and out["periods_per_year"] == 365.0
    assert out["n_periods"] == 60 and out["too_short"] is False
    assert out["trades"] is None and out["turnover"] is None
    assert not any("hourly" in w for w in out["warnings"])
    # A plain Series gives the same headline numbers but no exposure (no side information).
    series_out = m.summarize(m.equity_from_points(pts), n_boot=200)
    assert series_out["sharpe"] == pytest.approx(out["sharpe"], rel=1e-12)
    assert series_out["exposure"] is None and out["exposure"] is not None
    # An explicit frequency overrides the automatic choice.
    assert m.summarize(pts, freq="1h", n_boot=50)["periods_per_year"] == 8760.0


def test_summarize_too_short_and_degenerate_values_are_json_safe() -> None:
    eq = pd.Series([100.0, 101.0, 100.5, 102.0], index=[T0 + i * pd.Timedelta(hours=3) for i in range(4)])
    trades = pd.DataFrame({"pnl_net": [1.0, 1.0], "holding_s": [60.0, 60.0]})   # no losses -> PF inf
    out = m.summarize(eq, trades)
    text = json.dumps(out, allow_nan=False)
    assert out["too_short"] is True and out["n_periods"] == 9
    assert out["sharpe"] is None and out["sharpe_rf0"] is None and out["sortino"] is None
    assert out["sharpe_display"] == "n/a (too short)"
    assert out["sharpe_ci95_lo"] is None and out["sharpe_se"] is None
    assert out["trades"]["profit_factor"] is None and out["trades"]["n_losses"] == 0
    assert out["max_dd_censored"] is False and out["max_dd_duration_s"] == pytest.approx(6 * 3600.0)
    assert out["max_dd_recovery"] == (T0 + pd.Timedelta(hours=9)).isoformat()
    assert any("too short" in w for w in out["warnings"])
    assert "NaN" not in text and "Infinity" not in text
    with pytest.raises(ValueError):
        m.summarize(pd.Series(dtype=float))
