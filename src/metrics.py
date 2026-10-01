"""Performance statistics for backtests and paper trading (offline only; pandas allowed).

Every function is pure. The input is the simulator's equity curve (``EquityPoint``s with
``t_ns`` and liquidation-marked ``equity_liq``) and its trade table (``TradeRecord`` rows).

Definitions
-----------
* **Period returns.** Equity is sampled on a regular grid by last-observation-carried-
  forward (LOCF): ``r_k = E_k / E_{k-1} - 1``. Marks are never interpolated, because an
  interpolated mark is a price nobody could have traded at.
* **Sharpe.** ``mean(r - rf) / sd(r - rf, ddof=1) * sqrt(P)`` with the per-period risk-free
  rate ``rf = (1 + R_f)^(1/P) - 1``. Prediction markets trade 24/7, so P = 365 for daily
  returns and P = 8760 for hourly ones (not 252). Idle collateral on Polymarket is assumed
  to earn nothing, so flat periods carry a negative excess return - that is the honest
  comparison against T-bills.
* **Sortino.** The downside deviation ``sqrt(mean(min(r - rf, 0)^2))`` is averaged over *all*
  periods, not only the losing ones (averaging over losers only overstates the risk of a
  strategy that rarely loses and understates it for one that often does).
* **Drawdown.** ``DD_t = E_t / max_{s<=t} E_s - 1``. An episode runs from the last point at
  the high-water mark (the peak) to the first point back at or above it (the recovery).
  An episode still open at the end of the data has duration ``T_end - t_peak``; it is
  flagged ``censored`` because that duration is only a lower bound.

Pitfalls this module is designed around
---------------------------------------
* **Per-trade Sharpe inflation.** ``mean(pnl)/sd(pnl) * sqrt(trades per year)`` ignores
  idle time and overlapping trades and overstates the Sharpe; we only compute Sharpe from
  calendar-time returns of the whole account.
* **Mid-marking smooths equity.** Mid marks hide the spread you would pay to get out and
  produce a smoother curve with a higher Sharpe and a shallower drawdown. Headline numbers
  use liquidation marks (``equity_liq``); mid marks are a diagnostic only.
* **Annualising short samples.** A Sharpe from a few periods, or a CAGR from a few weeks,
  is noise raised to a large power. Sharpe needs at least 30 periods (otherwise it is
  reported as "n/a (too short)"), samples under 30 days switch to hourly returns, CAGR and
  Calmar under 90 days carry ``short_sample``, and fewer than 30 trades is "anecdotal".
* **Autocorrelated returns.** Mark-to-market P&L of a mean-reverting position is serially
  correlated, so ``sqrt(P)`` scaling is biased; Lo's (2002) correction is available and a
  stationary block bootstrap (Politis-Romano 1994) gives the confidence interval.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_RF = 0.042          # annual risk-free rate (3-month T-bill, Sep 2026; UNVERIFIED)
YEAR = pd.Timedelta(days=365)
MIN_PERIODS = 30            # fewer periods -> Sharpe/Sortino reported as "n/a (too short)"
MIN_TRADES = 30             # fewer trades -> "anecdotal"
DAILY_MIN_SPAN = pd.Timedelta(days=30)   # shorter samples use hourly returns
SHORT_SAMPLE = pd.Timedelta(days=90)     # shorter samples flag CAGR/Calmar as short_sample
_BOOT_CELLS = 2_000_000     # cap on resample-index matrix size per bootstrap batch

_TRADE_FIELDS = ("pnl_net", "holding_s", "entry_cost", "exit_proceeds",
                 "outside_band_at_entry", "exit_reason", "strategy")


# --------------------------------------------------------------------------- inputs

def _field(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def _has_field(obj: Any, name: str) -> bool:
    return name in obj if isinstance(obj, Mapping) else hasattr(obj, name)


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if pd.api.types.is_numeric_dtype(index):
        return pd.DatetimeIndex(pd.to_datetime(np.asarray(index, dtype=np.int64), unit="ns", utc=True))
    raise TypeError(f"expected a DatetimeIndex or integer epoch-ns index, got {type(index).__name__}")


def _series_from_points(points: Any, field: str) -> pd.Series:
    """Time series of ``field`` from points, a DataFrame with ``t_ns`` or a ready Series.

    The result has a sorted UTC ``DatetimeIndex``; of several marks with the same
    timestamp the last one (in input order) wins, matching "mark again after a fill".
    """
    if isinstance(points, pd.Series):
        s = pd.Series(points.to_numpy(dtype=float), index=_utc_index(points.index), name=points.name or field)
    else:
        if isinstance(points, pd.DataFrame):
            t = points["t_ns"].to_numpy(dtype=np.int64)
            v = points[field].to_numpy(dtype=float)
        else:
            rows = list(points)
            t = np.fromiter((_field(p, "t_ns") for p in rows), dtype=np.int64, count=len(rows))
            v = np.fromiter((_field(p, field) for p in rows), dtype=float, count=len(rows))
        s = pd.Series(v, index=pd.DatetimeIndex(pd.to_datetime(t, unit="ns", utc=True)), name=field)
    s = s.sort_index(kind="stable")
    return s[~s.index.duplicated(keep="last")].dropna()


def equity_from_points(points: Sequence[Any] | pd.DataFrame | pd.Series,
                       which: str = "equity_liq") -> pd.Series:
    """Equity curve as a float Series on a sorted UTC ``DatetimeIndex``.

    ``points`` may be ``EquityPoint`` objects or dicts with ``t_ns`` and ``which``
    (``"equity_liq"`` for headline numbers, ``"equity_mid"`` for the smoothing diagnostic),
    a DataFrame with those columns, or a ready Series (naive indexes are taken as UTC,
    integer indexes as epoch nanoseconds). NaN marks are dropped.
    """
    return _series_from_points(points, which)


def _trades_frame(trades: pd.DataFrame | Iterable[Any]) -> pd.DataFrame:
    if isinstance(trades, pd.DataFrame):
        return trades
    rows = [{f: _field(t, f) for f in _TRADE_FIELDS if _has_field(t, f)} for t in trades]
    return pd.DataFrame(rows)


def _clean(returns: Any) -> np.ndarray:
    x = np.asarray(returns, dtype=float).ravel()
    return x[~np.isnan(x)]


# --------------------------------------------------------------------------- returns

def _freq_delta(freq: str) -> pd.Timedelta:
    return pd.to_timedelta(freq if freq[:1].isdigit() else f"1{freq}")


def periods_per_year(freq: str) -> float:
    """Number of ``freq`` periods in a 365-day year (``"1D"`` -> 365, ``"1h"`` -> 8760)."""
    return float(YEAR / _freq_delta(freq))


def period_returns(equity: pd.Series, freq: str = "1D") -> pd.Series:
    """Simple returns of LOCF-resampled equity.

    Bins are ``(t - freq, t]`` labelled at their right edge, so the value at ``t`` is the
    last mark at or before ``t``; empty bins carry the previous mark forward (a flat
    period is a zero return, not a gap). Bins before the first mark are dropped, and so
    is the first return, which has no previous value. A partial first period is thereby
    discarded rather than compared with a full one.

    Equivalent to ``resample(freq, label="right", closed="right").last().ffill()``, but
    built from ``ceil`` + ``reindex`` because pandas 3's calendar-day resampler adds empty
    bins around marks that sit exactly on a bin edge, which would invent zero returns
    outside the sample.
    """
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float, name="returns")
    last = eq.groupby(eq.index.ceil(freq)).last()
    grid = last.reindex(pd.date_range(last.index[0], last.index[-1], freq=_freq_delta(freq))).ffill()
    return grid.pct_change().iloc[1:].rename("returns")


def per_period_rf(rf_annual: float, periods_per_year: float) -> float:
    """Per-period risk-free rate ``(1 + R)^(1/P) - 1`` (geometric, so P periods compound to R)."""
    return math.expm1(math.log1p(rf_annual) / periods_per_year)


def _excess(returns: Any, rf_annual: float, periods_per_year: float) -> np.ndarray:
    return _clean(returns) - per_period_rf(rf_annual, periods_per_year)


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelations at lags 1..max_lag (biased estimator, as in statsmodels' acf)."""
    xc = x - x.mean()
    c0 = float(xc @ xc)
    return np.array([float(xc[:-k] @ xc[k:]) / c0 for k in range(1, max_lag + 1)])


def _lo_scale(x: np.ndarray, q: float, max_lag: int | None) -> float:
    """Lo (2002) annualisation factor ``q / sqrt(q + 2 sum_{k=1}^{m} (q - k) rho_k)``.

    It replaces ``sqrt(q)``, which is only right for serially uncorrelated returns.
    Autocorrelations beyond lag ``m`` are taken as zero: ``m`` defaults to the
    Newey-West rule ``floor(4 (n/100)^(2/9))`` because estimating hundreds of lags from a
    short sample would make the factor pure noise. Returns NaN if the implied variance
    of q-period returns is not positive.
    """
    n = x.size
    m = math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)) if max_lag is None else max_lag
    m = int(min(m, n - 1, math.ceil(q) - 1))
    if m < 1:
        return math.sqrt(q)
    k = np.arange(1, m + 1, dtype=float)
    denom = q + 2.0 * float(np.sum((q - k) * _acf(x, m)))
    return q / math.sqrt(denom) if denom > 0 else math.nan


def sharpe_ratio(returns: pd.Series | np.ndarray | Sequence[float], *, rf_annual: float = DEFAULT_RF,
                 periods_per_year: float = 365.0, lo_correction: bool = False,
                 lo_max_lag: int | None = None, min_periods: int = MIN_PERIODS) -> float:
    """Annualised Sharpe ratio ``mean(r - rf) / sd(r - rf, ddof=1) * sqrt(P)``.

    NaN when there are fewer than ``max(2, min_periods)`` returns (callers show
    "n/a (too short)"; pass ``min_periods=0`` to compute it anyway) or when the excess
    returns have zero variance. ``lo_correction`` replaces ``sqrt(P)`` with Lo's (2002)
    autocorrelation-adjusted factor (see ``_lo_scale``).
    """
    ex = _excess(returns, rf_annual, periods_per_year)
    if ex.size < max(2, min_periods) or np.ptp(ex) == 0.0:
        return math.nan
    sd = float(ex.std(ddof=1))
    if not sd > 0.0:
        return math.nan
    scale = _lo_scale(ex, periods_per_year, lo_max_lag) if lo_correction else math.sqrt(periods_per_year)
    return float(ex.mean()) / sd * scale


def sortino_ratio(returns: pd.Series | np.ndarray | Sequence[float], *, rf_annual: float = DEFAULT_RF,
                  periods_per_year: float = 365.0, min_periods: int = MIN_PERIODS) -> float:
    """Annualised Sortino ratio ``mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(P)``.

    The downside deviation averages over all periods. With no downside it is ``inf`` for
    a positive mean and NaN otherwise; too-short samples are NaN as for ``sharpe_ratio``.
    """
    ex = _excess(returns, rf_annual, periods_per_year)
    if ex.size < max(2, min_periods):
        return math.nan
    mu = float(ex.mean())
    dd = math.sqrt(float(np.mean(np.minimum(ex, 0.0) ** 2)))
    if dd == 0.0:
        return math.inf if mu > 0 else math.nan
    return mu / dd * math.sqrt(periods_per_year)


def sharpe_se(sr: float, T: int, skew: float, kurt: float) -> float:
    """Analytic standard error of a *per-period* Sharpe ratio (Mertens 2002; Lo 2002 for IID).

    ``SE = sqrt((1 + SR^2/2 - skew*SR + (kurt - 3)*SR^2/4) / T)`` with ``kurt`` the raw
    (non-excess) kurtosis. Fat tails and negative skew widen it. Multiply by ``sqrt(P)``
    for the SE of the annualised ratio.
    """
    if T <= 0:
        return math.nan
    v = (1.0 + 0.5 * sr * sr - skew * sr + 0.25 * (kurt - 3.0) * sr * sr) / T
    return math.sqrt(v) if v >= 0 else math.nan


def _skew_kurt(x: np.ndarray) -> tuple[float, float]:
    """Moment-estimator skewness and raw kurtosis (the inputs ``sharpe_se`` expects)."""
    xc = x - x.mean()
    m2 = float(np.mean(xc**2))
    if m2 == 0.0:
        return math.nan, math.nan
    return float(np.mean(xc**3)) / m2**1.5, float(np.mean(xc**4)) / m2**2


# --------------------------------------------------------------------------- drawdown

@dataclass(frozen=True, slots=True)
class DrawdownStats:
    """Deepest drawdown episode plus the longest one.

    ``duration`` is peak -> recovery for the deepest episode, or peak -> end of data
    with ``censored=True`` if it never recovered (a lower bound). ``longest_duration``
    is the longest episode of any depth, with its own censoring flag.
    """

    max_dd: float
    peak_t: pd.Timestamp | None
    trough_t: pd.Timestamp | None
    recovery_t: pd.Timestamp | None
    duration: pd.Timedelta
    peak_to_trough: pd.Timedelta
    censored: bool
    longest_duration: pd.Timedelta
    longest_censored: bool


def drawdown_series(equity: pd.Series) -> pd.Series:
    """``E_t / max_{s<=t} E_s - 1`` (0 at a high-water mark, -0.1 when 10% below it)."""
    eq = equity.dropna()
    if (eq <= 0).any():
        raise ValueError("drawdowns need strictly positive equity")
    return (eq / eq.cummax() - 1.0).rename("drawdown")


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    """Depth, timing and duration of drawdown episodes (see ``DrawdownStats``).

    An episode is a maximal run of marks strictly below the running maximum. Its peak is
    the mark just before the run (the last time at the high-water mark), its recovery the
    first mark after the run (back at or above the peak), and its trough the deepest mark.
    """
    zero = pd.Timedelta(0)
    dd = drawdown_series(equity)
    if dd.empty:
        return DrawdownStats(math.nan, None, None, None, zero, zero, False, zero, False)
    under = dd.to_numpy() < 0.0
    if not under.any():
        return DrawdownStats(0.0, None, None, None, zero, zero, False, zero, False)
    t = dd.index
    n = len(dd)
    edges = np.diff(np.concatenate(([0], under.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)        # first underwater mark (>= 1: dd[0] == 0)
    ends = np.flatnonzero(edges == -1)         # first mark back at the high, n if never
    censored = ends == n
    durations = t[np.minimum(ends, n - 1)] - t[starts - 1]
    i_long = int(durations.argmax())

    i_trough = int(np.argmin(dd.to_numpy()))
    ep = int(np.searchsorted(starts, i_trough, side="right") - 1)
    peak_t, trough_t = t[starts[ep] - 1], t[i_trough]
    return DrawdownStats(
        max_dd=float(dd.iloc[i_trough]),
        peak_t=peak_t,
        trough_t=trough_t,
        recovery_t=None if censored[ep] else t[ends[ep]],
        duration=durations[ep],
        peak_to_trough=trough_t - peak_t,
        censored=bool(censored[ep]),
        longest_duration=durations[i_long],
        longest_censored=bool(censored[i_long]),
    )


def max_drawdown_from_returns(returns: np.ndarray | Sequence[float]) -> float:
    """Maximum drawdown of the equity path ``prod(1 + r)`` started at 1 (a bootstrap statistic)."""
    eq = np.cumprod(np.concatenate(([1.0], _clean(returns) + 1.0)))
    return float(np.min(eq / np.maximum.accumulate(eq) - 1.0))


# --------------------------------------------------------------------------- annualised

@dataclass(frozen=True, slots=True)
class AnnualisedStat:
    """An annualised figure with the sample span it was extrapolated from.

    ``short_sample`` is set below 90 days: annualising a few weeks raises noise to a large
    power, so such values are labelled and should not be headlined.
    """

    value: float
    span_days: float
    short_sample: bool


def cagr(equity: pd.Series) -> AnnualisedStat:
    """Compound annual growth ``(E_end / E_start)^(365 d / span) - 1``.

    A 365-day year matches P = 365 in the Sharpe. NaN for fewer than two marks or a zero
    span; ``inf`` if the extrapolation overflows (tiny spans).
    """
    eq = equity.dropna()
    if len(eq) < 2:
        return AnnualisedStat(math.nan, 0.0, True)
    span = eq.index[-1] - eq.index[0]
    span_days = span / pd.Timedelta(days=1)
    if span <= pd.Timedelta(0):
        return AnnualisedStat(math.nan, span_days, True)
    with np.errstate(over="ignore"):
        value = float(np.expm1(math.log(eq.iloc[-1] / eq.iloc[0]) / (span / YEAR)))
    return AnnualisedStat(value, span_days, span < SHORT_SAMPLE)


def calmar(equity: pd.Series) -> AnnualisedStat:
    """Calmar ratio ``CAGR / |max drawdown|`` (NaN without a drawdown), same span flag as ``cagr``."""
    g = cagr(equity)
    dd = drawdown_series(equity)
    mdd = float(dd.min()) if not dd.empty else math.nan
    value = g.value / abs(mdd) if mdd < 0 else math.nan
    return AnnualisedStat(value, g.span_days, g.short_sample)


# --------------------------------------------------------------------------- trades

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Unlike the normal (Wald) interval it stays inside [0, 1] and keeps its coverage for
    the small trade counts typical here: centre ``(p + z^2/2n) / (1 + z^2/n)``, half-width
    ``z / (1 + z^2/n) * sqrt(p(1-p)/n + z^2/4n^2)``.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return math.nan, math.nan
    p = k / n
    z2n = z * z / n
    centre = (p + z2n / 2.0) / (1.0 + z2n)
    half = z / (1.0 + z2n) * math.sqrt(p * (1.0 - p) / n + z2n / (4.0 * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def profit_factor(pnl: np.ndarray | Sequence[float]) -> float:
    """Gross profit / gross loss. ``inf`` with wins and no losses, NaN with neither."""
    x = _clean(pnl)
    gains = float(x[x > 0].sum())
    losses = -float(x[x < 0].sum())
    if losses == 0.0:
        return math.inf if gains > 0 else math.nan
    return gains / losses


def _group_stats(pnl: np.ndarray) -> dict[str, Any]:
    n = int(pnl.size)
    return {
        "n_trades": n,
        "total_pnl": float(pnl.sum()),
        "mean_pnl": float(pnl.mean()) if n else math.nan,
        "hit_rate": float(np.mean(pnl > 0)) if n else math.nan,
    }


def trade_stats(trades: pd.DataFrame | Iterable[Any]) -> dict[str, Any]:
    """Per-trade statistics from a frame (or records) with ``pnl_net`` and ``holding_s``.

    A win is ``pnl_net > 0``; break-even trades count in ``n_trades`` but not as wins.
    Optional columns add: ``entry_cost`` -> mean return on cost; ``outside_band_at_entry``
    -> ``by_entry`` split into ``convergence`` (entered inside the no-arbitrage band, so
    any profit needed the sum to swing across the band) and ``outside_band`` (the entry
    itself was an executable arbitrage); ``exit_reason`` -> counts; ``strategy`` ->
    per-strategy stats. ``anecdotal`` is set below 30 trades.
    """
    df = _trades_frame(trades)
    pnl = df["pnl_net"].to_numpy(dtype=float) if "pnl_net" in df else np.empty(0)
    n = int(pnl.size)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    ci_lo, ci_hi = wilson_interval(int(wins.size), n)
    holding = df["holding_s"].to_numpy(dtype=float) if "holding_s" in df and n else np.empty(0)
    out: dict[str, Any] = {
        "n_trades": n,
        "n_wins": int(wins.size),
        "n_losses": int(losses.size),
        "hit_rate": wins.size / n if n else math.nan,
        "hit_rate_ci_lo": ci_lo,
        "hit_rate_ci_hi": ci_hi,
        "total_pnl": float(pnl.sum()),
        "mean_pnl": float(pnl.mean()) if n else math.nan,
        "avg_win": float(wins.mean()) if wins.size else math.nan,
        "avg_loss": float(losses.mean()) if losses.size else math.nan,
        "profit_factor": profit_factor(pnl),
        "holding_mean_s": float(np.nanmean(holding)) if holding.size else math.nan,
        "holding_median_s": float(np.nanmedian(holding)) if holding.size else math.nan,
        "anecdotal": n < MIN_TRADES,
    }
    if n and "entry_cost" in df:
        cost = df["entry_cost"].to_numpy(dtype=float)
        ok = cost > 0
        out["mean_return_on_cost"] = float(np.mean(pnl[ok] / cost[ok])) if ok.any() else math.nan
    if n and "outside_band_at_entry" in df:
        outside = df["outside_band_at_entry"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out["by_entry"] = {"convergence": _group_stats(pnl[~outside]),
                           "outside_band": _group_stats(pnl[outside])}
    if n and "exit_reason" in df:
        counts = df["exit_reason"].fillna("none").astype(str).value_counts(sort=True)
        out["exit_reasons"] = {str(k): int(v) for k, v in counts.items()}
    if n and "strategy" in df:
        names = df["strategy"].astype(str).to_numpy()
        out["by_strategy"] = {str(s): _group_stats(pnl[names == s]) for s in sorted(set(names))}
    return out


def _time_weighted_mean(s: pd.Series) -> float:
    """LOCF time average: each value holds until the next timestamp (the last one for 0 s)."""
    if len(s) < 2:
        return math.nan
    dt = (s.index[1:] - s.index[:-1]).total_seconds().to_numpy()
    total = float(dt.sum())
    return float(np.dot(dt, s.to_numpy(dtype=float)[:-1])) / total if total > 0 else math.nan


def exposure(positions: Sequence[Any] | pd.DataFrame | pd.Series) -> float:
    """Fraction of wall-clock time with a non-flat position.

    ``positions`` is a sequence of points (or a DataFrame) with ``t_ns`` and ``side``, or
    a side Series on a time index. Each side holds until the next timestamp (LOCF).
    """
    side = _series_from_points(positions, "side")
    return _time_weighted_mean((side != 0).astype(float))


def turnover(trades: pd.DataFrame | Iterable[Any], equity: pd.Series) -> float:
    """Annual turnover ``sum |traded notional| / time-weighted mean equity / years``.

    Notional per trade is ``|entry_cost| + |exit_proceeds|`` (exit proceeds are included
    when the column exists, so a round trip counts both legs).
    """
    df = _trades_frame(trades)
    if "entry_cost" not in df:
        raise ValueError("turnover needs an 'entry_cost' column")
    notional = np.abs(df["entry_cost"].to_numpy(dtype=float))
    if "exit_proceeds" in df:
        notional = notional + np.abs(df["exit_proceeds"].fillna(0.0).to_numpy(dtype=float))
    eq = equity.dropna()
    if len(eq) < 2:
        return math.nan
    years = (eq.index[-1] - eq.index[0]) / YEAR
    mean_eq = _time_weighted_mean(eq)
    if not years > 0 or not mean_eq > 0:
        return math.nan
    return float(np.nansum(notional)) / mean_eq / years


# --------------------------------------------------------------------------- uncertainty

def _stationary_indices(rng: np.random.Generator, n: int, rows: int, p_new: float) -> np.ndarray:
    """Politis-Romano resample indices, shape ``(rows, n)``.

    Each position starts a new block with probability ``p_new`` (so block lengths are
    geometric with mean ``1/p_new``) at a uniform random index; otherwise it continues the
    current block, wrapping circularly. Vectorised: the latest block start ``j <= t`` is
    found with a running maximum and position ``t`` reads ``(start_j + t - j) mod n``.
    """
    start = rng.integers(0, n, size=(rows, n))
    new = rng.random((rows, n)) < p_new
    new[:, 0] = True
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    return (np.take_along_axis(start, last, axis=1) + (pos - last)) % n


def stationary_bootstrap(returns: pd.Series | np.ndarray | Sequence[float],
                         stat: Callable[[np.ndarray], float], *, n_boot: int = 2000,
                         mean_block: float, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile ``1 - alpha`` confidence interval of ``stat`` by the stationary bootstrap.

    Resampling blocks of geometric length (mean ``mean_block`` periods) keeps the serial
    dependence of returns that an IID bootstrap would destroy, so the interval is not
    spuriously narrow for autocorrelated mark-to-market P&L. Deterministic for a given
    ``seed``. Resamples where ``stat`` is NaN are dropped; ``(nan, nan)`` if all are.
    """
    if mean_block < 1:
        raise ValueError("mean_block must be >= 1 period")
    x = _clean(returns)
    n = x.size
    if n < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    batch = max(1, min(n_boot, _BOOT_CELLS // n))
    for b0 in range(0, n_boot, batch):
        idx = _stationary_indices(rng, n, min(batch, n_boot - b0), 1.0 / mean_block)
        for j, row in enumerate(idx):
            stats[b0 + j] = stat(x[row])
    ok = stats[~np.isnan(stats)]
    if ok.size < n_boot:
        log.debug("stationary_bootstrap: %d of %d resamples gave NaN", n_boot - ok.size, n_boot)
    if ok.size == 0:
        return math.nan, math.nan
    lo, hi = np.percentile(ok, [50.0 * alpha, 100.0 - 50.0 * alpha])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- summary

def _jsonable(v: Any) -> Any:
    """Plain JSON types: non-finite floats -> None, timestamps -> ISO, timedeltas -> seconds."""
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, pd.Timedelta):
        return v.total_seconds()
    raise TypeError(f"not JSON-serialisable: {type(v).__name__}")


def _side_source(equity: Any) -> Any | None:
    """The points themselves if they carry a ``side`` field (for ``exposure``), else None."""
    if isinstance(equity, pd.Series):
        return None
    if isinstance(equity, pd.DataFrame):
        return equity if "side" in equity else None
    rows = list(equity)
    return rows if rows and all(_has_field(p, "side") for p in rows) else None


def summarize(equity: Sequence[Any] | pd.DataFrame | pd.Series,
              trades: pd.DataFrame | Iterable[Any] | None = None, *,
              rf_annual: float = DEFAULT_RF, freq: str | None = None,
              side: pd.Series | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Headline metrics as a JSON-serialisable dict (``results/metrics.json``).

    ``equity`` is anything ``equity_from_points`` accepts (liquidation marks). The return
    frequency defaults to daily (P = 365) when the sample spans at least 30 days and to
    hourly (P = 8760) otherwise; ``sharpe_freq`` records the choice. Sharpe-type figures
    and their bootstrap CIs are None ("n/a (too short)") below 30 periods. The bootstrap
    mean block is ``max(5 days, mean holding time)`` in periods, capped at a quarter of
    the sample so resamples still differ. Drawdowns use the raw marks; the drawdown CI uses
    the resampled returns. Exposure comes from ``side`` or from points that carry ``side``.
    Non-finite values (e.g. a profit factor with no losses) are emitted as None; every
    caveat that applies is spelled out in ``warnings``.
    """
    points = equity if isinstance(equity, (pd.Series, pd.DataFrame)) else list(equity)
    eq = equity_from_points(points)
    if eq.empty:
        raise ValueError("summarize needs at least one equity mark")
    span = eq.index[-1] - eq.index[0]
    auto_freq = freq is None
    freq = freq or ("1D" if span >= DAILY_MIN_SPAN else "1h")
    P = periods_per_year(freq)
    rets = period_returns(eq, freq)
    n = len(rets)
    too_short = n < MIN_PERIODS
    warnings: list[str] = []
    if auto_freq and freq == "1h":
        warnings.append(f"span {span / pd.Timedelta(days=1):.1f} d < 30 d: Sharpe from hourly returns (P=8760)")
    if too_short:
        warnings.append(f"Sharpe n/a (too short): {n} {freq} periods < {MIN_PERIODS}")

    sr = sharpe_ratio(rets, rf_annual=rf_annual, periods_per_year=P)
    ex = _excess(rets, rf_annual, P)
    skew, kurt = _skew_kurt(ex) if n >= 2 else (math.nan, math.nan)

    tdf = _trades_frame(trades) if trades is not None else None
    ts = trade_stats(tdf) if tdf is not None else None
    hold_s = ts["holding_mean_s"] if ts is not None else math.nan
    period_s = _freq_delta(freq).total_seconds()
    block = max(5 * 86400.0, hold_s if math.isfinite(hold_s) else 0.0) / period_s
    block = float(min(max(block, 1.0), max(1.0, n / 4.0)))
    if too_short:
        sr_ci = mdd_ci = (math.nan, math.nan)
    else:
        def boot_sharpe(x: np.ndarray) -> float:
            return sharpe_ratio(x, rf_annual=rf_annual, periods_per_year=P, min_periods=0)
        sr_ci = stationary_bootstrap(rets, boot_sharpe, n_boot=n_boot, mean_block=block, seed=seed)
        mdd_ci = stationary_bootstrap(rets, max_drawdown_from_returns, n_boot=n_boot,
                                      mean_block=block, seed=seed)

    dd = drawdown_stats(eq)
    if dd.censored:
        warnings.append("max drawdown unrecovered at end of data: duration is a lower bound (censored)")
    g, cm = cagr(eq), calmar(eq)
    if g.short_sample:
        warnings.append(f"CAGR/Calmar annualised from {g.span_days:.1f} d < 90 d (short_sample)")

    out: dict[str, Any] = {
        "start": eq.index[0],
        "end": eq.index[-1],
        "span_days": span / pd.Timedelta(days=1),
        "n_marks": len(eq),
        "sharpe_freq": freq,
        "periods_per_year": P,
        "n_periods": n,
        "too_short": too_short,
        "rf_annual": rf_annual,
        "sharpe": sr,
        "sharpe_rf0": sharpe_ratio(rets, rf_annual=0.0, periods_per_year=P),
        "sharpe_lo_adjusted": sharpe_ratio(rets, rf_annual=rf_annual, periods_per_year=P, lo_correction=True),
        "sharpe_display": "n/a (too short)" if too_short else ("n/a" if math.isnan(sr) else f"{sr:.2f}"),
        "sharpe_se": sharpe_se(sr / math.sqrt(P), n, skew, kurt) * math.sqrt(P) if not too_short else math.nan,
        "sharpe_ci95_lo": sr_ci[0],
        "sharpe_ci95_hi": sr_ci[1],
        "sortino": sortino_ratio(rets, rf_annual=rf_annual, periods_per_year=P),
        "bootstrap_mean_block": block,
        "bootstrap_n": n_boot,
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "max_dd": dd.max_dd,
        "max_dd_ci95_lo": mdd_ci[0],
        "max_dd_ci95_hi": mdd_ci[1],
        "max_dd_peak": dd.peak_t,
        "max_dd_trough": dd.trough_t,
        "max_dd_recovery": dd.recovery_t,
        "max_dd_duration_s": dd.duration,
        "max_dd_censored": dd.censored,
        "peak_to_trough_s": dd.peak_to_trough,
        "longest_dd_duration_s": dd.longest_duration,
        "longest_dd_censored": dd.longest_censored,
        "cagr": g.value,
        "calmar": cm.value,
        "short_sample": g.short_sample,
        "exposure": None,
        "turnover": None,
        "trades": ts,
    }
    side_src = side if side is not None else _side_source(points)
    if side_src is not None:
        out["exposure"] = exposure(side_src)
    if ts is not None:
        if ts["anecdotal"]:
            warnings.append(f"anecdotal: {ts['n_trades']} trades < {MIN_TRADES}")
        if "entry_cost" in tdf:
            out["turnover"] = turnover(tdf, eq)
    out["warnings"] = warnings
    return _jsonable(out)
