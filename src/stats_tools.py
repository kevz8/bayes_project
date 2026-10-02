"""Statistics toolkit for the research notebooks (01 sum-to-one EDA, 02 cross-market OLS).

Keeping the statistics here keeps the notebooks thin and makes every number they print
testable. pandas, SciPy and statsmodels are allowed (offline research code, not hot path).

Data preparation
----------------
* **As-of LOCF on a fixed grid.** Irregular ticks make an AR coefficient meaningless (φ
  depends on Δt), so quotes are put on a regular clock grid. The value at grid time ``g`` is
  the last update with ``t <= g``: never interpolated (interpolation reads the future) and
  never looking ahead.
* **Stale quotes vs recording gaps.** An unchanged CLOB book is still a live, executable
  quote, so per-leg staleness is *not* missing data (it is reported as ``age_s``). Our own
  outages are: when no leg at all updates for more than ``max_gap`` the grid points are
  marked as a gap (segment ``-1``), a new segment starts at the next update and nothing is
  carried across the gap. Rows with ``clean=False`` (book known stale, explicit session-end
  markers) yield NaN until the leg's next clean update.
* **Basket sums.** ``S_mid`` uses the dust policy of ``arb_engine.BasketState`` (no bid and
  ``ask <= dust_max_ask`` -> mid ``ask/2``, counted; any other one-sided leg invalidates
  ``S_mid`` rather than inventing a value). Executable sums are strict: an empty bid counts
  0 and an empty ask counts 1, so they never overstate an edge.

NaN policy
----------
Leading/trailing NaNs are always stripped. Estimators that are sums over complete
observations or complete ``(x_{t-q}, x_t)`` pairs (OLS, means, the variogram, the AR(1) fit,
rolling OLS) simply skip incomplete ones, which is exact across recording gaps. Tests whose
null distribution depends on the time ordering (ADF, KPSS, variance ratio, Engle-Granger,
Johansen) need a contiguous sample and raise ``ValueError`` on interior NaNs instead of
silently splicing across a gap; test one segment at a time or ``dropna()`` explicitly.

HAC standard errors
-------------------
Every Newey-West (Bartlett kernel) standard error uses, unless given, the Andrews (1991)
AR(1) plug-in lag ``1.1447 (α T)^{1/3}`` with ``α = 4ρ²/((1-ρ)²(1+ρ)²)``. The textbook
fixed rule ``4 (T/100)^{2/9}`` ignores persistence and badly under-covers series such as
``S_mid`` whose autocorrelation decays over hundreds of grid steps.
"""
from __future__ import annotations

import inspect
import logging
import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .events import TOB_COLUMNS
from .metrics import stationary_bootstrap

log = logging.getLogger(__name__)

__all__ = [
    "TOB_FIELDS", "resample_locf", "prices_wide", "basket_sums",
    "adf_test", "kpss_test", "adf_kpss_verdict", "variance_ratio", "variogram",
    "ar1_fit", "hac_mean_test", "holm", "engle_granger", "johansen",
    "aggregate_tail_legs", "identity_regression", "rolling_ols_spread",
    "identity_residual_stats", "walk_forward_splits", "plateau_select",
]

TOB_FIELDS = ("bid", "ask", "bid_sz", "ask_sz", "mid", "age_s", "clean")
_PRICE_FIELDS = ("bid", "ask", "bid_sz", "ask_sz")
_MIN_TEST_N = 20  # shortest sample a unit-root / cointegration test is run on

# statsmodels 0.15 warns that adfuller/kpss will return result objects; pin the tuple API.
_ADF_KW = {"result_object": False} if "result_object" in inspect.signature(adfuller).parameters else {}
_KPSS_KW = {"result_object": False} if "result_object" in inspect.signature(kpss).parameters else {}


# --------------------------------------------------------------------------- input helpers

def _values(x: Any) -> np.ndarray:
    a = x.to_numpy(dtype=float) if isinstance(x, (pd.Series, pd.Index)) else np.asarray(x, dtype=float)
    if a.ndim == 2 and 1 in a.shape:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {a.shape}")
    return a


def _strip_ends(mask: np.ndarray) -> slice:
    """Slice between the first and last True of ``mask`` (empty slice if none)."""
    idx = np.flatnonzero(mask)
    return slice(0, 0) if idx.size == 0 else slice(int(idx[0]), int(idx[-1]) + 1)


def _contiguous(x: Any, name: str = "x", min_n: int = _MIN_TEST_N) -> np.ndarray:
    """Finite 1-D sample with leading/trailing NaNs stripped; interior NaNs are an error."""
    a = _values(x)
    a = a[_strip_ends(np.isfinite(a))]
    bad = int(np.count_nonzero(~np.isfinite(a)))
    if bad:
        raise ValueError(
            f"{name} has {bad} interior NaN/inf values (recording gaps?): test one segment at a "
            "time, or dropna() explicitly if splicing across the gaps is intended")
    if a.size < min_n:
        raise ValueError(f"{name} too short: {a.size} finite observations < {min_n}")
    if np.ptp(a) == 0.0:
        raise ValueError(f"{name} is constant")
    return a


def _contiguous_frame(df: pd.DataFrame, min_n: int = _MIN_TEST_N) -> pd.DataFrame:
    """Rows of ``df`` with leading/trailing incomplete rows stripped; interior gaps raise."""
    full = np.isfinite(df.to_numpy(dtype=float)).all(axis=1)
    out = df.iloc[_strip_ends(full)]
    bad = int(np.count_nonzero(~np.isfinite(out.to_numpy(dtype=float)).all(axis=1)))
    if bad:
        raise ValueError(
            f"{bad} interior rows contain NaN (recording gaps?): test one segment at a time, "
            "or dropna() explicitly if splicing across the gaps is intended")
    if len(out) < min_n:
        raise ValueError(f"sample too short: {len(out)} complete rows < {min_n}")
    const = [str(c) for c in out.columns if np.ptp(out[c].to_numpy(dtype=float)) == 0.0]
    if const:
        raise ValueError(f"constant column(s): {const}")
    return out


def _pair_frame(y: Any, x: Any) -> pd.DataFrame:
    """``y`` and ``x`` side by side: aligned on the index for Series, positionally otherwise."""
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        return pd.concat([y.astype(float).rename("y"), x.astype(float).rename("x")], axis=1, join="inner")
    ys, xs = _values(y), _values(x)
    if ys.size != xs.size:
        raise ValueError(f"y and x differ in length ({ys.size} vs {xs.size})")
    index = y.index if isinstance(y, pd.Series) else x.index if isinstance(x, pd.Series) else None
    return pd.DataFrame({"y": ys, "x": xs}, index=index)


def _andrews_lags(u: np.ndarray) -> int:
    """Andrews (1991) AR(1) plug-in Bartlett bandwidth, clipped to ``[1, T/4]``."""
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    n = u.size
    if n < 4:
        return 1
    u = u - u.mean()
    den = float(u[:-1] @ u[:-1])
    rho = float(u[1:] @ u[:-1]) / den if den > 0 else 0.0
    rho = min(max(rho, -0.99), 0.99)
    alpha = 4.0 * rho * rho / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    lags = int(math.floor(1.1447 * (alpha * n) ** (1.0 / 3.0)))
    return int(min(max(lags, 1), max(1, n // 4)))


def _two_sided_p(z: float) -> float:
    return float(2.0 * sps.norm.sf(abs(z))) if math.isfinite(z) else math.nan


def _crit_dict(levels: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return {k: float(v) for k, v in zip(levels, values)}


# --------------------------------------------------------------------------- LOCF grids

def _grid_ns(t_first: int, t_last: int, step_ns: int) -> np.ndarray:
    """Epoch-aligned grid points in ``[ceil(t_first), floor(t_last)]``."""
    start = -(-t_first // step_ns) * step_ns
    end = (t_last // step_ns) * step_ns
    return np.arange(start, end + 1, step_ns, dtype=np.int64) if end >= start else np.empty(0, np.int64)


def _grid_index(grid: np.ndarray, step_ns: int) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(grid, unit="ns", utc=True), name="t")
    return pd.DatetimeIndex(idx, freq=pd.Timedelta(step_ns, unit="ns")) if len(idx) > 1 else idx


def _locf_rows(t_ns: np.ndarray, codes: np.ndarray, n_legs: int, step_ns: int, gap_ns: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """As-of row lookup shared by the book and price grids.

    Returns ``(order, grid, segment, rows)``: ``order`` sorts the input by time (stable, so of
    equal timestamps the last input row wins), ``segment[g]`` is the recording segment of grid
    point ``g`` (``-1`` inside a gap) and ``rows[g, k]`` the position (in sorted order) of
    leg ``k``'s last update at or before ``g`` *within the same segment*, or ``-1``.

    Liveness is judged on every row, whatever its leg: a gap is a stretch longer than
    ``gap_ns`` without any update. A grid point more than ``gap_ns`` after the latest update
    is inside the gap - the causal rule a live system would apply.
    """
    order = np.argsort(t_ns, kind="stable")
    t = t_ns[order]
    c = codes[order]
    seg_row = np.zeros(t.size, dtype=np.int64)
    seg_row[np.flatnonzero(np.diff(t) > gap_ns) + 1] = 1
    seg_row = np.cumsum(seg_row)

    grid = _grid_ns(int(t[0]), int(t[-1]), step_ns)
    last_any = np.searchsorted(t, grid, side="right") - 1  # >= 0: grid starts at/after t[0]
    in_gap = (grid - t[last_any]) > gap_ns
    segment = np.where(in_gap, -1, seg_row[last_any])

    rows = np.full((grid.size, n_legs), -1, dtype=np.int64)
    by_leg = np.argsort(c, kind="stable")  # positions grouped by leg, time-sorted within
    bounds = np.searchsorted(c[by_leg], np.arange(n_legs + 1))
    for k in range(n_legs):
        pos = by_leg[bounds[k]:bounds[k + 1]]
        if pos.size == 0:
            continue
        i = np.searchsorted(t[pos], grid, side="right") - 1
        r = pos[np.maximum(i, 0)]
        ok = (i >= 0) & ~in_gap & (seg_row[r] == segment)
        rows[:, k] = np.where(ok, r, -1)
    return order, grid, segment, rows


def _gather(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    out = values[np.maximum(rows, 0)].astype(float)
    out[rows < 0] = np.nan
    return out


def resample_locf(tob_long: pd.DataFrame, legs: Sequence[str] | None = None, grid: str = "1min",
                  max_gap: str = "60s") -> tuple[pd.DataFrame, pd.Series]:
    """Long top-of-book table (``events.TOB_COLUMNS``) -> wide as-of grid.

    Returns ``(wide, segment)``. ``wide`` has a UTC ``DatetimeIndex`` on the epoch-aligned
    ``grid`` between the first and last update, and MultiIndex columns ``(field, leg)`` for
    the fields ``bid, ask, bid_sz, ask_sz, mid, age_s, clean``:

    * prices/sizes are the leg's last *clean* update with ``t <= g`` (NaN = empty side);
    * ``mid`` is ``(bid + ask) / 2`` only when both sides exist (no dust policy here, see
      :func:`basket_sums`);
    * ``age_s`` is ``g`` minus the time of that update - a stale but live quote is data;
    * ``clean`` is False when the leg has no usable quote at ``g``.

    ``segment`` (int) numbers the recording segments; a stretch of more than ``max_gap``
    without an update on *any* leg is a gap (``-1``, all NaN), and no value is carried
    across it - after a gap every leg is NaN until its own next update. ``legs`` selects and
    orders the legs (an absent leg gives NaN columns); default: order of first appearance.
    ``tob_long.attrs`` (e.g. the ``synthetic`` flag) are copied to ``wide.attrs``.
    """
    missing = [c for c in TOB_COLUMNS if c not in tob_long.columns]
    if missing:
        raise ValueError(f"top-of-book table lacks columns {missing}")
    if tob_long.empty:
        raise ValueError("empty top-of-book table")
    step_ns, gap_ns = pd.Timedelta(grid).value, pd.Timedelta(max_gap).value
    if step_ns <= 0 or gap_ns <= 0:
        raise ValueError("grid and max_gap must be positive")

    leg_col = tob_long["leg"].astype(str).to_numpy()
    leg_list = [str(x) for x in (pd.unique(leg_col) if legs is None else legs)]
    if not leg_list or len(set(leg_list)) != len(leg_list):
        raise ValueError("legs must be non-empty and unique")
    absent = sorted(set(leg_list) - set(leg_col))
    if absent:
        log.warning("resample_locf: legs without any update: %s", absent)
    codes = pd.Index(leg_list).get_indexer(leg_col).astype(np.int64)  # -1: leg not selected

    t_ns = tob_long["t_ns"].to_numpy(dtype=np.int64)
    order, grid_ns, segment, rows = _locf_rows(t_ns, codes, len(leg_list), step_ns, gap_ns)
    clean_rows = tob_long["clean"].to_numpy(dtype=bool)[order]
    clean = (rows >= 0) & clean_rows[np.maximum(rows, 0)]
    use = np.where(clean, rows, -1)

    blocks = {f: _gather(tob_long[f].to_numpy(dtype=float)[order], use) for f in _PRICE_FIELDS}
    blocks["mid"] = 0.5 * (blocks["bid"] + blocks["ask"])
    blocks["age_s"] = (grid_ns[:, None] - _gather(t_ns[order].astype(float), use)) / 1e9
    blocks["clean"] = clean
    index = _grid_index(grid_ns, step_ns)
    wide = pd.DataFrame({(f, leg): blocks[f][:, k] for f in TOB_FIELDS for k, leg in enumerate(leg_list)},
                        index=index)
    wide.columns.names = ["field", "leg"]
    wide.attrs.update(tob_long.attrs)
    wide.attrs.update(grid=grid, max_gap=max_gap)
    return wide, pd.Series(segment, index=index, name="segment", dtype=np.int64)


def prices_wide(prices_long: pd.DataFrame, grid: str = "1min", max_gap: str = "2h"
                ) -> tuple[pd.DataFrame, pd.Series]:
    """``/prices-history`` backfill (long: ``t`` unix seconds or ``t_ns``, ``leg``, ``p``)
    -> ``(wide, segment)`` with one column of as-of ``p`` per leg on the grid.

    Same LOCF and gap rules as :func:`resample_locf` (``max_gap`` defaults to 2 h because
    the hourly backfill has one point per hour). These are last-trade style prices, not
    executable quotes - label anything derived from them accordingly.
    """
    if "t_ns" in prices_long.columns:
        t_ns = prices_long["t_ns"].to_numpy(dtype=np.int64)
    elif "t" in prices_long.columns:
        t = prices_long["t"].to_numpy()
        t_ns = (t.astype(np.int64) * 1_000_000_000 if np.issubdtype(t.dtype, np.integer)
                else np.round(t.astype(float) * 1e9).astype(np.int64))
    else:
        raise ValueError("prices table needs a 't' (unix seconds) or 't_ns' column")
    for c in ("leg", "p"):
        if c not in prices_long.columns:
            raise ValueError(f"prices table lacks column {c!r}")
    if prices_long.empty:
        raise ValueError("empty prices table")
    step_ns, gap_ns = pd.Timedelta(grid).value, pd.Timedelta(max_gap).value
    leg_col = prices_long["leg"].astype(str).to_numpy()
    leg_list = [str(x) for x in pd.unique(leg_col)]
    codes = pd.Index(leg_list).get_indexer(leg_col).astype(np.int64)
    order, grid_ns, segment, rows = _locf_rows(t_ns, codes, len(leg_list), step_ns, gap_ns)
    p = _gather(prices_long["p"].to_numpy(dtype=float)[order], rows)
    index = _grid_index(grid_ns, step_ns)
    wide = pd.DataFrame(p, index=index, columns=pd.Index(leg_list, name="leg"))
    wide.attrs.update(prices_long.attrs)
    wide.attrs.update(grid=grid, max_gap=max_gap)
    return wide, pd.Series(segment, index=index, name="segment", dtype=np.int64)


def basket_sums(wide: pd.DataFrame, *, dust_max_ask: float = 0.02) -> pd.DataFrame:
    """Basket sums per grid row from a :func:`resample_locf` frame.

    Columns: ``s_mid, s_bid, s_ask, valid, spread_sum, n_imputed, max_age_s``.

    * ``S_mid`` per leg: two-sided -> mid; no bid and ``ask <= dust_max_ask`` -> ``ask/2``
      (imputed, counted in ``n_imputed``); otherwise the row is invalid and ``s_mid`` NaN.
      This matches ``arb_engine.BasketState`` so EDA and the engine see the same series.
    * ``s_bid``/``s_ask`` are strict: an empty bid counts 0 and an empty ask counts 1. A leg
      whose book is *unknown* (``clean`` False: gap, stale book) is not an empty book, so the
      executable sums are NaN there rather than a fabricated conservative bound.
    * ``spread_sum = s_ask - s_bid`` is the width of the no-arbitrage band.
    """
    fields = set(wide.columns.get_level_values(0)) if isinstance(wide.columns, pd.MultiIndex) else set()
    if not {"bid", "ask"} <= fields:
        raise ValueError("basket_sums needs a wide frame with ('bid', leg) and ('ask', leg) columns")
    legs = list(wide["bid"].columns)
    bid = wide["bid"].to_numpy(dtype=float)
    ask = wide["ask"][legs].to_numpy(dtype=float)
    clean = (wide["clean"][legs].to_numpy(dtype=bool) if "clean" in fields
             else np.ones(bid.shape, dtype=bool))
    hb, ha = ~np.isnan(bid), ~np.isnan(ask)
    two = clean & hb & ha
    dust = clean & ~hb & ha & (ask <= dust_max_ask)
    valid = (two | dust).all(axis=1)
    mid = np.where(two, 0.5 * (bid + ask), np.where(dust, 0.5 * ask, 0.0))
    known = clean.all(axis=1)
    s_bid = np.where(known, np.where(hb, bid, 0.0).sum(axis=1), np.nan)
    s_ask = np.where(known, np.where(ha, ask, 1.0).sum(axis=1), np.nan)
    if "age_s" in fields:
        max_age = wide["age_s"][legs].max(axis=1, skipna=True).to_numpy(dtype=float)
    else:
        max_age = np.full(len(wide), np.nan)
    out = pd.DataFrame({
        "s_mid": np.where(valid, mid.sum(axis=1), np.nan),
        "s_bid": s_bid,
        "s_ask": s_ask,
        "valid": valid,
        "spread_sum": s_ask - s_bid,
        "n_imputed": dust.sum(axis=1).astype(np.int64),
        "max_age_s": max_age,
    }, index=wide.index)
    out.attrs.update(wide.attrs)
    return out


# --------------------------------------------------------------------------- unit roots

def adf_test(x: Any, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test (H0: unit root) via ``statsmodels.adfuller``.

    ``regression='c'`` is the headline for basket sums (reversion to a constant baseline);
    ``'ct'`` is the robustness variant because the overround drifts as long shots decay.
    Returns ``dict(stat, pvalue, lags, nobs, crit, regression)``.
    """
    a = _contiguous(x)
    r = adfuller(a, maxlag=maxlag, regression=regression, autolag=autolag, **_ADF_KW)
    return {"stat": float(r[0]), "pvalue": float(r[1]), "lags": int(r[2]), "nobs": int(r[3]),
            "crit": {k: float(v) for k, v in r[4].items()}, "regression": regression}


def kpss_test(x: Any, regression: str = "c", nlags: str | int = "auto") -> dict:
    """KPSS test (H0: stationarity) via ``statsmodels.kpss``.

    statsmodels only tabulates p-values in ``[0.01, 0.10]`` and warns (``InterpolationWarning``)
    when the statistic is outside the table; the returned ``pvalue`` is then the bound and
    ``pvalue_str`` reads ``'<0.01'`` or ``'>0.10'`` so it is never reported as a point value.
    Returns ``dict(stat, pvalue, pvalue_str, bounded, lags, crit, regression)``.
    """
    a = _contiguous(x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InterpolationWarning)
        stat, p, lags, crit = kpss(a, regression=regression, nlags=nlags, **_KPSS_KW)
    bounded = False
    for w in caught:
        if issubclass(w.category, InterpolationWarning):
            bounded = True
        else:  # only the table-bound warning is consumed; anything else is passed on
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    p = float(p)
    if bounded:
        p_str = "<0.01" if p <= 0.01 else ">0.10"
    else:
        p_str = f"{p:.3f}"
    return {"stat": float(stat), "pvalue": p, "pvalue_str": p_str, "bounded": bounded,
            "lags": int(lags), "crit": {k: float(v) for k, v in crit.items()}, "regression": regression}


def adf_kpss_verdict(adf_p: float, kpss_p: float, alpha: float = 0.05) -> str:
    """Combine the two tests, whose nulls are opposite (ADF: unit root, KPSS: stationary).

    ADF rejects and KPSS does not -> ``'stationary'``; ADF does not and KPSS rejects ->
    ``'unit_root'``; both or neither reject (breaks, fractional integration, low power) or a
    missing p-value -> ``'inconclusive'``.
    """
    if not (math.isfinite(adf_p) and math.isfinite(kpss_p)):
        return "inconclusive"
    adf_rej, kpss_rej = adf_p < alpha, kpss_p < alpha
    if adf_rej and not kpss_rej:
        return "stationary"
    if kpss_rej and not adf_rej:
        return "unit_root"
    return "inconclusive"


def variance_ratio(x: Any, q: int, robust: bool = True) -> tuple[float, float, float]:
    """Lo-MacKinlay (1988) variance ratio ``VR(q) = Var(x_t - x_{t-q}) / (q Var(Δx_t))``.

    Overlapping q-differences with the unbiased denominator ``m = q (T-q+1)(1-q/T)``. Under a
    random walk VR = 1; a stationary AR(1) has ``VR(q) = (1-φ^q)/(q(1-φ))`` -> 0 like 1/q;
    random walk plus quote noise drops to a plateau below 1. The z statistic uses the
    homoskedastic variance ``2(2q-1)(q-1)/(3qT)`` or, with ``robust=True``, the
    heteroskedasticity-consistent ``θ(q)/T = Σ_j [2(q-j)/q]² δ(j) / T``. Returns
    ``(vr, z, two-sided p)``.
    """
    q = int(q)
    if q < 2:
        raise ValueError("q must be >= 2")
    a = _contiguous(x, min_n=q + 3)
    d = np.diff(a)
    n = d.size
    if q >= n:
        raise ValueError(f"q={q} needs more than {q} increments, got {n}")
    mu = (a[-1] - a[0]) / n
    e = d - mu
    s2a = float(e @ e) / (n - 1)
    if s2a == 0.0:
        raise ValueError("increments are constant")
    dq = a[q:] - a[:-q] - q * mu
    m = q * (n - q + 1) * (1.0 - q / n)
    vr = float(dq @ dq) / m / s2a
    if robust:
        e2 = e * e
        j = np.arange(1, q)
        delta = np.array([n * float(e2[k:] @ e2[:-k]) for k in j]) / float(e2.sum()) ** 2
        var = float(np.sum((2.0 * (q - j) / q) ** 2 * delta)) / n
    else:
        var = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)
    z = (vr - 1.0) / math.sqrt(var) if var > 0 else math.nan
    return vr, z, _two_sided_p(z)


def variogram(x: Any, lags: Sequence[int]) -> np.ndarray:
    """``V(q) = Var(x_t - x_{t-q})`` (ddof=1) over complete pairs, for each ``q`` in ``lags``.

    The clearest picture of "legs wander, the sum reverts": V grows linearly in q for a
    random walk (slope σ²) and levels off at ``2γ₀`` for a stationary series. Lags with
    fewer than two complete pairs give NaN.
    """
    a = _values(x)
    out = np.full(len(lags), np.nan)
    for i, q in enumerate(lags):
        q = int(q)
        if q < 1:
            raise ValueError("lags must be >= 1")
        if q >= a.size:
            continue
        d = a[q:] - a[:-q]
        d = d[np.isfinite(d)]
        if d.size >= 2:
            out[i] = float(np.var(d, ddof=1))
    return out


# --------------------------------------------------------------------------- AR(1) / OU

def _ols_slope(a: np.ndarray, b: np.ndarray) -> float:
    am = a - a.mean()
    den = float(am @ am)
    return float(am @ (b - b.mean())) / den if den > 0 else math.nan


def _half_life_steps(phi: float) -> float:
    """``-ln 2 / ln φ`` grid steps; inf for φ >= 1 (no reversion), NaN for φ <= 0."""
    if not math.isfinite(phi) or phi <= 0.0:
        return math.nan
    if phi >= 1.0:
        return math.inf
    return -math.log(2.0) / math.log(phi)


def _half_life_bound(phi: float) -> float:
    """Monotone map of a φ confidence bound to a half-life bound (φ <= 0 -> 0 steps)."""
    if not math.isfinite(phi):
        return math.nan
    return 0.0 if phi <= 0.0 else _half_life_steps(phi)


def ar1_fit(x: Any, dt_s: float, hac_lags: int | None = None, n_boot: int = 0, seed: int = 0,
            mean_block: float | None = None) -> dict:
    """AR(1)/OU fit ``x_t = c + φ x_{t-1} + ε_t`` by OLS on complete consecutive pairs.

    Only pairs with both ``x_{t-1}`` and ``x_t`` finite enter, so recording gaps never create
    a false transition. Derived quantities (``dt_s`` = grid step in seconds):

    * ``mu = c/(1-φ)``, ``sigma_inf = σ_ε/sqrt(1-φ²)`` (stationary sd), both for ``|φ| < 1``;
    * ``half_life_s = -ln2/ln φ · dt_s`` for ``0 < φ < 1`` (inf for φ >= 1, NaN for φ <= 0)
      and ``theta = -ln φ / dt_s`` (OU rate per second);
    * ``half_life_se_s`` by the delta method ``ln2/(φ (ln φ)²) · SE_HAC(φ) · dt_s``;
    * ``ci_boot``: with ``n_boot > 0``, a 95% percentile interval from a stationary block
      bootstrap of the regression pairs (Politis-Romano, mean block ``mean_block`` pairs,
      default ``T^{1/3}``). The interval is computed for φ (``ci_phi_boot``) and mapped
      through the monotone half-life function, so it is exact under that transform and
      never averages infinite half-lives.

    ``bias_note``/``phi_bias``: Kendall's small-sample bias ``E[φ̂ - φ] ≈ -(1+3φ)/T`` biases
    the half-life down; it matters for short windows. If ĥ is about one grid step at every Δ
    you are measuring bid-ask bounce, not dislocation decay - check invariance across grids.
    """
    if not dt_s > 0:
        raise ValueError("dt_s must be positive")
    a = _values(x)
    prev, cur = a[:-1], a[1:]
    ok = np.isfinite(prev) & np.isfinite(cur)
    xp, xc = prev[ok], cur[ok]
    n = xp.size
    if n < 10:
        raise ValueError(f"need at least 10 complete (x[t-1], x[t]) pairs, got {n}")
    if np.ptp(xp) == 0.0:
        raise ValueError("x is constant")
    exog = np.column_stack([np.ones(n), xp])
    ols = sm.OLS(xc, exog).fit()
    lags = int(hac_lags) if hac_lags is not None else _andrews_lags(ols.resid)
    hac = sm.OLS(xc, exog).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    c, phi = (float(v) for v in hac.params)
    phi_se = float(hac.bse[1])
    sigma_eps = math.sqrt(float(ols.ssr) / (n - 2))
    stationary = abs(phi) < 1.0
    reverting = 0.0 < phi < 1.0
    hl = _half_life_steps(phi)
    hl_se = (math.log(2.0) / (phi * math.log(phi) ** 2) * phi_se * dt_s) if reverting else math.nan

    ci_phi = ci = None
    if n_boot > 0:
        mb = float(mean_block) if mean_block is not None else max(1.0, n ** (1.0 / 3.0))

        def stat(ix: np.ndarray) -> float:
            k = ix.astype(np.int64)
            return _ols_slope(xp[k], xc[k])

        lo, hi = stationary_bootstrap(np.arange(n, dtype=float), stat, n_boot=n_boot,
                                      mean_block=mb, seed=seed)
        ci_phi = (lo, hi)
        ci = (_half_life_bound(lo) * dt_s, _half_life_bound(hi) * dt_s)

    bias = -(1.0 + 3.0 * phi) / n
    return {
        "phi": phi, "phi_se": phi_se, "c": c,
        "mu": c / (1.0 - phi) if stationary else math.nan,
        "sigma_eps": sigma_eps,
        "sigma_inf": sigma_eps / math.sqrt(1.0 - phi * phi) if stationary else math.nan,
        "half_life_s": hl * dt_s,
        "half_life_se_s": hl_se,
        "ci_boot": ci, "ci_phi_boot": ci_phi,
        "theta": -math.log(phi) / dt_s if reverting else math.nan,
        "nobs": n, "hac_lags": lags, "phi_bias": bias,
        "bias_note": (f"Kendall small-sample bias E[phi_hat - phi] ~ -(1+3 phi)/T = {bias:+.2g} "
                      f"(T={n}); the half-life is biased down by roughly that much in phi"),
    }


def hac_mean_test(x: Any, mu0: float = 1.0, lags: int | None = None) -> dict:
    """Newey-West test of ``H0: E[x] = mu0`` (e.g. ``E[S_mid] = 1``, the structural overround).

    A plain t-test treats autocorrelated observations as independent and is far too
    confident; the HAC long-run variance (Bartlett kernel, Andrews lag by default) corrects
    that. NaNs are dropped (a mean is a sum over observations). Normal p-value.
    Returns ``dict(mean, t, pvalue, se, lags, nobs)``.
    """
    a = _values(x)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 10:
        raise ValueError(f"need at least 10 finite observations, got {n}")
    lags = int(lags) if lags is not None else _andrews_lags(a)
    res = sm.OLS(a, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    mean, se = float(res.params[0]), float(res.bse[0])
    if not se > 0:
        raise ValueError("zero variance: x is constant")
    t = (mean - mu0) / se
    return {"mean": mean, "t": t, "pvalue": _two_sided_p(t), "se": se, "lags": lags, "nobs": n}


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1, input order).

    ``p_adj(i) = max_{j <= i} min(1, (m - j + 1) p_(j))`` over the sorted p-values; reject at
    level α where ``p_adj < α``. Controls the family-wise error across baskets/pairs without
    independence assumptions. NaNs are excluded from ``m`` and stay NaN.
    """
    p = np.asarray(pvals, dtype=float).ravel()
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    if q.size == 0:
        return out
    if ((q < 0) | (q > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    m = q.size
    order = np.argsort(q, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * q[order]))
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


# --------------------------------------------------------------------------- cointegration

def engle_granger(y: Any, x: Any) -> dict:
    """Engle-Granger cointegration test in both directions (H0: no cointegration).

    ``coint(trend='c', autolag='aic')`` with MacKinnon residual-based p-values (plain ADF
    critical values would over-reject). The two directions can disagree, so both are
    reported together with the static cointegrating regression ``a = alpha + beta b``.
    Inputs are assumed I(1); establish that first.
    """
    df = _contiguous_frame(_pair_frame(y, x))
    out: dict[str, Any] = {"nobs": len(df)}
    for name, dep, ind in (("y_on_x", "y", "x"), ("x_on_y", "x", "y")):
        a, b = df[dep].to_numpy(), df[ind].to_numpy()
        stat, p, crit = tuple(coint(a, b, trend="c", autolag="aic"))[:3]
        beta, alpha = np.polyfit(b, a, 1)
        out[name] = {"stat": float(stat), "pvalue": float(p),
                     "crit": _crit_dict(("1%", "5%", "10%"), crit),
                     "alpha": float(alpha), "beta": float(beta)}
    return out


def johansen(endog: pd.DataFrame, det_order: int = 0, k_ar_diff: int | None = None) -> dict:
    """Johansen trace and max-eigenvalue tests, with the leading vector's angle to ι.

    For a basket of n exhaustive outcomes the legs share ``n-1`` stochastic trends and
    ``Σ p_i`` is stationary, so the prediction is cointegration rank 1 with vector
    ``∝ ι = (1, ..., 1)``; ``angle_to_iota_deg`` (in [0, 90]) measures how close the
    estimate is. Pinned long-shot legs are stationary by themselves and add spurious rank:
    aggregate them first (:func:`aggregate_tail_legs`). ``k_ar_diff=None`` picks ``p - 1``
    (at least 1) where ``p`` is the AIC order of a VAR in levels. Ranks use 5% critical
    values. The test assumes a Gaussian VAR - descriptive under fat tails.
    """
    df = endog if isinstance(endog, pd.DataFrame) else pd.DataFrame(np.asarray(endog, dtype=float))
    df = _contiguous_frame(df.astype(float))
    n, k = df.shape
    if k < 2:
        raise ValueError("johansen needs at least two series")
    if k > 12:
        raise ValueError("statsmodels has critical values for at most 12 series; aggregate legs")
    data = df.to_numpy(dtype=float)
    singular = ("singular system: the series are (near-)exactly collinear, e.g. legs that sum to "
                "a constant; drop one leg or aggregate")
    d = np.diff(data, axis=0)
    if np.linalg.matrix_rank(d - d.mean(axis=0)) < k:
        raise ValueError(singular)
    try:
        if k_ar_diff is None:
            maxlags = int(max(1, min(12, n // (10 * k))))
            sel = VAR(data).select_order(maxlags=maxlags, trend="c")
            k_ar_diff = max(int(sel.selected_orders["aic"]) - 1, 1)
        res = coint_johansen(data, det_order, int(k_ar_diff))
    except np.linalg.LinAlgError as exc:
        raise ValueError(singular) from exc
    lr1, lr2 = np.asarray(res.lr1, float), np.asarray(res.lr2, float)
    cvt, cvm = np.asarray(res.cvt, float), np.asarray(res.cvm, float)
    rank_trace = next((r for r in range(k) if lr1[r] <= cvt[r, 1]), k)
    rank_maxeig = next((r for r in range(k) if lr2[r] <= cvm[r, 1]), k)
    v = np.real(np.asarray(res.evec, dtype=complex)[:, 0])
    v = v / np.linalg.norm(v)
    if v.sum() < 0:
        v = -v
    cos = min(1.0, abs(float(v.sum())) / math.sqrt(k))
    return {
        "rank_trace": int(rank_trace), "rank_maxeig": int(rank_maxeig),
        "lr1": lr1, "cvt": cvt, "lr2": lr2, "cvm": cvm,
        "eig": np.real(np.asarray(res.eig, dtype=complex)),
        "evec0": pd.Series(v, index=df.columns),
        "angle_to_iota_deg": math.degrees(math.acos(cos)),
        "k_ar_diff": int(k_ar_diff), "det_order": det_order, "nobs": n,
    }


def aggregate_tail_legs(wide_mid: pd.DataFrame, max_legs: int = 6) -> pd.DataFrame:
    """Keep the ``max_legs - 1`` legs with the largest mean and sum the rest into ``'rest'``.

    Long shots pinned at the tick floor are (near-)stationary on their own and each adds a
    spurious cointegration rank; their sum is one well-behaved "field" leg. Johansen's
    tables also stop at 12 series. ``'rest'`` is NaN where any aggregated leg is NaN.
    Column order of the kept legs is preserved.
    """
    if max_legs < 2:
        raise ValueError("max_legs must be >= 2")
    if wide_mid.shape[1] <= max_legs:
        return wide_mid.copy()
    if "rest" in wide_mid.columns:
        raise ValueError("a column is already named 'rest'")
    means = wide_mid.mean(axis=0, skipna=True)
    keep = set(means.sort_values(ascending=False, kind="stable").index[: max_legs - 1])
    kept = [c for c in wide_mid.columns if c in keep]
    rest = [c for c in wide_mid.columns if c not in keep]
    out = wide_mid[kept].copy()
    out["rest"] = wide_mid[rest].sum(axis=1, min_count=len(rest))
    return out


# --------------------------------------------------------------------------- regressions

def identity_regression(y: Any, X: Any, lags: int | None = None) -> dict:
    """OLS ``y = alpha + X beta + e`` with Newey-West standard errors.

    Built for the within-basket identity ``m_1 ~ Σ_{j≠1} m_j``: exhaustiveness implies
    ``beta = -1`` and ``alpha = 1 + overround``, so this validates the data more than it
    estimates anything. ``t_beta_eq_minus1``/``p_beta_eq_minus1`` test ``beta = -1`` with HAC
    SEs (levels are persistent; plain OLS SEs would be spuriously tight). Rows with any NaN
    are dropped; ``resid`` is returned on the full input index (NaN on dropped rows) so a
    follow-up AR(1) fit does not splice across them. ``beta``/``beta_se``/... are floats for
    one regressor and Series for several.
    """
    if isinstance(y, pd.Series):
        ys = y.astype(float).rename("y")
    else:
        ys = pd.Series(_values(y), name="y", index=X.index if isinstance(X, (pd.Series, pd.DataFrame)) else None)
    if isinstance(X, pd.DataFrame):
        xf = X.astype(float)
    elif isinstance(X, pd.Series):
        xf = X.astype(float).to_frame(X.name if X.name is not None else "x")
    else:
        arr = np.asarray(X, dtype=float)
        xf = pd.DataFrame(arr.reshape(len(arr), -1), index=ys.index)
        xf.columns = ["x"] if xf.shape[1] == 1 else [f"x{i}" for i in range(xf.shape[1])]
    xf.columns = [str(c) for c in xf.columns]
    if "const" in xf.columns:
        raise ValueError("regressor named 'const' clashes with the intercept")
    df = pd.concat([ys, xf], axis=1, join="inner")
    full_index = df.index
    df = df.dropna()
    k = xf.shape[1]
    if len(df) < k + 10:
        raise ValueError(f"too few complete rows ({len(df)}) for {k} regressor(s)")
    exog = sm.add_constant(df[list(xf.columns)], has_constant="add")
    ols = sm.OLS(df["y"], exog).fit()
    lags = int(lags) if lags is not None else _andrews_lags(ols.resid.to_numpy())
    hac = sm.OLS(df["y"], exog).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    slopes, se = hac.params.iloc[1:], hac.bse.iloc[1:]
    t_m1 = (slopes + 1.0) / se
    p_m1 = t_m1.map(_two_sided_p)

    def one(s: pd.Series) -> Any:
        return float(s.iloc[0]) if k == 1 else s

    return {
        "alpha": float(hac.params.iloc[0]), "alpha_se": float(hac.bse.iloc[0]),
        "beta": one(slopes), "beta_se": one(se),
        "t_beta_eq_minus1": one(t_m1), "p_beta_eq_minus1": one(p_m1),
        "params": hac.params, "se": hac.bse, "r2": float(hac.rsquared),
        "nobs": int(hac.nobs), "hac_lags": lags,
        "resid": hac.resid.reindex(full_index).rename("resid"),
    }


def rolling_ols_spread(y: Any, x: Any, window: int) -> pd.DataFrame:
    """Rolling hedge ratio and spread without look-ahead.

    ``RollingOLS`` estimates at row t include observation t, so the parameters are shifted
    one step: ``spread_t = y_t - alpha_{t-1} - beta_{t-1} x_t`` uses only data up to t-1 to
    hedge t. The returned ``alpha``/``beta`` are the shifted values actually used at t.
    Windows containing a NaN or a constant ``x`` give NaN parameters (no carry across gaps).
    """
    df = _pair_frame(y, x)
    window = int(window)
    if window < 3:
        raise ValueError("window must be >= 3 (strictly more observations than 2 regressors)")
    if window > len(df):
        raise ValueError(f"window {window} exceeds the sample ({len(df)} rows)")
    yv, xv = df["y"].to_numpy(dtype=float), df["x"].to_numpy(dtype=float)
    exog = np.column_stack([np.ones(len(df)), xv])
    fit = RollingOLS(yv, exog, window=window, missing="skip").fit(params_only=True, reset=window)
    params = np.array(fit.params, dtype=float)
    # Enforce the window rules explicitly: statsmodels' "skip" only flags a NaN in the last
    # window-1 rows, and moving-window inner products can leave a constant-x window barely
    # invertible (huge params) instead of singular.
    bad = pd.Series(~(np.isfinite(yv) & np.isfinite(xv)), dtype=float)
    has_nan = bad.rolling(window, min_periods=1).sum().to_numpy() > 0
    xr = pd.Series(xv).rolling(window)
    flat = (xr.max() - xr.min()).to_numpy() == 0.0
    params[has_nan | flat] = np.nan
    used = np.vstack([np.full((1, 2), np.nan), params[:-1]])
    return pd.DataFrame({"alpha": used[:, 0], "beta": used[:, 1],
                         "spread": yv - used[:, 0] - used[:, 1] * xv}, index=df.index)


def identity_residual_stats(resid: Any, dt_s: float) -> dict:
    """Mean, HAC t (H0: mean 0) and AR(1) half-life of an identity residual.

    For pairings with a payoff identity the residual should be zero-mean noise that decays
    fast; analysing it directly is more honest than presenting Engle-Granger p-values on a
    relation that holds by construction.
    """
    m = hac_mean_test(resid, mu0=0.0)
    ar = ar1_fit(resid, dt_s)
    return {"mean": m["mean"], "hac_t": m["t"], "pvalue": m["pvalue"], "se": m["se"],
            "half_life_s": ar["half_life_s"], "phi": ar["phi"], "nobs": m["nobs"]}


# --------------------------------------------------------------------------- model selection

def walk_forward_splits(n_or_index: int | Sequence[Any] | pd.Index, train: float = 0.6, val: float = 0.2,
                        embargo: int | str | pd.Timedelta = 0) -> tuple[slice, slice, slice]:
    """Chronological train / validation / test position slices with embargo gaps.

    Boundaries sit at ``floor(train n)`` and ``floor((train + val) n)``; the first
    ``embargo`` observations after each boundary are dropped (López de Prado's embargo), so
    a rolling window or an open trade straddling a boundary cannot leak information into the
    next split. ``embargo`` is a row count, or a time span (``'2h'``/``Timedelta``) for a
    ``DatetimeIndex``: rows within that span after the last row of the previous split are
    dropped. Use at least ``max(window, max holding time)``.
    """
    if isinstance(n_or_index, (int, np.integer)):
        n, index = int(n_or_index), None
    else:
        index = pd.Index(n_or_index)
        n = len(index)
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("need 0 < train, 0 < val and train + val < 1")
    b1 = int(math.floor(train * n + 1e-9))
    b2 = int(math.floor((train + val) * n + 1e-9))
    if isinstance(embargo, (int, np.integer)):
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        v0, t0 = b1 + int(embargo), b2 + int(embargo)
    else:
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("a time-span embargo needs a DatetimeIndex")
        if not index.is_monotonic_increasing:
            raise ValueError("index must be sorted")
        span = pd.Timedelta(embargo)
        if b1 < 1 or b2 < 1:
            raise ValueError(f"sample of {n} rows too short to split")
        v0 = int(index.searchsorted(index[b1 - 1] + span, side="right"))
        t0 = int(index.searchsorted(index[b2 - 1] + span, side="right"))
    if b1 < 1 or v0 >= b2 or t0 >= n:
        raise ValueError(f"empty split: n={n}, boundaries {b1}/{b2}, embargo {embargo!r} too long?")
    return slice(0, b1), slice(v0, b2), slice(t0, n)


def _py(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def plateau_select(grid_df: pd.DataFrame, metric: str, params: Sequence[str]) -> dict:
    """Pick the configuration at the centre of a stable plateau, not the single best cell.

    Each parameter's sorted distinct values form a grid axis; a cell's neighbourhood is
    every evaluated cell within one step on every axis (3×3 in 2-D, including itself). The
    selected cell maximises the *median* metric of its neighbourhood (ties: own metric),
    so an isolated spike - most likely noise from searching many configurations - loses to
    a broad region of good values. Only cells with a finite own metric are eligible.
    Returns ``dict(params, metric, plateau_score, n_neighbours, n_configs, best_single)``;
    ``n_configs`` (all rows tried) must be reported alongside any selected result.
    """
    params = list(params)
    if not params:
        raise ValueError("params must name at least one column")
    missing = [c for c in [*params, metric] if c not in grid_df.columns]
    if missing:
        raise ValueError(f"grid lacks columns {missing}")
    df = grid_df.reset_index(drop=True)
    if df[params].isna().any().any():
        raise ValueError("parameter columns contain NaN")
    if df.duplicated(subset=params).any():
        raise ValueError("duplicate parameter combinations in the grid")
    pos = np.column_stack([df[p].rank(method="dense").to_numpy(dtype=float) for p in params])
    near = (np.abs(pos[:, None, :] - pos[None, :, :]) <= 1.0).all(axis=2)
    vals = df[metric].to_numpy(dtype=float)
    hood = np.where(near, vals[None, :], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbourhoods -> NaN
        score = np.nanmedian(hood, axis=1)
    eligible = np.flatnonzero(np.isfinite(vals) & np.isfinite(score))
    if eligible.size == 0:
        raise ValueError(f"no finite {metric!r} values in the grid")
    best = int(eligible[np.lexsort((-vals[eligible], -score[eligible]))[0]])
    top = int(eligible[np.argmax(vals[eligible])])
    return {
        "params": {p: _py(df.at[best, p]) for p in params},
        "metric": float(vals[best]),
        "plateau_score": float(score[best]),
        "n_neighbours": int(np.count_nonzero(near[best] & np.isfinite(vals))),
        "n_configs": int(len(df)),
        "best_single": {"params": {p: _py(df.at[top, p]) for p in params}, "metric": float(vals[top])},
    }
