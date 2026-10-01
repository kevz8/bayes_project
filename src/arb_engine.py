"""Streaming basket sums, rolling z-scores and the signal state machine (Phase 3).

Economics
---------
A basket is a set of n mutually exclusive and exhaustive outcomes (a Polymarket negRisk
event): exactly one leg resolves YES. With best YES bid ``b_i`` and ask ``a_i`` we track

* ``S_mid = Σ (a_i + b_i) / 2`` - a statistic only; nobody can trade at it;
* ``S_ask = Σ a_i`` - the cost of one full YES set;
* ``S_bid = Σ b_i`` - YES/NO books are mirrored (``NO ask_i = 1 - b_i``), so one full NO set
  costs ``n - S_bid``.

Whichever leg wins, a full YES set pays exactly 1 and a full NO set pays exactly ``n - 1``
(the NegRisk convert turns a full NO set into ``n - 1`` collateral immediately). Hence

* long basket  (buy one YES of every leg): P&L = ``1 - S_ask - fees``;
* short basket (buy one NO of every leg):  P&L = ``S_bid - 1 - fees``.

The **no-arbitrage band** is ``S_bid <= 1 <= S_ask``; its width ``S_ask - S_bid`` is the sum
of the leg spreads (at least n ticks). The taker fee of an order is
``shares * r * (p(1-p))**e`` (makers pay 0, symmetric in p), so a basket costs about
``F = r Σ p_i(1-p_i) = r (1 - Σ p_i²)`` per side when ``Σ p_i = 1``.

Why z decides *timing* and the executable edge decides *whether to trade*
------------------------------------------------------------------------
The blueprint signal ``z = (S_mid - μ) / σ`` over a rolling window is statistical
convergence, not arbitrage. A taker round trip pays the whole band (Σ spreads) plus an entry
and an exit fee, so a reversion of ``S_mid`` *inside* the band cannot be monetised: profit
needs an entry outside the band (``S_ask < 1`` or ``S_bid > 1`` after fees) or the sum
swinging across the whole band. ``z`` therefore flags *when* S is unusually rich or cheap;
:func:`executable_edges` and :func:`basket_cost_hurdle` decide *whether* the trade clears
costs, typically through the ``gate`` callback of :meth:`SignalStateMachine.step`.

Numerics
--------
Hot-path objects (:class:`RollingStats`, :class:`BasketState`, :class:`ZScoreEngine`,
:class:`SignalStateMachine`) use Python scalars per update and NumPy only for storage, with
O(1) work per tick and periodic exact recomputes that bound floating-point drift. The batch
functions (:func:`zscore_batch`, :func:`positions_from_z`) reproduce the streaming path to
~1e-12 and are causal bit-for-bit, so they can drive fast offline grid searches.
This module never imports pandas.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from .config import Basket, FeeSchedule
from .events import ASK, BID, BookView, Side

__all__ = [
    "Action", "BasketSnapshot", "BasketState", "RollingStats", "Signal", "SignalConfig",
    "SignalStateMachine", "ZConfig", "ZScoreEngine", "ZState", "basket_cost_hurdle",
    "edge_to_cost_ratio", "executable_edges", "positions_from_z", "push_mask",
    "rolling_mean_std", "zscore_batch",
]

log = logging.getLogger(__name__)

_NAN = math.nan
_NEVER_NS = int(np.iinfo(np.int64).min)  # t_last sentinel: leg never observed


# --------------------------------------------------------------------------- rolling stats
class RollingStats:
    """Mean and variance of the last ``window`` pushes in O(1) per push.

    Sliding-window Welford on centred values ``y = x - c``: warm-up uses the Welford add
    step; once the ring buffer is full each push replaces the oldest value ``y_old``::

        r'  = r + (y - y_old) / N                       (r = mean - c)
        M2' = M2 + (y - y_old) (y + y_old - r - r')

    Every ``recompute_every`` pushes (default ``window``; amortised O(1)) an exact two-pass
    recompute re-centres ``c`` on the window mean and keeps the sub-ulp remainder in ``r``.
    Centring matters because a basket sum's level dwarfs its noise (S ≈ 1, σ ≈ 1e-4): the
    uncentred replace step loses about log10(|x̄|/σ) digits, and a running Σx/Σx² twice that
    (it can even return a negative variance). Variance uses ddof=1 by default, like pandas.
    Pushed values must be finite.
    """

    __slots__ = ("window", "recompute_every", "_buf", "_k", "_cnt", "_c", "_r", "_m2", "_since")

    def __init__(self, window: int, recompute_every: int | None = None) -> None:
        if int(window) < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if recompute_every is not None and int(recompute_every) < 1:
            raise ValueError(f"recompute_every must be >= 1, got {recompute_every}")
        self.window = int(window)
        self.recompute_every = int(recompute_every) if recompute_every is not None else self.window
        self._buf = np.zeros(self.window, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self._k = 0
        self._cnt = 0
        self._c = 0.0
        self._r = 0.0
        self._m2 = 0.0
        self._since = 0

    def push(self, x: float) -> None:
        x = float(x)
        k = self._k
        n = self.window
        if self._cnt < n:  # warm-up: buffer slots 0..cnt-1 hold the values in order
            if self._cnt == 0:
                self._c = x
            self._buf[k] = x
            self._cnt += 1
            y = x - self._c
            d = y - self._r
            self._r += d / self._cnt
            self._m2 += d * (y - self._r)
        else:
            c = self._c
            y_old = self._buf.item(k) - c
            self._buf[k] = x
            y = x - c
            r = self._r
            r_new = r + (y - y_old) / n
            self._m2 += (y - y_old) * (y + y_old - r - r_new)
            self._r = r_new
        k += 1
        self._k = 0 if k == n else k
        self._since += 1
        if self._since >= self.recompute_every:
            self._recompute()

    def _recompute(self) -> None:
        v = self._buf[: self._cnt]
        c = float(v.mean())
        d = v - c
        corr = float(d.sum())  # what the rounded first-pass mean missed
        self._c = c
        self._r = corr / self._cnt
        self._m2 = max(float(d @ d) - corr * self._r, 0.0)
        self._since = 0

    @property
    def count(self) -> int:
        return self._cnt

    @property
    def full(self) -> bool:
        return self._cnt == self.window

    def mean(self) -> float:
        return self._c + self._r if self._cnt else _NAN

    def var(self, ddof: int = 1) -> float:
        if self._cnt <= ddof:
            return _NAN
        return (self._m2 if self._m2 > 0.0 else 0.0) / (self._cnt - ddof)

    def std(self, ddof: int = 1) -> float:
        v = self.var(ddof)
        return math.sqrt(v) if v == v else _NAN

    def values(self) -> np.ndarray:
        """Chronological copy of the window (oldest first)."""
        if self._cnt < self.window:
            return self._buf[: self._cnt].copy()
        return np.concatenate((self._buf[self._k:], self._buf[: self._k]))


# --------------------------------------------------------------------------- basket state
_OK, _IMPUTED, _INVALID = 0, 1, 2  # per-leg S_mid status


@dataclass(slots=True)
class BasketSnapshot:
    """Basket sums at one instant.

    ``s_mid`` is NaN unless ``valid``. ``s_bid``/``s_ask`` are always reported and are
    conservative (missing bid = 0, missing ask = 1). ``spread_sum = s_ask - s_bid`` is the
    width of the no-arbitrage band. ``tradable`` is False while any leg is paused;
    ``clean`` is False while any leg's book is known to be stale.
    """

    t_ns: int
    s_mid: float
    s_bid: float
    s_ask: float
    valid: bool
    spread_sum: float
    n_legs: int
    n_imputed: int
    max_age_s: float
    tradable: bool
    clean: bool = True

    @property
    def band_side(self) -> Side:
        """Pre-fee executable arbitrage: SHORT if ``S_bid > 1``, LONG if ``S_ask < 1``."""
        if self.s_bid > 1.0:
            return Side.SHORT_BASKET
        if self.s_ask < 1.0:
            return Side.LONG_BASKET
        return Side.FLAT


class BasketState:
    """Top of book of every YES leg plus incrementally maintained basket sums.

    Executable sums are strict: a missing bid counts 0 and a missing ask counts 1, so
    ``S_bid``/``S_ask`` never overstate an edge. ``S_mid`` policy per leg:

    * both sides present (and the book clean) -> ``mid = (bid + ask) / 2``;
    * a *dust* leg - no bid and ``ask <= dust_max_ask`` - gets ``mid = ask / 2``, flagged in
      ``imputed`` (many-leg baskets routinely have long shots with an empty bid side);
    * anything else (one-sided, empty or unclean) makes ``S_mid`` invalid instead of invented.

    Sums are updated by deltas once per frame (a half-applied multi-leg frame would create a
    false jump) and recomputed exactly with ``math.fsum`` every ``exact_every`` frames.
    ``ticks`` is per-leg metadata kept aligned with the legs (e.g. through
    :meth:`remove_leg`) for callers that derive limit prices from the same state.
    """

    __slots__ = (
        "leg_ids", "leg_index", "n", "ticks", "exact_every", "dust_max_ask",
        "bid", "ask", "bid_sz", "ask_sz", "mid", "t_last",
        "has_bid", "has_ask", "clean", "tradable", "imputed",
        "_cb", "_ca", "_cm", "_status", "_s_bid", "_s_ask", "_s_mid",
        "_n_bad", "_n_imp", "_n_unclean", "_n_untradable", "_since_exact",
    )

    _ARRAYS = ("ticks", "bid", "ask", "bid_sz", "ask_sz", "mid", "t_last",
               "has_bid", "has_ask", "clean", "tradable", "imputed")

    def __init__(
        self,
        leg_ids: Sequence[str],
        ticks: Sequence[float] | None = None,
        *,
        exact_every: int = 256,
        dust_max_ask: float = 0.02,
    ) -> None:
        ids = tuple(str(x) for x in leg_ids)
        if not ids:
            raise ValueError("a basket needs at least one leg")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate leg ids")
        n = len(ids)
        self.leg_ids = ids
        self.leg_index = {aid: i for i, aid in enumerate(ids)}
        self.n = n
        self.ticks = np.full(n, 0.01) if ticks is None else np.asarray(ticks, dtype=np.float64).copy()
        if self.ticks.shape != (n,):
            raise ValueError("ticks must have one entry per leg")
        self.exact_every = max(int(exact_every), 1)
        self.dust_max_ask = float(dust_max_ask)
        self.bid = np.full(n, _NAN)
        self.ask = np.full(n, _NAN)
        self.bid_sz = np.full(n, _NAN)
        self.ask_sz = np.full(n, _NAN)
        self.mid = np.full(n, _NAN)
        self.t_last = np.full(n, _NEVER_NS, dtype=np.int64)
        self.has_bid = np.zeros(n, dtype=bool)
        self.has_ask = np.zeros(n, dtype=bool)
        self.clean = np.zeros(n, dtype=bool)  # never observed = not clean
        self.tradable = np.ones(n, dtype=bool)
        self.imputed = np.zeros(n, dtype=bool)
        # Per-leg contributions to the sums, as Python floats (fast scalar access, exact fsum).
        self._cb = [0.0] * n
        self._ca = [1.0] * n
        self._cm = [0.0] * n
        self._status = [_INVALID] * n
        self._n_bad = n
        self._n_imp = 0
        self._n_unclean = n
        self._n_untradable = 0
        self._since_exact = 0
        self.recompute()

    @classmethod
    def from_basket(cls, basket: Basket, **kwargs: Any) -> BasketState:
        """State for a configured basket; paused/closed legs start non-tradable."""
        st = cls(basket.yes_ids, basket.ticks, **kwargs)
        for i, leg in enumerate(basket.legs):
            if not (leg.active and leg.accepting_orders and not leg.closed):
                st.set_tradable(i, False)
        return st

    # ------------------------------------------------------------------ updates
    def _write_leg(self, i: int, bid: float, ask: float, bid_sz: float, ask_sz: float,
                   t_ns: int, clean: bool, tradable: bool) -> tuple[bool, float, float, float]:
        """Store one leg; update the exact integer counters; return (changed, Δbid, Δask, Δmid)."""
        hb = bid == bid
        ha = ask == ask
        if not clean:
            st, cm = _INVALID, 0.0
        elif hb and ha:
            st, cm = _OK, 0.5 * (bid + ask)
        elif ha and ask <= self.dust_max_ask:
            st, cm = _IMPUTED, 0.5 * ask
        else:
            st, cm = _INVALID, 0.0
        cb = bid if hb else 0.0
        ca = ask if ha else 1.0

        ost = self._status[i]
        ocb, oca, ocm = self._cb[i], self._ca[i], self._cm[i]
        ocl = self.clean.item(i)
        otr = self.tradable.item(i)
        if (ost == _INVALID) != (st == _INVALID):
            self._n_bad += 1 if st == _INVALID else -1
        if (ost == _IMPUTED) != (st == _IMPUTED):
            self._n_imp += 1 if st == _IMPUTED else -1
        if ocl != clean:
            self._n_unclean += -1 if clean else 1
        if otr != tradable:
            self._n_untradable += -1 if tradable else 1

        self._status[i] = st
        self._cb[i], self._ca[i], self._cm[i] = cb, ca, cm
        self.bid[i] = bid
        self.ask[i] = ask
        self.bid_sz[i] = bid_sz if hb else _NAN
        self.ask_sz[i] = ask_sz if ha else _NAN
        self.mid[i] = _NAN if st == _INVALID else cm
        self.t_last[i] = t_ns
        self.has_bid[i] = hb
        self.has_ask[i] = ha
        self.clean[i] = clean
        self.tradable[i] = tradable
        self.imputed[i] = st == _IMPUTED
        changed = (cb != ocb or ca != oca or cm != ocm or st != ost
                   or ocl != clean or otr != tradable)
        return changed, cb - ocb, ca - oca, cm - ocm

    def _apply(self, d_bid: float, d_ask: float, d_mid: float) -> None:
        self._s_bid += d_bid
        self._s_ask += d_ask
        self._s_mid += d_mid
        self._since_exact += 1
        if self._since_exact >= self.exact_every:
            self.recompute()

    def set_leg(self, i: int, bid: float, ask: float, bid_sz: float, ask_sz: float,
                t_ns: int, clean: bool, tradable: bool = True) -> bool:
        """Overwrite leg ``i`` (NaN = empty side) and update the sums. Returns True if any
        reported quantity (sums, validity, imputation, cleanliness, tradability) changed."""
        changed, db, da, dm = self._write_leg(
            i, float(bid), float(ask), float(bid_sz), float(ask_sz), int(t_ns), bool(clean), bool(tradable))
        self._apply(db, da, dm)
        return changed

    def update_from_view(self, view: BookView, changed: Iterable[str], t_ns: int) -> bool:
        """Apply every changed leg of one frame, then update the sums once.

        Ids outside the basket (other baskets, NO tokens) are ignored; tradability is kept.
        Returns True if anything a snapshot reports (other than ages) changed.
        """
        idx = self.leg_index
        t_ns = int(t_ns)
        any_changed = False
        db = da = dm = 0.0
        hit = False
        for aid in changed:
            i = idx.get(aid)
            if i is None:
                continue
            hit = True
            bid, ask = view.top(aid)
            bid, ask = float(bid), float(ask)
            bsz = _top_size(view, aid, BID) if bid == bid else _NAN
            asz = _top_size(view, aid, ASK) if ask == ask else _NAN
            ch, b, a, m = self._write_leg(i, bid, ask, bsz, asz, t_ns, bool(view.is_clean(aid)),
                                          self.tradable.item(i))
            any_changed |= ch
            db += b
            da += a
            dm += m
        if hit:
            self._apply(db, da, dm)
        return any_changed

    def set_tradable(self, i: int, tradable: bool) -> None:
        """Mark leg ``i`` paused (False) or tradable; sums are still reported while paused."""
        tradable = bool(tradable)
        if self.tradable.item(i) != tradable:
            self._n_untradable += -1 if tradable else 1
            self.tradable[i] = tradable

    def remove_leg(self, yes_id: str) -> None:
        """Drop a leg that resolved NO early (e.g. an eliminated team).

        The remaining legs still form a partition, but n changed - a structural break: the
        caller must reset its :class:`ZScoreEngine`.
        """
        i = self.leg_index[yes_id]
        if self.n == 1:
            raise ValueError("cannot remove the last leg of a basket")
        for name in self._ARRAYS:
            setattr(self, name, np.delete(getattr(self, name), i))
        for lst in (self._cb, self._ca, self._cm, self._status):
            del lst[i]
        self.leg_ids = self.leg_ids[:i] + self.leg_ids[i + 1:]
        self.leg_index = {aid: j for j, aid in enumerate(self.leg_ids)}
        self.n -= 1
        self._n_bad = sum(1 for s in self._status if s == _INVALID)
        self._n_imp = sum(1 for s in self._status if s == _IMPUTED)
        self._n_unclean = int(self.n - np.count_nonzero(self.clean))
        self._n_untradable = int(self.n - np.count_nonzero(self.tradable))
        self.recompute()
        log.info("basket leg %s removed (n=%d); z engine must be reset", yes_id, self.n)

    def recompute(self) -> None:
        """Exact sums from the per-leg contributions (``math.fsum``)."""
        self._s_bid = math.fsum(self._cb)
        self._s_ask = math.fsum(self._ca)
        self._s_mid = math.fsum(self._cm)
        self._since_exact = 0

    # ------------------------------------------------------------------ reads
    @property
    def valid(self) -> bool:
        return self._n_bad == 0

    @property
    def s_mid(self) -> float:
        return self._s_mid if self._n_bad == 0 else _NAN

    @property
    def s_bid(self) -> float:
        return self._s_bid

    @property
    def s_ask(self) -> float:
        return self._s_ask

    def snapshot(self, t_ns: int) -> BasketSnapshot:
        valid = self._n_bad == 0
        t_min = int(self.t_last.min())
        age = math.inf if t_min == _NEVER_NS else (int(t_ns) - t_min) / 1e9
        return BasketSnapshot(
            int(t_ns), self._s_mid if valid else _NAN, self._s_bid, self._s_ask, valid,
            self._s_ask - self._s_bid, self.n, self._n_imp, age,
            self._n_untradable == 0, self._n_unclean == 0,
        )


def _top_size(view: BookView, asset_id: str, side: int) -> float:
    _, sizes = view.depth(asset_id, side, 1)
    return float(sizes[0]) if len(sizes) else _NAN


# --------------------------------------------------------------------------- z-score engine
@dataclass(frozen=True, slots=True)
class ZConfig:
    """Rolling z-score settings.

    ``window`` counts pushes (event mode) or grid samples (clock mode); ``min_periods``
    defaults to ``window``. ``sigma_floor`` (default half a 1c tick) stops a quiet window
    from turning one quote flicker into |z| > 2. ``change_eps < 0`` pushes every valid
    observation (useful on pre-resampled grids). ``max_clock_fill`` bounds LOCF catch-up
    pushes after a long silence; longer outages should be handled by the caller's reset.
    """

    window: int = 500
    min_periods: int | None = None
    sigma_floor: float = 0.005
    include_current: bool = False
    sample_mode: Literal["event", "clock"] = "event"
    clock_interval_s: float = 1.0
    change_eps: float = 1e-12
    recompute_every: int | None = None
    max_clock_fill: int = 60

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must be >= 2")
        if self.min_periods is not None and not 1 <= self.min_periods <= self.window:
            raise ValueError("min_periods must be in [1, window]")
        if not self.sigma_floor >= 0.0:
            raise ValueError("sigma_floor must be >= 0")
        if self.sample_mode not in ("event", "clock"):
            raise ValueError(f"unknown sample_mode {self.sample_mode!r}")
        if not self.clock_interval_s > 0.0:
            raise ValueError("clock_interval_s must be > 0")
        if self.max_clock_fill < 1:
            raise ValueError("max_clock_fill must be >= 1")
        if self.sample_mode == "clock" and self.include_current:
            raise ValueError("include_current is only defined for sample_mode='event'")

    @property
    def min_obs(self) -> int:
        return self.window if self.min_periods is None else self.min_periods


@dataclass(slots=True)
class ZState:
    """Engine output for one observation.

    ``mu``/``sigma`` are the reference statistics ``z`` was computed against (NaN during
    warm-up and for invalid S); ``pushed`` says whether this update added samples to the
    window; ``n_obs`` is the window count after the update.
    """

    t_ns: int
    s: float
    mu: float
    sigma: float
    z: float
    pushed: bool
    n_obs: int


class ZScoreEngine:
    """Streaming ``z_t = (S_t - μ_{t-1}) / max(σ_{t-1}, σ_floor)``.

    Event mode: z is computed *before* S_t enters the window, so it is a one-step-ahead
    standardised innovation (including S_t would let a jump inflate σ and damp its own z).
    S is pushed only when valid and ``|S - previous valid S| > change_eps`` (the first valid
    S is always pushed); otherwise σ would shrink with the message rate rather than with
    price movement. An unchanged valid S returns the cached state; an invalid S returns NaN
    and pushes nothing. ``include_current=True`` pushes first.

    Clock mode: at every grid boundary crossed (multiples of ``clock_interval_s``) the last
    valid S is pushed (LOCF, at most ``max_clock_fill`` times); boundaries crossed while S is
    invalid push nothing. z uses only grid samples strictly before t.
    """

    __slots__ = ("cfg", "stats", "n_resets", "n_fill_capped", "_min_obs", "_floor", "_eps",
                 "_include", "_clock", "_dt_ns", "_prev", "_z", "_mu", "_sigma", "_next_grid", "_held")

    def __init__(self, cfg: ZConfig | None = None) -> None:
        self.cfg = cfg = cfg or ZConfig()
        self.stats = RollingStats(cfg.window, cfg.recompute_every)
        self._min_obs = cfg.min_obs
        self._floor = float(cfg.sigma_floor)
        self._eps = float(cfg.change_eps)
        self._include = bool(cfg.include_current)
        self._clock = cfg.sample_mode == "clock"
        self._dt_ns = max(int(round(cfg.clock_interval_s * 1e9)), 1)
        self.n_resets = 0
        self.n_fill_capped = 0
        self._clear()

    def _clear(self) -> None:
        self.stats.reset()
        self._prev: float | None = None
        self._z = self._mu = self._sigma = _NAN
        self._next_grid: int | None = None
        self._held: float | None = None

    def reset(self, reason: str = "") -> None:
        """Forget the window (recording gap, structural break such as a removed leg)."""
        self._clear()
        self.n_resets += 1
        log.info("z engine reset (%s)", reason or "unspecified")

    @property
    def warm(self) -> bool:
        return self.stats.count >= self._min_obs

    def _score(self, s: float) -> tuple[float, float, float]:
        rs = self.stats
        if rs.count < self._min_obs:
            return _NAN, _NAN, _NAN
        mu = rs.mean()
        sigma = rs.std()
        if sigma != sigma:
            return mu, sigma, _NAN
        den = sigma if sigma > self._floor else self._floor
        if den <= 0.0:
            return mu, sigma, _NAN
        return mu, sigma, (s - mu) / den

    def update(self, t_ns: int, s: float, valid: bool) -> ZState:
        s = float(s)
        if self._clock:
            return self._update_clock(int(t_ns), s, valid and s == s)
        rs = self.stats
        if not valid or s != s:
            return ZState(t_ns, s, _NAN, _NAN, _NAN, False, rs.count)
        prev = self._prev
        self._prev = s
        if prev is not None and abs(s - prev) <= self._eps:
            return ZState(t_ns, s, self._mu, self._sigma, self._z, False, rs.count)
        if self._include:
            rs.push(s)
            mu, sigma, z = self._score(s)
        else:
            mu, sigma, z = self._score(s)
            rs.push(s)
        self._mu, self._sigma, self._z = mu, sigma, z
        return ZState(t_ns, s, mu, sigma, z, True, rs.count)

    def _update_clock(self, t_ns: int, s: float, ok: bool) -> ZState:
        rs = self.stats
        dt = self._dt_ns
        pushed = False
        if self._next_grid is None:
            self._next_grid = -(-t_ns // dt) * dt  # first boundary at or after t
        elif t_ns > self._next_grid:
            k = (t_ns - self._next_grid + dt - 1) // dt  # boundaries g with g < t
            self._next_grid += k * dt
            held = self._held
            if held is not None:
                m = min(k, self.cfg.max_clock_fill)
                if m < k:
                    self.n_fill_capped += 1
                    log.debug("clock fill capped: %d boundaries crossed, %d pushed", k, m)
                for _ in range(min(m, rs.window)):  # > window identical pushes change nothing
                    rs.push(held)
                pushed = True
        self._held = s if ok else None
        if not ok:
            return ZState(t_ns, s, _NAN, _NAN, _NAN, pushed, rs.count)
        mu, sigma, z = self._score(s)
        return ZState(t_ns, s, mu, sigma, z, pushed, rs.count)


# --------------------------------------------------------------------------- signals
class Action(Enum):
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"


_SIDE_AFTER = {Action.ENTER_LONG: Side.LONG_BASKET, Action.ENTER_SHORT: Side.SHORT_BASKET, Action.EXIT: Side.FLAT}


@dataclass(frozen=True, slots=True)
class SignalConfig:
    """Entry/exit thresholds.

    ``z_exit == 0`` means a zero-crossing exit (z touches or crosses 0 relative to the entry
    sign) - ``|z| < 0`` could never fire. ``exit_ref='frozen_entry_mu'`` tests the revert
    exit on ``(S - μ_entry) / max(σ_t, floor)`` instead of the rolling z, so a rolling mean
    that drifts toward S without S reverting does not count as convergence.
    """

    z_entry: float = 2.0
    z_exit: float = 0.2
    z_stop: float | None = None
    max_hold_s: float | None = None
    cooldown_s: float = 0.0
    allow_long: bool = True
    allow_short: bool = True
    exit_ref: Literal["rolling", "frozen_entry_mu"] = "rolling"

    def __post_init__(self) -> None:
        if not 0.0 <= self.z_exit < self.z_entry:
            raise ValueError("need 0 <= z_exit < z_entry")
        if self.z_stop is not None and not self.z_stop > self.z_entry:
            raise ValueError("z_stop must be None or > z_entry")
        if self.max_hold_s is not None and not self.max_hold_s > 0.0:
            raise ValueError("max_hold_s must be None or > 0")
        if not self.cooldown_s >= 0.0:
            raise ValueError("cooldown_s must be >= 0")
        if self.exit_ref not in ("rolling", "frozen_entry_mu"):
            raise ValueError(f"unknown exit_ref {self.exit_ref!r}")


@dataclass(slots=True)
class Signal:
    """A state transition. ``mu_entry`` is μ when the position was opened, so for exits
    ``baseline_drift = mu - mu_entry`` measures how much of a 'revert' was μ moving."""

    t_ns: int
    action: Action
    reason: str
    z: float
    side_before: Side
    s: float = _NAN
    mu: float = _NAN
    sigma: float = _NAN
    mu_entry: float = _NAN

    @property
    def side_after(self) -> Side:
        return _SIDE_AFTER[self.action]

    @property
    def baseline_drift(self) -> float:
        return self.mu - self.mu_entry


class SignalStateMachine:
    """FLAT / LONG_BASKET / SHORT_BASKET driven by z (rules evaluated in this order).

    * NaN z: no action, but a timeout can still fire.
    * FLAT: nothing inside the cooldown; ``z > z_entry`` -> ENTER_SHORT (S rich: buy the NO
      set); ``z < -z_entry`` -> ENTER_LONG (S cheap: buy the YES set); each needs its
      ``allow_*`` flag and ``gate(side)`` (e.g. a positive executable edge).
    * In a position, with ``z'`` = z signed so that the entry side is positive:
      ``stop`` if ``z' > z_stop``; ``flip`` if ``z' < -z_entry`` (re-entry on a later tick);
      ``revert`` if the exit reference has come back to within ``z_exit`` of zero or beyond
      (``<= 0`` when ``z_exit == 0``); ``timeout`` once held longer than ``max_hold_s``.
    """

    __slots__ = ("cfg", "sigma_floor", "state", "entry_t_ns", "entry_z", "mu_entry",
                 "last_exit_t_ns", "_cool_ns", "_hold_ns", "_frozen")

    def __init__(self, cfg: SignalConfig | None = None, sigma_floor: float = 0.005) -> None:
        self.cfg = cfg = cfg or SignalConfig()
        self.sigma_floor = float(sigma_floor)
        self._cool_ns = int(round(cfg.cooldown_s * 1e9))
        self._hold_ns = None if cfg.max_hold_s is None else int(round(cfg.max_hold_s * 1e9))
        self._frozen = cfg.exit_ref == "frozen_entry_mu"
        self.state = Side.FLAT
        self.entry_t_ns: int | None = None
        self.entry_z = _NAN
        self.mu_entry = _NAN
        self.last_exit_t_ns: int | None = None

    def step(self, t_ns: int, zs: ZState, gate: Callable[[Side], bool] | None = None) -> Signal | None:
        return self._step(t_ns, zs.s, zs.mu, zs.sigma, zs.z, gate)

    def _step(self, t_ns: int, s: float, mu: float, sigma: float, z: float,
              gate: Callable[[Side], bool] | None) -> Signal | None:
        cfg = self.cfg
        state = self.state
        if state is Side.FLAT:
            if z != z:
                return None
            if self.last_exit_t_ns is not None and t_ns - self.last_exit_t_ns < self._cool_ns:
                return None
            if z > cfg.z_entry and cfg.allow_short and (gate is None or gate(Side.SHORT_BASKET)):
                return self._enter(t_ns, Side.SHORT_BASKET, s, mu, sigma, z)
            if z < -cfg.z_entry and cfg.allow_long and (gate is None or gate(Side.LONG_BASKET)):
                return self._enter(t_ns, Side.LONG_BASKET, s, mu, sigma, z)
            return None

        reason = None
        if z == z:
            sgn = 1.0 if state is Side.SHORT_BASKET else -1.0
            zsg = sgn * z  # > 0 while S is still on the entry side of μ
            if cfg.z_stop is not None and zsg > cfg.z_stop:
                reason = "stop"
            elif zsg < -cfg.z_entry:
                reason = "flip"
            else:
                if self._frozen:
                    floor = self.sigma_floor
                    den = sigma if sigma > floor else floor
                    zr = sgn * (s - self.mu_entry) / den if den > 0.0 else _NAN
                else:
                    zr = zsg
                if (zr <= 0.0) if cfg.z_exit == 0.0 else (zr < cfg.z_exit):
                    reason = "revert"
        if reason is None and self._hold_ns is not None and t_ns - self.entry_t_ns > self._hold_ns:
            reason = "timeout"
        if reason is None:
            return None
        return self._exit(t_ns, reason, s, mu, sigma, z)

    def _enter(self, t_ns: int, side: Side, s: float, mu: float, sigma: float, z: float) -> Signal:
        self.state = side
        self.entry_t_ns = t_ns
        self.entry_z = z
        self.mu_entry = mu
        action = Action.ENTER_SHORT if side is Side.SHORT_BASKET else Action.ENTER_LONG
        return Signal(t_ns, action, "z_entry", z, Side.FLAT, s, mu, sigma, mu)

    def _exit(self, t_ns: int, reason: str, s: float, mu: float, sigma: float, z: float) -> Signal:
        sig = Signal(t_ns, Action.EXIT, reason, z, self.state, s, mu, sigma, self.mu_entry)
        self._to_flat()
        self.last_exit_t_ns = t_ns
        return sig

    def _to_flat(self) -> None:
        self.state = Side.FLAT
        self.entry_t_ns = None
        self.entry_z = _NAN
        self.mu_entry = _NAN

    def force_flat(self, t_ns: int, reason: str, zs: ZState | None = None) -> Signal | None:
        """Exit regardless of z (end of data, resolution, basket changed). None if flat."""
        if self.state is Side.FLAT:
            return None
        if zs is None:
            return self._exit(t_ns, reason, _NAN, _NAN, _NAN, _NAN)
        return self._exit(t_ns, reason, zs.s, zs.mu, zs.sigma, zs.z)

    def on_entry_failed(self) -> None:
        """Entry filled nothing: back to FLAT without an exit record or a cooldown."""
        self._to_flat()


# --------------------------------------------------------------------------- batch / offline
def _segments(n: int, reset: np.ndarray | None) -> list[tuple[int, int]]:
    """[lo, hi) slices between resets (``reset[i]`` = reset before observation i)."""
    if reset is None:
        return [(0, n)] if n else []
    r = np.asarray(reset, dtype=bool)
    if r.shape != (n,):
        raise ValueError("reset must have the same length as s")
    starts = sorted({0, *np.flatnonzero(r).tolist()}) if n else []
    return list(zip(starts, starts[1:] + [n], strict=True))


_DIRECT_MAX_WINDOW = 32  # below this, exact per-window two-pass is cheap (O(T·w))
_RECENTRE_RATIO = 1e4  # bounds the blocked path's relative error of M2 to ~1e-11


def rolling_mean_std(x: np.ndarray, window: int, min_periods: int | None = None,
                     ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over the inclusive windows ``x[t-window+1 .. t]``.

    Windows at the start are partial (count ``t + 1``); outputs with fewer than
    ``min_periods`` (default ``window``) values are NaN, and std is NaN while
    ``count <= ddof`` - the same conventions as :class:`RollingStats` and pandas.

    Small windows use an exact two-pass per window built from ``window`` shifted slices.
    Larger windows are O(T): blocks of outputs use cumulative sums of values centred on a
    per-block reference ``c`` (the mean of the block's first window), so the shifted-data
    variance ``(Σy² - (Σy)²/k) / (k - ddof)`` with ``y = x - c`` does not cancel
    catastrophically; a block is restarted (re-centred) as soon as the cumulative sums
    dwarf a window's own M2, e.g. after a level shift. Every output and every restart
    decision uses data at or before its index only, so results are causal bit-for-bit.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be 1-D")
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    mp = window if min_periods is None else int(min_periods)
    if x.size and not np.isfinite(x).all():
        raise ValueError("x must be finite")
    if window <= _DIRECT_MAX_WINDOW:
        mean, m2, cnt = _rolling_direct(x, window)
    else:
        mean, m2, cnt = _rolling_blocked(x, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = m2 / (cnt - ddof)
    var[cnt <= ddof] = _NAN
    ok = cnt >= mp
    return np.where(ok, mean, _NAN), np.where(ok, np.sqrt(var), _NAN)


def _rolling_direct(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    pad = np.concatenate((np.zeros(w - 1), x))
    mask = np.concatenate((np.zeros(w - 1), np.ones(n)))
    cnt = np.minimum(np.arange(1, n + 1), w).astype(np.float64)
    acc = np.zeros(n)
    for k in range(w):
        acc += pad[k:k + n]
    mean = acc / cnt
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    for k in range(w):
        d = (pad[k:k + n] - mean) * mask[k:k + n]
        s1 += d
        s2 += d * d
    return mean + s1 / cnt, np.maximum(s2 - s1 * s1 / cnt, 0.0), cnt


def _rolling_blocked(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    mean = np.empty(n)
    m2 = np.empty(n)
    cnt = np.minimum(np.arange(1, n + 1), w).astype(np.float64)
    block = min(max(16 * w, 256), 1 << 16)
    start = 0
    while start < n:
        stop = min(start + block, n)
        lo = max(0, start - w + 1)
        c = float(x[lo:start + 1].mean())
        y = x[lo:stop] - c
        c1 = np.zeros(y.size + 1)
        c2 = np.zeros(y.size + 1)
        np.cumsum(y, out=c1[1:])
        np.cumsum(y * y, out=c2[1:])
        j = np.arange(start, stop)
        hi = j + 1 - lo
        lw = np.maximum(j - w + 1, 0) - lo
        s1 = c1[hi] - c1[lw]
        mu = s1 / cnt[start:stop]
        q = (c2[hi] - c2[lw]) - s1 * mu
        # Rounding error of q is ~eps * c2[hi]; after a level shift that dwarfs the window's
        # own M2, so restart the block (re-centre) at the first such output.
        bad = np.flatnonzero(c2[hi] > _RECENTRE_RATIO * q)
        r = stop - start if bad.size == 0 else max(int(bad[0]), 1)
        mean[start:start + r] = c + mu[:r]
        m2[start:start + r] = np.maximum(q[:r], 0.0)
        start += r
    return mean, m2, cnt


def push_mask(s: np.ndarray, valid: np.ndarray, eps: float = 1e-12,
              reset: np.ndarray | None = None) -> np.ndarray:
    """Which observations :class:`ZScoreEngine` (event mode) pushes: valid, finite and
    ``|s - previous valid s| > eps``; the first valid one after each reset is always pushed."""
    s = np.asarray(s, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(s)
    out = np.zeros(s.size, dtype=bool)
    for lo, hi in _segments(s.size, reset):
        vi = lo + np.flatnonzero(ok[lo:hi])
        if vi.size:
            out[vi[0]] = True
            out[vi[1:]] = np.abs(np.diff(s[vi])) > eps
    return out


def zscore_batch(s: np.ndarray, valid: np.ndarray, cfg: ZConfig,
                 reset: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Vectorised event-mode :class:`ZScoreEngine`: returns ``z``, ``mu``, ``sigma``,
    ``pushed`` arrays equal to the streaming states (to ~1e-12).

    Pushed values get reference stats from :func:`rolling_mean_std` (shifted by one push
    when excluding the current value); unchanged valid observations carry the last pushed
    state forward; invalid ones are NaN. ``reset[i]`` mimics ``engine.reset()`` before i.
    """
    if cfg.sample_mode != "event":
        raise ValueError("zscore_batch implements event mode; resample to a grid and use change_eps < 0")
    s = np.asarray(s, dtype=np.float64)
    ok = np.asarray(valid, dtype=bool) & np.isfinite(s)
    n = s.size
    z = np.full(n, _NAN)
    mu = np.full(n, _NAN)
    sig = np.full(n, _NAN)
    pushed = push_mask(s, ok, cfg.change_eps, reset)
    floor = float(cfg.sigma_floor)
    for lo, hi in _segments(n, reset):
        pidx = lo + np.flatnonzero(pushed[lo:hi])
        if pidx.size == 0:
            continue
        p = s[pidx]
        m, sd = rolling_mean_std(p, cfg.window, min_periods=cfg.min_obs)
        if not cfg.include_current:
            m = np.concatenate(([_NAN], m[:-1]))
            sd = np.concatenate(([_NAN], sd[:-1]))
        den = np.where(sd > floor, sd, floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            zp = np.where(np.isfinite(sd) & (den > 0.0), (p - m) / den, _NAN)
        # forward-fill each pushed state over the following valid observations
        last = np.maximum.accumulate(np.where(pushed[lo:hi], np.arange(hi - lo), -1))
        vi = np.flatnonzero(ok[lo:hi])
        rank = np.cumsum(pushed[lo:hi]) - 1  # position of a pushed index within pidx
        j = rank[last[vi]]
        z[lo + vi] = zp[j]
        mu[lo + vi] = m[j]
        sig[lo + vi] = sd[j]
    return {"z": z, "mu": mu, "sigma": sig, "pushed": pushed}


def positions_from_z(t_ns: np.ndarray, s: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                     z: np.ndarray, cfg: SignalConfig, sigma_floor: float = 0.005,
                     ) -> tuple[np.ndarray, list[Signal]]:
    """Run the same :class:`SignalStateMachine` (no gate) over arrays.

    Returns the side held *after* each observation (int8) and the emitted signals. The
    machine is path dependent, so this is an O(T) scalar loop - offline use only.
    """
    sm = SignalStateMachine(cfg, sigma_floor)
    step = sm._step
    cols = [np.asarray(a).tolist() for a in (t_ns, s, mu, sigma, z)]
    if len({len(c) for c in cols}) != 1:
        raise ValueError("all input arrays must have the same length")
    sides: list[int] = []
    signals: list[Signal] = []
    for t, si, mi, sd, zi in zip(*cols, strict=True):
        sig = step(int(t), si, mi, sd, zi, None)
        if sig is not None:
            signals.append(sig)
        sides.append(sm.state)
    return np.asarray(sides, dtype=np.int8), signals


# --------------------------------------------------------------------------- economics
def _fee_params(fees: Sequence[FeeSchedule] | None, n: int) -> tuple[np.ndarray, np.ndarray]:
    if fees is None:
        return np.zeros(n), np.ones(n)
    if len(fees) != n:
        raise ValueError(f"expected {n} fee schedules, got {len(fees)}")
    return (np.array([f.rate for f in fees], dtype=np.float64),
            np.array([f.exponent for f in fees], dtype=np.float64))


def _unit_fee(p: np.ndarray, rates: np.ndarray, exps: np.ndarray) -> np.ndarray:
    """Unrounded taker fee per share, ``r (p(1-p))**e`` (as ``FeeSchedule.fee_per_unit``)."""
    q = np.clip(p, 0.0, 1.0)
    return rates * (q * (1.0 - q)) ** exps


def _strict_tops(bid: np.ndarray, ask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(bid, dtype=np.float64)
    a = np.asarray(ask, dtype=np.float64)
    if b.shape != a.shape or b.ndim not in (1, 2):
        raise ValueError("bid and ask must have equal shape (n,) or (T, n)")
    return np.where(np.isfinite(b), b, 0.0), np.where(np.isfinite(a), a, 1.0)


def _out(x: np.ndarray) -> float | np.ndarray:
    return float(x) if np.ndim(x) == 0 else x


def executable_edges(bid: np.ndarray, ask: np.ndarray, fees: Sequence[FeeSchedule] | None = None,
                     ) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Per-unit taker edges at the top of book, held to resolution (or converted).

    * long  = ``1 - Σ a_i - Σ fee(a_i)``      (buy the YES set; it pays exactly 1)
    * short = ``Σ b_i - 1 - Σ fee(1 - b_i)``  (buy the NO set at ``1 - b_i``; pays ``n - 1``)

    Missing bids count 0 and missing asks 1 (no edge from an empty side). Fees are the USD
    equivalent; collecting them in shares is identical to first order in r. Inputs of shape
    ``(T, n)`` give arrays of length T.
    """
    b, a = _strict_tops(bid, ask)
    rates, exps = _fee_params(fees, b.shape[-1])
    long_edge = 1.0 - a.sum(axis=-1) - _unit_fee(a, rates, exps).sum(axis=-1)
    short_edge = b.sum(axis=-1) - 1.0 - _unit_fee(1.0 - b, rates, exps).sum(axis=-1)
    return _out(long_edge), _out(short_edge)


def basket_cost_hurdle(bid: np.ndarray, ask: np.ndarray, fees: Sequence[FeeSchedule] | None = None,
                       ) -> dict[str, float | np.ndarray]:
    """Move in ``S_mid`` a taker z round trip must exceed to break even.

    Long: buy at ``Σa = S_mid + ½Σs``, sell at ``Σb = S_mid' - ½Σs'``, so with similar
    spreads the move must clear ``spread_sum + F_entry + F_exit``. Short is the mirror image
    through NO tokens (buy at ``1 - b``, sell at ``1 - a``). The fee curve is symmetric, so
    both hurdles coincide; both are reported for clarity.
    """
    b, a = _strict_tops(bid, ask)
    rates, exps = _fee_params(fees, b.shape[-1])
    spread = (a - b).sum(axis=-1)
    f_long_in = _unit_fee(a, rates, exps).sum(axis=-1)          # buy YES at a
    f_long_out = _unit_fee(b, rates, exps).sum(axis=-1)         # sell YES at b
    f_short_in = _unit_fee(1.0 - b, rates, exps).sum(axis=-1)   # buy NO at 1 - b
    f_short_out = _unit_fee(1.0 - a, rates, exps).sum(axis=-1)  # sell NO at 1 - a
    return {
        "spread_sum": _out(spread),
        "fee_entry_long": _out(f_long_in),
        "fee_entry_short": _out(f_short_in),
        "fee_exit_long": _out(f_long_out),
        "fee_exit_short": _out(f_short_out),
        "hurdle_long": _out(spread + f_long_in + f_long_out),
        "hurdle_short": _out(spread + f_short_in + f_short_out),
    }


def edge_to_cost_ratio(s_mid: np.ndarray, hurdle: float) -> float:
    """ECR = std(S_mid) / hurdle - how many hurdles a typical S excursion spans (pro-tip 2:
    rank baskets by it). Non-finite S values are ignored; NaN with fewer than 2 values."""
    if not hurdle > 0.0:
        raise ValueError("hurdle must be > 0")
    x = np.asarray(s_mid, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return _NAN
    return float(x.std(ddof=1)) / float(hurdle)
