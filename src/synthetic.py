"""Seeded synthetic negRisk basket: latent probabilities, quotes, L2 books and wire JSON.

**SYNTHETIC DATA.** This is an offline fallback for tests and demos and is never evidence
about real markets. Everything it produces is labelled: basket ids start with
``SYNTHETIC_``, token ids with ``SYNTHETIC-``, DataFrames carry ``attrs["synthetic"] = True``,
and recordings are written only below ``data/synthetic/`` next to a ``meta.json`` that says
``"synthetic": true``. Any statistical "finding" on this data is implied by the generator.

Model (research_quant.md section 6)
-----------------------------------
1. **Latent probabilities.** Leg logits follow a driftless random walk plus compound-Poisson
   news jumps on one leg at a time, and ``pi = softmax(logits)``. The outcome set is
   therefore exhaustive and ``sum(pi) == 1`` to rounding (softmax of a driftless walk is only
   approximately a martingale). The winner is drawn from ``pi`` at the horizon, so
   hold-to-resolution payoffs are consistent with prices.
2. **Quoted fair value** ``q = FLB(pi) + d + u + regime / n`` where
   ``FLB(pi) = pi^g / (pi^g + (1 - pi)^g)`` (``g < 1``: long shots over-, favourites
   under-priced, so ``sum(q)`` carries a drifting overround); ``d`` is a per-leg AR(1)
   dislocation with half-life ``disloc_half_life_s`` plus sweep shocks that hit one leg and
   decay with the same coefficient (the mean-reverting part of the basket sum); ``u`` is
   i.i.d. quote noise; ``regime`` is a rare *permanent* overround shift (a z-signal that
   never reverts). Shock times, signs and sizes come from their own RNG stream, independent
   of everything observable before the shock.
3. **Books.** Tick-quantised ladders around ``q`` (tick 0.001 near 0/1), wider spreads on
   long shots, LogNormal depth decaying with the level, an occasionally thin top level, and
   dust legs whose bid side is empty.
4. **Wire.** Asynchronous per-leg Poisson arrivals (rate rising with ``pi``, so long-shot
   quotes go stale) emit Polymarket-shaped JSON: an initial ``book`` per leg, v2
   ``price_change`` messages with only the changed levels, ``book`` re-snapshots every 30 min
   and after injected recording gaps, ``tick_size_change`` and finally ``market_resolved``.

Economics the calibration targets
---------------------------------
With ``S_bid``/``S_mid``/``S_ask`` the sums of YES best bids/mids/asks, the no-arbitrage band
is ``S_bid <= 1 <= S_ask``. A long basket buys one YES of every leg (cost ``S_ask`` + fees,
pays exactly 1); a short basket BUYS one NO of every leg (NO ask = 1 - YES bid because the
books are mirrored; cost ``n - S_bid`` + fees, pays exactly ``n - 1``). Taker fees are
``shares * r * p * (1 - p)`` per order. Mid-sum reversion *inside* the band cannot be
monetised by a taker, so defaults put the mean sweep size at roughly 0.5-1.5x the
round-trip cost ``sum(spread) + 2 * r * sum(p(1-p))`` (see ``calibration_summary``):
results are marginal, parameter sweeps can lose money, and regime breaks, latency and
legging cost money.

Offline module: NumPy/SciPy at import time, pandas only inside the DataFrame builders.
"""
from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.signal import lfilter

from . import CONFIG_PATH, SYNTH_ROOT, code_version
from .config import Basket, FeeSchedule, Leg, basket_to_dict, load_markets
from .events import ASK, BID, PRICE_SCALE, TOB_COLUMNS

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)

SYNTHETIC_PREFIX = "SYNTHETIC_"
TOKEN_PREFIX = "SYNTHETIC-"
BOP_LABELS = ("D Senate, D House", "R Senate, D House", "D Senate, R House", "R Senate, R House")
_DEMO_PI0 = (0.60, 0.30, 0.06, 0.03, 0.01)
_RECV_LAG_NS = 40_000_000  # server timestamp -> local receive time
_BLOCK = 8192  # arrivals per block of pre-drawn ladder randomness
# Independent RNG streams (SeedSequence spawn keys): adding a stream never shifts the others.
_STREAMS = ("logit", "jumps", "disloc", "shocks", "noise", "regime", "winner", "gaps",
            "arrivals", "book", "pairing")


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class SyntheticParams:
    """All knobs of the generator. ``name`` is the basket id and always starts with
    ``SYNTHETIC_`` (it is prefixed if missing). ``pi0=None`` picks a default: the demo
    distribution for 5 legs, otherwise a Zipf-like ``(i + 1) ** -1.5`` profile."""

    name: str = "SYNTHETIC_demo5"
    n_legs: int = 5
    pi0: tuple[float, ...] | None = None
    leg_labels: tuple[str, ...] | None = None
    seed: int = 7
    start: str = "2026-01-01T00:00:00Z"
    duration_s: float = 3 * 86400.0
    dt_s: float = 1.0  # latent grid; quotes are piecewise constant on it
    # latent probabilities
    sigma_logit_per_sqrt_day: float = 0.25
    jump_rate_per_day: float = 2.0
    jump_sigma: float = 0.5
    # quoted fair value
    flb_gamma: float = 0.97
    disloc_half_life_s: float = 120.0
    disloc_sigma: float = 0.002  # stationary sd of each leg's AR(1) dislocation
    shock_rate_per_hour: float = 1.0
    shock_size: float = 0.08  # mean |sweep| (exponential magnitude, random sign)
    noise_sigma: float = 0.0005
    regime_break_prob_per_day: float = 0.1
    regime_break_size: float = 0.01
    # books
    tick: float = 0.01
    fine_tick: float = 0.001
    fine_tick_edge: float = 0.04  # fine tick when q < edge or q > 1 - edge
    tick_hysteresis: float = 0.005  # back to the coarse tick only inside [edge + h, 1 - edge - h]
    base_spread_ticks: int = 1
    spread_lambda0: float = 0.3
    longshot_spread_lambda: float = 2.0
    depth_levels: int = 8
    depth_mu: float = 5.0
    depth_nu: float = 0.5
    depth_alpha: float = 0.3
    depth_sigma: float = 0.7
    thin_top_prob: float = 0.1
    level_refresh_prob: float = 0.3  # chance an unchanged price level gets a new size
    # wire / recording
    base_rate_hz: float = 0.5  # arrival rate of a leg at pi = 1 (rate = base * sqrt(pi))
    snapshot_interval_s: float = 1800.0
    gap_prob_per_hour: float = 0.02
    gap_len_s: tuple[float, float] = (30.0, 300.0)
    resolve_at_end: bool = True
    fee_rate: float = 0.04  # synthetic taker fee rate, so costs exist

    def __post_init__(self) -> None:
        put = functools.partial(object.__setattr__, self)
        if not self.name.startswith(SYNTHETIC_PREFIX):
            put("name", SYNTHETIC_PREFIX + self.name)
        if int(self.n_legs) != self.n_legs or self.n_legs < 2:
            raise ValueError(f"n_legs must be an integer >= 2, got {self.n_legs!r}")
        put("n_legs", int(self.n_legs))
        put("seed", int(self.seed))
        if self.pi0 is None:
            w = np.asarray(_DEMO_PI0) if self.n_legs == len(_DEMO_PI0) else (np.arange(self.n_legs) + 1.0) ** -1.5
            put("pi0", tuple(float(x) for x in w / w.sum()))
        else:
            put("pi0", tuple(float(x) for x in self.pi0))
        if len(self.pi0) != self.n_legs or not all(x > 0 and math.isfinite(x) for x in self.pi0):
            raise ValueError(f"pi0 needs {self.n_legs} positive entries, got {self.pi0!r}")
        if self.leg_labels is not None:
            put("leg_labels", tuple(str(x) for x in self.leg_labels))
            if len(self.leg_labels) != self.n_legs:
                raise ValueError(f"leg_labels needs {self.n_legs} entries, got {len(self.leg_labels)}")
        put("gap_len_s", tuple(float(x) for x in self.gap_len_s))
        lo, hi = self.gap_len_s if len(self.gap_len_s) == 2 else (0.0, -1.0)
        checks = {
            "duration_s > 0": self.duration_s > 0,
            "0 < dt_s <= duration_s": 0 < self.dt_s <= self.duration_s,
            "disloc_half_life_s > 0": self.disloc_half_life_s > 0,
            "flb_gamma > 0": self.flb_gamma > 0,
            "0 < fine_tick < tick < 0.5": 0 < self.fine_tick < self.tick < 0.5,
            "0 <= fine_tick_edge < 0.5": 0 <= self.fine_tick_edge < 0.5,
            "tick_hysteresis >= 0": self.tick_hysteresis >= 0,
            "base_spread_ticks >= 1": self.base_spread_ticks >= 1,
            "depth_levels >= 1": self.depth_levels >= 1,
            "base_rate_hz > 0": self.base_rate_hz > 0,
            "snapshot_interval_s > 0": self.snapshot_interval_s > 0,
            "0 < gap_len_s[0] <= gap_len_s[1]": 0 < lo <= hi,
            "fee_rate >= 0": self.fee_rate >= 0,
            "probabilities in [0, 1]": all(0 <= x <= 1 for x in (self.thin_top_prob, self.level_refresh_prob)),
            "rates >= 0": min(self.jump_rate_per_day, self.shock_rate_per_hour, self.gap_prob_per_hour,
                              self.regime_break_prob_per_day, self.disloc_sigma, self.noise_sigma,
                              self.shock_size, self.jump_sigma, self.sigma_logit_per_sqrt_day) >= 0,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"invalid SyntheticParams: {bad}")
        for t in (self.tick, self.fine_tick):
            if abs(t * PRICE_SCALE - round(t * PRICE_SCALE)) > 1e-6 or PRICE_SCALE % round(t * PRICE_SCALE):
                raise ValueError(f"tick {t} must divide 1 in micro-units")
        _start_ns(self.start)  # validates the timestamp

    @classmethod
    def from_dict(cls, params: Mapping[str, Any], *, name: str, seed: int | None = None) -> SyntheticParams:
        """Build from a ``markets.json`` ``synthetic_baskets[].params`` mapping.

        Unknown keys raise ``ValueError`` (a typo must not silently fall back to a default);
        JSON lists become tuples. ``name``/``seed`` override any value inside ``params``.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown synthetic parameter(s) {unknown}; known: {sorted(known)}")
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
        kw["name"] = name
        if seed is not None:
            kw["seed"] = seed
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (tuples -> lists); ``from_dict(to_dict(p), name=p.name) == p``."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out


@dataclass
class SyntheticTruth:
    """Latent state on the grid ``t_s`` (seconds since ``start``). Never visible to a strategy."""

    t_s: np.ndarray  # (T,)
    pi: np.ndarray  # (T, n) latent probabilities, rows sum to 1
    flb: np.ndarray  # (T, n) FLB(pi)
    disloc: np.ndarray  # (T, n) AR(1) dislocation incl. sweep shocks
    regime: np.ndarray  # (T,) cumulative overround shift (spread as regime / n over the legs)
    q_fair: np.ndarray  # (T, n) flb + disloc + noise + regime / n (unclipped, unquantised)
    winner: int
    shock_t: np.ndarray  # shock times (s)
    shock_leg: np.ndarray
    shock_size: np.ndarray  # signed
    break_t: np.ndarray  # regime-break times (s)
    break_size: np.ndarray  # signed
    gaps: list[tuple[float, float]]  # injected recording gaps [start_s, end_s)

    @property
    def basket_dislocation(self) -> np.ndarray:
        """``sum_i d_i``: the transient (mean-reverting) part of the basket sum."""
        return self.disloc.sum(axis=1)


# --------------------------------------------------------------------------- helpers
def _rng(seed: int, stream: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(_STREAMS.index(stream),)))


def _start_ns(start: str) -> int:
    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000_000_000 + dt.microsecond * 1000


def _px_int(x: float) -> int:
    return int(round(x * PRICE_SCALE))


@functools.lru_cache(maxsize=4096)
def _fmt_px(p: int) -> str:
    """Micro-units -> wire decimal string without trailing zeros (``480000 -> "0.48"``)."""
    whole, frac = divmod(p, PRICE_SCALE)
    return str(whole) if frac == 0 else f"{whole}.{frac:06d}".rstrip("0")


def _fmt_size(s: float) -> str:
    return f"{s:.2f}".rstrip("0").rstrip(".")


def flb(pi: np.ndarray | float, gamma: float) -> np.ndarray:
    """Favourite-long-shot distortion ``pi^g / (pi^g + (1 - pi)^g)``; ``g < 1`` overprices long shots."""
    x = np.clip(np.asarray(pi, dtype=float), 0.0, 1.0)
    a, b = x**gamma, (1.0 - x) ** gamma
    return a / (a + b)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def _spread_lambda(pi: np.ndarray, p: SyntheticParams) -> np.ndarray:
    """Poisson mean of the extra spread ticks: ``lambda0 + lambda1 * exp(-pi / 0.1)`` (long shots wider)."""
    return p.spread_lambda0 + p.longshot_spread_lambda * np.exp(-np.asarray(pi, dtype=float) / 0.1)


def select_tick(q: float, prev_tick: float | None, p: SyntheticParams) -> float:
    """Tick for fair value ``q``: ``fine_tick`` when ``q < 0.04`` or ``q > 0.96`` (Polymarket's
    rule), otherwise ``tick``. A leg on the fine tick returns to the coarse tick only once
    ``q`` is ``tick_hysteresis`` inside the band, so quote noise cannot flip it every update."""
    lo, hi = p.fine_tick_edge, 1.0 - p.fine_tick_edge
    if q < lo or q > hi:
        return p.fine_tick
    h = p.tick_hysteresis
    if prev_tick is not None and _px_int(prev_tick) == _px_int(p.fine_tick) and (q < lo + h or q > hi - h):
        return p.fine_tick
    return p.tick


def _quote_ticks(x: float, k: int, max_t: int) -> tuple[int | None, int | None]:
    """Best bid/ask in tick units around fair value ``x`` (in ticks) with a ``k``-tick spread.

    ``bid = floor(x - s/2)``, ``ask = ceil(x + s/2)`` with ``s = k - 1`` ticks: the outward
    rounding then yields exactly ``k`` ticks for generic ``x`` (``s = k`` would add one tick on
    average). Prices are clipped to ``[1, max_t - 1]`` ticks with ``ask >= bid + 1``; a fair
    value below one tick leaves the bid side empty (dust), above ``1 - tick`` the ask side.
    """
    half = 0.5 * (k - 1)
    bid = min(max(math.floor(x - half), 1), max_t - 1) if x >= 1.0 else None
    ask = max(min(math.ceil(x + half), max_t - 1), 1) if x <= max_t - 1 else None
    if bid is not None and ask is not None and ask <= bid:
        if bid + 1 <= max_t - 1:
            ask = bid + 1
        else:
            bid = ask - 1
    return bid, ask


def _level_sizes(pi: np.ndarray, z: np.ndarray, u_thin: np.ndarray, p: SyntheticParams) -> np.ndarray:
    """Sizes ``LogNormal(mu + nu log pi - alpha k, sigma)`` for levels ``k = 0..L-1``, shape
    ``(m, 2, L)``; with probability ``thin_top_prob`` a side's top level is cut to 2-20 %
    (exercises the book walk). Rounded to cents, at least 1 share."""
    k = np.arange(p.depth_levels)
    log_pi = np.log(np.maximum(np.asarray(pi, dtype=float), 1e-6))[:, None, None]
    sizes = np.exp(p.depth_mu + p.depth_nu * log_pi - p.depth_alpha * k + p.depth_sigma * z)
    thin = u_thin[..., 0] < p.thin_top_prob
    sizes[..., 0] = np.where(thin, sizes[..., 0] * (0.02 + 0.18 * u_thin[..., 1]), sizes[..., 0])
    return np.maximum(np.round(sizes, 2), 1.0)


def build_ladder(q: float, tick: float, spread_ticks: int, rng: np.random.Generator, p: SyntheticParams,
                 *, pi: float | None = None) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Fresh L2 ladder around fair value ``q``: ``(bids, asks)`` as ``(price, size)`` best-first.

    Up to ``depth_levels`` levels per side one ``tick`` apart (see ``_quote_ticks`` for the
    top). Sizes follow ``_level_sizes`` with ``pi`` (default: ``q``) as the popularity proxy.
    Also used by ``data_io`` to model books around real price histories.
    """
    tick_i = _px_int(tick)
    max_t = PRICE_SCALE // tick_i
    b, a = _quote_ticks(q * max_t, int(spread_ticks), max_t)
    L = p.depth_levels
    z, u = rng.standard_normal((1, 2, L)), rng.random((1, 2, 2))
    sz = _level_sizes(np.array([q if pi is None else pi]), z, u, p)[0]
    bids = [((b - j) * tick_i / PRICE_SCALE, float(sz[BID, j])) for j in range(min(L, b))] if b is not None else []
    asks = [((a + j) * tick_i / PRICE_SCALE, float(sz[ASK, j])) for j in range(min(L, max_t - a))] if a is not None else []
    return bids, asks


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _condition_id(seed: int, i: int) -> str:
    return "0x" + hashlib.sha256(f"{TOKEN_PREFIX}{seed}-{i}".encode()).hexdigest()


# --------------------------------------------------------------------------- latent model
def _dislocation(p: SyntheticParams, T: int, pi: np.ndarray, rng_d: np.random.Generator,
                 rng_s: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-leg AR(1) ``d_t = phi d_{t-1} + eta_t + J_t`` on the grid, ``phi = 0.5^(dt/h)``.

    ``eta`` is scaled so the stationary sd is ``disloc_sigma``; ``d_0`` starts stationary.
    Sweeps ``J`` arrive at ``shock_rate_per_hour`` on one leg (chosen with probability
    ``pi_i``), with random sign and exponential magnitude of mean ``shock_size``, and decay
    with the same ``phi``. Returns ``(d, shock_idx, shock_leg, shock_signed)``.
    """
    n = pi.shape[1]
    phi = 0.5 ** (p.dt_s / p.disloc_half_life_s)
    eta = rng_d.standard_normal((T, n))
    eta[1:] *= p.disloc_sigma * math.sqrt(1.0 - phi * phi)
    eta[0] *= p.disloc_sigma
    n_s = int(rng_s.poisson(p.shock_rate_per_hour * p.duration_s / 3600.0))
    idx = np.sort(rng_s.integers(1, T, n_s))  # T >= 2 because dt_s <= duration_s
    u = rng_s.random(n_s)
    leg = np.minimum((u[:, None] > np.cumsum(pi[idx], axis=1)).sum(axis=1), n - 1)
    signed = np.where(rng_s.random(n_s) < 0.5, -1.0, 1.0) * rng_s.exponential(p.shock_size, n_s)
    np.add.at(eta, (idx, leg), signed)
    d = lfilter([1.0], [1.0, -phi], eta, axis=0)
    return d, idx, leg, signed


def _gaps(p: SyntheticParams) -> list[tuple[float, float]]:
    rng = _rng(p.seed, "gaps")
    k = int(rng.poisson(p.gap_prob_per_hour * p.duration_s / 3600.0))
    starts = rng.uniform(0.0, p.duration_s, k)
    ends = starts + rng.uniform(p.gap_len_s[0], p.gap_len_s[1], k)
    out: list[list[float]] = []
    for s, e in sorted(zip(starts.tolist(), ends.tolist())):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, min(e, p.duration_s)) for s, e in out if s < p.duration_s]


def simulate_latent(p: SyntheticParams) -> SyntheticTruth:
    """Simulate the latent model of the module docstring on the grid ``0, dt_s, ..., duration_s``."""
    T = int(round(p.duration_s / p.dt_s)) + 1
    n = p.n_legs
    t_s = np.arange(T) * p.dt_s
    pi0 = np.asarray(p.pi0) / sum(p.pi0)

    logits = _rng(p.seed, "logit").standard_normal((T, n))
    logits *= p.sigma_logit_per_sqrt_day * math.sqrt(p.dt_s / 86400.0)
    logits[0] = np.log(pi0)
    rj = _rng(p.seed, "jumps")
    n_j = int(rj.poisson(p.jump_rate_per_day * p.duration_s / 86400.0))
    np.add.at(logits, (rj.integers(1, T, n_j), rj.integers(0, n, n_j)), rj.normal(0.0, p.jump_sigma, n_j))
    np.cumsum(logits, axis=0, out=logits)
    pi = _softmax(logits)
    del logits

    d, s_idx, s_leg, s_signed = _dislocation(p, T, pi, _rng(p.seed, "disloc"), _rng(p.seed, "shocks"))
    rr = _rng(p.seed, "regime")
    n_b = int(rr.poisson(p.regime_break_prob_per_day * p.duration_s / 86400.0))
    b_idx = np.sort(rr.integers(1, T, n_b))
    b_size = np.where(rr.random(n_b) < 0.5, -1.0, 1.0) * p.regime_break_size
    regime = np.zeros(T)
    np.add.at(regime, b_idx, b_size)
    np.cumsum(regime, out=regime)

    f = flb(pi, p.flb_gamma)
    q = _rng(p.seed, "noise").normal(0.0, p.noise_sigma, (T, n))
    q += f
    q += d
    q += regime[:, None] / n
    winner = int(_rng(p.seed, "winner").choice(n, p=pi[-1] / pi[-1].sum()))
    return SyntheticTruth(
        t_s=t_s, pi=pi, flb=f, disloc=d, regime=regime, q_fair=q, winner=winner,
        shock_t=t_s[s_idx], shock_leg=s_leg, shock_size=s_signed,
        break_t=t_s[b_idx], break_size=b_size, gaps=_gaps(p),
    )


# --------------------------------------------------------------------------- basket
def synthetic_basket(p: SyntheticParams) -> Basket:
    """``config.Basket`` for the synthetic event (``synthetic=True``, complete negRisk partition)."""
    labels = p.leg_labels or tuple(f"Outcome {i}" for i in range(p.n_legs))
    fee = FeeSchedule(rate=p.fee_rate, source="synthetic")
    seen: set[str] = set()
    legs = []
    for i, label in enumerate(labels):
        leg_id = _slug(label) or f"leg-{i}"
        if leg_id in seen:
            leg_id = f"{leg_id}-{i}"
        seen.add(leg_id)
        legs.append(Leg(
            leg_id=leg_id, label=label,
            yes_token_id=f"{TOKEN_PREFIX}{p.seed}-{i}-YES", no_token_id=f"{TOKEN_PREFIX}{p.seed}-{i}-NO",
            condition_id=_condition_id(p.seed, i), question=f"SYNTHETIC: will '{label}' win?",
            tick_size=0.01, fee=fee, created_at=p.start,
        ))
    end = datetime.fromtimestamp(_start_ns(p.start) / 1e9, tz=timezone.utc) + timedelta(seconds=p.duration_s)
    return Basket(
        basket_id=p.name, title=f"SYNTHETIC DATA: {p.name} ({p.n_legs} legs, seed {p.seed})", legs=tuple(legs),
        category="synthetic", status="active", neg_risk=True, neg_risk_augmented=False,
        end_date=end.strftime("%Y-%m-%dT%H:%M:%SZ"), synthetic=True,
        notes="SYNTHETIC DATA generated by src.synthetic - not real market data.",
    )


# --------------------------------------------------------------------------- book simulation
def _arrivals(p: SyntheticParams, truth: SyntheticTruth) -> tuple[np.ndarray, np.ndarray]:
    """Per-leg Poisson quote arrivals with rate ``base_rate_hz * sqrt(pi_i(t))`` (thinning)."""
    rng = _rng(p.seed, "arrivals")
    T = truth.t_s.size
    ts, legs = [], []
    for i in range(p.n_legs):
        k = int(rng.poisson(p.base_rate_hz * p.duration_s))
        t = np.sort(rng.uniform(0.0, p.duration_s, k))
        idx = np.minimum((t / p.dt_s).astype(np.int64), T - 1)
        keep = rng.random(k) < np.sqrt(truth.pi[idx, i])
        ts.append(t[keep])
        legs.append(np.full(int(keep.sum()), i, dtype=np.int64))
    t_all, leg_all = np.concatenate(ts), np.concatenate(legs)
    order = np.argsort(t_all, kind="stable")
    return t_all[order], leg_all[order]


Top = tuple[int | None, int | None, float | None, float | None]  # bid, ask (micro-units), sizes


class _BookSim:
    """Order-book state machine shared by ``iter_wire_messages`` and ``synthetic_tob``.

    One code path for both guarantees that the fast top-of-book table equals a replay of the
    wire stream. ``run()`` yields time-ordered records; the consumer reads the current state
    (``bids``/``asks``/``ticks``/``top``) at the moment of each yield:

    * ``(t, "book", leg)`` - full snapshot (start, every ``snapshot_interval_s``, after gaps)
    * ``(t, "tick", leg, old_tick, new_tick)``
    * ``(t, "update", leg, entries)`` with entries ``(side, price, size, best_bid, best_ask)``
      in application order; ``best_*`` is the top after that entry (``None`` = empty side)
    * ``(t, "gap_start", t_end)`` - nothing is emitted until ``t_end``; the books keep evolving
    * ``(t, "resolved", leg, won)``
    """

    def __init__(self, p: SyntheticParams, truth: SyntheticTruth) -> None:
        self.p, self.truth = p, truth
        n = p.n_legs
        self.bids: list[dict[int, float]] = [{} for _ in range(n)]
        self.asks: list[dict[int, float]] = [{} for _ in range(n)]
        self.ticks: list[int] = [0] * n
        self.top: list[Top] = [(None, None, None, None)] * n
        self.coarse, self.fine = _px_int(p.tick), _px_int(p.fine_tick)
        self.arr_t, self.arr_leg = _arrivals(p, truth)
        idx = np.minimum((self.arr_t / p.dt_s).astype(np.int64), truth.t_s.size - 1)
        self.arr_q = truth.q_fair[idx, self.arr_leg]
        self.arr_pi = truth.pi[idx, self.arr_leg]

    def _select(self, q: float, prev: int | None) -> int:
        p = self.p
        return _px_int(select_tick(q, None if prev is None else prev / PRICE_SCALE, p))

    def _draw(self, rng: np.random.Generator, pi: np.ndarray) -> tuple[list, list]:
        m, L = pi.size, self.p.depth_levels
        z, u_thin, keep = rng.standard_normal((m, 2, L)), rng.random((m, 2, 2)), rng.random((m, 2, L))
        return _level_sizes(pi, z, u_thin, self.p).tolist(), keep.tolist()

    def _rebuild(self, leg: int, q: float, k: int, sizes: list, keep: list) -> list[tuple]:
        """New ladder for ``leg``; unchanged price levels keep their size with probability
        ``1 - level_refresh_prob``. Returns the diff as ``price_change`` entries: removals
        first, then new/resized levels, so intermediate states are never crossed."""
        p, tick = self.p, self.ticks[leg]
        max_t, L, refresh = PRICE_SCALE // tick, p.depth_levels, p.level_refresh_prob
        b, a = _quote_ticks(q * max_t, k, max_t)
        old_b, old_a = self.bids[leg], self.asks[leg]
        new_b: dict[int, float] = {}
        new_a: dict[int, float] = {}
        if b is not None:
            sz, kp = sizes[BID], keep[BID]
            for j in range(min(L, b)):
                px = (b - j) * tick
                prev = old_b.get(px)
                new_b[px] = prev if prev is not None and kp[j] >= refresh else sz[j]
        if a is not None:
            sz, kp = sizes[ASK], keep[ASK]
            for j in range(min(L, max_t - a)):
                px = (a + j) * tick
                prev = old_a.get(px)
                new_a[px] = prev if prev is not None and kp[j] >= refresh else sz[j]

        bb, ba = self.top[leg][0], self.top[leg][1]
        cur_b, cur_a = set(old_b), set(old_a)
        entries: list[tuple] = []
        for px in sorted(cur_b - new_b.keys()):
            cur_b.discard(px)
            if px == bb:
                bb = max(cur_b) if cur_b else None
            entries.append((BID, px, 0.0, bb, ba))
        for px in sorted(cur_a - new_a.keys()):
            cur_a.discard(px)
            if px == ba:
                ba = min(cur_a) if cur_a else None
            entries.append((ASK, px, 0.0, bb, ba))
        for px, s in new_b.items():
            if old_b.get(px) != s:
                if bb is None or px > bb:
                    bb = px
                entries.append((BID, px, s, bb, ba))
        for px, s in new_a.items():
            if old_a.get(px) != s:
                if ba is None or px < ba:
                    ba = px
                entries.append((ASK, px, s, bb, ba))
        self.bids[leg], self.asks[leg] = new_b, new_a
        bpx, apx = (b * tick if b is not None else None), (a * tick if a is not None else None)
        self.top[leg] = (bpx, apx, new_b.get(bpx) if bpx is not None else None,
                         new_a.get(apx) if apx is not None else None)
        return entries

    def _schedule(self) -> list[tuple[float, int, str, float]]:
        p = self.p
        sched = [(g0, 0, "gap_start", g1) for g0, g1 in self.truth.gaps]
        sched += [(g1, 1, "gap_end", g1) for _, g1 in self.truth.gaps if g1 < p.duration_s]
        k = 1
        while k * p.snapshot_interval_s < p.duration_s:
            sched.append((k * p.snapshot_interval_s, 2, "snapshot", 0.0))
            k += 1
        return sorted(sched)

    def run(self) -> Iterator[tuple]:
        p, n, truth = self.p, self.p.n_legs, self.truth
        rng = _rng(p.seed, "book")
        k0 = p.base_spread_ticks + rng.poisson(_spread_lambda(truth.pi[0], p))
        arr_k = (p.base_spread_ticks + rng.poisson(_spread_lambda(self.arr_pi, p))).tolist()
        sizes, keep = self._draw(rng, truth.pi[0])
        for leg in range(n):
            q = float(truth.q_fair[0, leg])
            self.ticks[leg] = self._select(q, None)
            self._rebuild(leg, q, int(k0[leg]), sizes[leg], keep[leg])
            yield (0.0, "book", leg)

        sched, j, in_gap = self._schedule(), 0, False
        arr_t, arr_leg, arr_q = self.arr_t.tolist(), self.arr_leg.tolist(), self.arr_q.tolist()
        for a, t in enumerate(arr_t):
            while j < len(sched) and sched[j][0] <= t:
                ts, _, kind, t_end = sched[j]
                j += 1
                if kind == "gap_start":
                    in_gap = True
                    yield (ts, "gap_start", t_end)
                elif kind == "gap_end":
                    in_gap = False
                if kind in ("gap_end", "snapshot") and not in_gap:
                    for leg in range(n):
                        yield (ts, "book", leg)
            if a % _BLOCK == 0:
                sizes, keep = self._draw(rng, self.arr_pi[a:a + _BLOCK])
            leg, q, r = arr_leg[a], arr_q[a], a % _BLOCK
            new_tick = self._select(q, self.ticks[leg])
            if new_tick != self.ticks[leg]:
                old, self.ticks[leg] = self.ticks[leg], new_tick
                if not in_gap:
                    yield (t, "tick", leg, old, new_tick)
            entries = self._rebuild(leg, q, arr_k[a], sizes[r], keep[r])
            if entries and not in_gap:
                yield (t, "update", leg, entries)
        for ts, _, kind, t_end in sched[j:]:
            if kind == "gap_start":
                in_gap = True
                yield (ts, "gap_start", t_end)
            elif kind == "gap_end":
                in_gap = False
            if kind in ("gap_end", "snapshot") and not in_gap:
                for leg in range(n):
                    yield (ts, "book", leg)
        if p.resolve_at_end:
            for leg in range(n):
                yield (p.duration_s, "resolved", leg, leg == truth.winner)


# --------------------------------------------------------------------------- wire stream
def _wire(p: SyntheticParams, truth: SyntheticTruth | None) -> Iterator[tuple[int, str, dict]]:
    """``(t_recv_ns, "msg" | "gap", payload)`` in receive order."""
    truth = truth if truth is not None else simulate_latent(p)
    basket = synthetic_basket(p)
    yes, no = basket.yes_ids, basket.no_ids
    markets = [leg.condition_id for leg in basket.legs]
    event = {"id": f"{TOKEN_PREFIX}{p.seed}", "ticker": p.name.lower(), "slug": p.name.lower(),
             "title": basket.title, "description": "SYNTHETIC DATA - not a real market"}
    start_ns = _start_ns(p.start)
    sim = _BookSim(p, truth)
    seq = 0
    for rec in sim.run():
        t_srv = start_ns + int(round(rec[0] * 1e9))
        t_recv, ts = t_srv + _RECV_LAG_NS, str(t_srv // 1_000_000)
        kind = rec[1]
        seq += 1
        if kind == "update":
            leg, entries = rec[2], rec[3]
            h = hashlib.sha1(f"{yes[leg]}:{ts}:{seq}".encode()).hexdigest()
            changes = [{
                "asset_id": yes[leg], "price": _fmt_px(px), "size": _fmt_size(s),
                "side": "BUY" if side == BID else "SELL", "hash": h,
                "best_bid": _fmt_px(bb) if bb is not None else "0",
                "best_ask": _fmt_px(ba) if ba is not None else "1",
            } for side, px, s, bb, ba in entries]
            yield t_recv, "msg", {"event_type": "price_change", "market": markets[leg],
                                  "price_changes": changes, "timestamp": ts}
        elif kind == "book":
            leg = rec[2]
            yield t_recv, "msg", {
                "event_type": "book", "asset_id": yes[leg], "market": markets[leg],
                # real wire order: bids ascending and asks descending (best level last)
                "bids": [{"price": _fmt_px(px), "size": _fmt_size(s)} for px, s in sorted(sim.bids[leg].items())],
                "asks": [{"price": _fmt_px(px), "size": _fmt_size(s)}
                         for px, s in sorted(sim.asks[leg].items(), reverse=True)],
                "timestamp": ts, "hash": hashlib.sha1(f"{yes[leg]}:{ts}:{seq}".encode()).hexdigest(),
                "tick_size": _fmt_px(sim.ticks[leg]),
            }
        elif kind == "tick":
            leg = rec[2]
            yield t_recv, "msg", {"event_type": "tick_size_change", "asset_id": yes[leg], "market": markets[leg],
                                  "old_tick_size": _fmt_px(rec[3]), "new_tick_size": _fmt_px(rec[4]),
                                  "timestamp": ts}
        elif kind == "gap_start":
            end_ns = start_ns + int(round(rec[2] * 1e9)) + _RECV_LAG_NS
            yield t_recv, "gap", {"start_ns": t_recv, "end_ns": end_ns, "duration_s": rec[2] - rec[0],
                                  "reason": "synthetic_injected_gap"}
        elif kind == "resolved":
            leg, won = rec[2], rec[3]
            yield t_recv, "msg", {"event_type": "market_resolved", "id": f"{TOKEN_PREFIX}{p.seed}-{leg}",
                                  "market": markets[leg], "assets_ids": [yes[leg], no[leg]],
                                  "winning_asset_id": yes[leg] if won else no[leg],
                                  "winning_outcome": "Yes" if won else "No",
                                  "event_message": event, "timestamp": ts}


def iter_wire_messages(p: SyntheticParams, truth: SyntheticTruth | None = None) -> Iterator[tuple[int, dict]]:
    """``(t_recv_ns, message)`` with Polymarket market-channel JSON (prices/sizes as decimal
    strings, ms timestamps as strings), deterministic given the seed. Injected recording gaps
    are silent here (``truth.gaps``); ``iter_raw_lines`` also writes them as meta lines."""
    for t, kind, payload in _wire(p, truth):
        if kind == "msg":
            yield t, payload


def iter_raw_lines(p: SyntheticParams, truth: SyntheticTruth | None = None) -> Iterator[dict]:
    """Records in the shared raw recording format (``events.py``): a ``session_start`` meta
    line, ``{"t", "src": "synthetic", "conn": 0, "msg"}`` lines, a ``gap`` meta line at the
    start of every injected gap, and a final ``stop`` meta line."""
    basket = synthetic_basket(p)
    start_ns = _start_ns(p.start)
    yield {"t": start_ns, "src": "meta", "kind": "session_start",
           "data": {"synthetic": True, "basket_id": p.name, "seed": p.seed, "mode": "synthetic",
                    "assets": list(basket.yes_ids), "code_version": code_version()}}
    t_last, n_msgs = start_ns, 0
    for t, kind, payload in _wire(p, truth):
        t_last = t
        if kind == "msg":
            n_msgs += 1
            yield {"t": t, "src": "synthetic", "conn": 0, "msg": payload}
        else:
            yield {"t": t, "src": "meta", "kind": "gap", "data": payload}
    yield {"t": t_last, "src": "meta", "kind": "stop", "data": {"synthetic": True, "n_messages": n_msgs}}


# --------------------------------------------------------------------------- top of book
def synthetic_tob(p: SyntheticParams, truth: SyntheticTruth | None = None) -> pd.DataFrame:
    """Long top-of-book table (``events.TOB_COLUMNS``), one row per change of a leg's
    ``(bid, ask, bid_sz, ask_sz, clean)``, computed from the same ladders as the wire stream.

    ``t_ns`` is the receive time of the causing message, ``leg`` the ``leg_id``. At the start
    of each injected gap every leg gets a ``clean=False`` row (last known quotes); the
    post-gap snapshot restores ``clean=True``. Empty sides are ``nan``.
    """
    import pandas as pd

    truth = truth if truth is not None else simulate_latent(p)
    leg_ids = [leg.leg_id for leg in synthetic_basket(p).legs]
    start_ns = _start_ns(p.start) + _RECV_LAG_NS
    sim = _BookSim(p, truth)
    last: list[tuple | None] = [None] * p.n_legs
    rows: list[tuple] = []
    for rec in sim.run():
        kind = rec[1]
        if kind in ("book", "update"):
            leg = rec[2]
            state = (*sim.top[leg], True)
            if state != last[leg]:
                last[leg] = state
                rows.append((start_ns + int(round(rec[0] * 1e9)), leg, *state))
        elif kind == "gap_start":
            t = start_ns + int(round(rec[0] * 1e9))
            for leg, prev in enumerate(last):
                if prev is not None and prev[4]:
                    last[leg] = (*prev[:4], False)
                    rows.append((t, leg, *last[leg]))
    nan = math.nan
    df = pd.DataFrame({
        "t_ns": np.array([r[0] for r in rows], dtype=np.int64),
        "leg": [leg_ids[r[1]] for r in rows],
        "bid": np.array([nan if r[2] is None else r[2] / PRICE_SCALE for r in rows], dtype=float),
        "ask": np.array([nan if r[3] is None else r[3] / PRICE_SCALE for r in rows], dtype=float),
        "bid_sz": np.array([nan if r[4] is None else r[4] for r in rows], dtype=float),
        "ask_sz": np.array([nan if r[5] is None else r[5] for r in rows], dtype=float),
        "clean": np.array([r[6] for r in rows], dtype=bool),
    }, columns=list(TOB_COLUMNS))
    df.attrs.update(synthetic=True, kind="synthetic", basket_id=p.name, seed=p.seed)
    return df


def _locf(t_rows: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    i = np.searchsorted(t_rows, grid, side="right") - 1
    out = values[np.maximum(i, 0)].astype(float)
    out[i < 0] = np.nan
    return out


def calibration_summary(p: SyntheticParams, *, grid_s: float = 60.0,
                        truth: SyntheticTruth | None = None) -> dict[str, float]:
    """Edge-to-cost diagnostics, time-averaged on a ``grid_s`` LOCF grid.

    Executable sums use the strict convention (empty bid = 0, empty ask = 1). The round-trip
    cost of a taker basket trade is ``sum(spread) + 2 * fee_rate * sum(m (1 - m))``;
    ``dislocation_to_cost = shock_size / round_trip_cost`` should sit in ``[0.5, 1.5]``.
    """
    truth = truth if truth is not None else simulate_latent(p)
    tob = synthetic_tob(p, truth)
    t0 = _start_ns(p.start) + _RECV_LAG_NS
    grid = t0 + (np.arange(0.0, p.duration_s, grid_s) * 1e9).astype(np.int64)
    s_bid = np.zeros(grid.size)
    s_ask = np.zeros(grid.size)
    fee = np.zeros(grid.size)
    for leg_id, g in tob.groupby("leg", sort=False):
        t = g["t_ns"].to_numpy()
        bid = np.nan_to_num(_locf(t, g["bid"].to_numpy(), grid), nan=0.0)
        ask = np.nan_to_num(_locf(t, g["ask"].to_numpy(), grid), nan=1.0)
        mid = 0.5 * (bid + ask)
        s_bid += bid
        s_ask += ask
        fee += p.fee_rate * mid * (1.0 - mid)
    spread, f = float(np.mean(s_ask - s_bid)), float(np.mean(fee))
    rt = spread + 2.0 * f
    return {
        "spread_sum": spread, "fee_per_side": f, "round_trip_cost": rt, "shock_size": p.shock_size,
        "dislocation_to_cost": p.shock_size / rt if rt > 0 else math.inf,
        "s_bid_mean": float(np.mean(s_bid)), "s_mid_mean": float(np.mean(0.5 * (s_bid + s_ask))),
        "s_ask_mean": float(np.mean(s_ask)),
        "frac_outside_band": float(np.mean((s_bid > 1.0) | (s_ask < 1.0))),
        "basket_disloc_sd": float(np.std(truth.basket_dislocation)),
    }


# --------------------------------------------------------------------------- recording
def write_synthetic_recording(p: SyntheticParams, root: Path = SYNTH_ROOT) -> Path:
    """Write ``root/<name>_seed<k>/{meta.json, raw/<name>_seed<k>.jsonl.gz}`` and return the
    directory. Byte-for-byte reproducible for a given seed and code version (gzip mtime 0).

    Raises ``ValueError`` for any ``root`` inside ``src.HIST_ROOT``: synthetic data must never
    be mistaken for a real recording.
    """
    from . import HIST_ROOT  # read at call time so tests can redirect it

    root = Path(root).resolve()
    hist = Path(HIST_ROOT).resolve()
    if root == hist or hist in root.parents:
        raise ValueError(f"refusing to write synthetic data under {hist} (real recordings only)")
    out = root / f"{p.name}_seed{p.seed}"
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{out.name}.jsonl.gz"
    tmp = path.with_name(path.name + ".tmp")
    truth = simulate_latent(p)
    n_lines, batch = 0, []
    with open(tmp, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        for rec in iter_raw_lines(p, truth):
            batch.append(json.dumps(rec, separators=(",", ":")))
            if len(batch) >= 4096:
                gz.write(("\n".join(batch) + "\n").encode())
                n_lines += len(batch)
                batch.clear()
        if batch:
            gz.write(("\n".join(batch) + "\n").encode())
            n_lines += len(batch)
    os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    start_ns = _start_ns(p.start)
    meta = {
        "synthetic": True, "kind": "synthetic", "generator": "src.synthetic",
        "basket_id": p.name, "seed": p.seed, "params": p.to_dict(), "code_version": code_version(),
        "basket": basket_to_dict(synthetic_basket(p)),
        "start_ns": start_ns, "end_ns": start_ns + int(round(p.duration_s * 1e9)),
        "gaps": [{"start_s": a, "end_s": b} for a, b in truth.gaps],
        "files": [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "lines": n_lines,
                   "sha256": digest}],
        "notes": "SYNTHETIC DATA - generated, not recorded from Polymarket.",
    }
    meta_tmp = out / "meta.json.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(meta_tmp, out / "meta.json")
    logger.info("wrote SYNTHETIC recording %s (%d lines, %d gaps)", out, n_lines, len(truth.gaps))
    return out


# --------------------------------------------------------------------------- pairing demo
def _quote_mids(q: np.ndarray, pi: np.ndarray, rng: np.random.Generator, p: SyntheticParams) -> np.ndarray:
    """Vectorised top-of-book mids of ``_quote_ticks`` quotes (no dust; fine tick near 0/1)."""
    fine = (q < p.fine_tick_edge) | (q > 1.0 - p.fine_tick_edge)
    tick = np.where(fine, p.fine_tick, p.tick)
    max_t = np.rint(1.0 / tick)
    half = 0.5 * (p.base_spread_ticks - 1 + rng.poisson(_spread_lambda(pi, p)))
    x = q / tick
    bid = np.clip(np.floor(x - half), 1, max_t - 1)
    ask = np.clip(np.ceil(x + half), 1, max_t - 1)
    ask = np.where(ask <= bid, bid + 1, ask)
    over = ask > max_t - 1
    ask = np.where(over, max_t - 1, ask)
    bid = np.where(over, ask - 1, bid)
    return np.round(0.5 * (bid + ask) * tick, 6)


def generate_pairing_demo(seed: int = 11, n_minutes: int = 3 * 1440) -> dict[str, Any]:
    """1-minute mids for a Balance-of-Power-style joint basket and its two marginal events.

    ``joint`` has four legs ``"<D|R> Senate, <D|R> House"``. ``house``/``senate`` are 2-leg
    events whose fair prices are the sums of the matching joint fair prices (e.g.
    ``D House = (D Senate, D House) + (R Senate, D House)``) plus their own independent AR(1)
    dislocations, sweeps and quote noise, so the linear identity holds up to a stationary
    residual. All frames are SYNTHETIC (``attrs["synthetic"] = True``).
    """
    import pandas as pd

    p = SyntheticParams(name="SYNTHETIC_bop4_pairing", n_legs=4, pi0=(0.45, 0.30, 0.15, 0.10),
                        leg_labels=BOP_LABELS, seed=seed, duration_s=float(n_minutes) * 60.0, dt_s=60.0,
                        gap_prob_per_hour=0.0)
    truth = simulate_latent(p)
    rng = _rng(seed, "pairing")
    T = truth.t_s.size
    base = truth.flb + truth.regime[:, None] / p.n_legs
    index = pd.date_range(p.start.replace("Z", ""), periods=n_minutes, freq="1min", tz="UTC")

    def frame(q: np.ndarray, pi: np.ndarray, labels: Sequence[str], event: str) -> pd.DataFrame:
        df = pd.DataFrame(_quote_mids(q, pi, rng, p)[:n_minutes], index=index, columns=list(labels))
        df.index.name = "t"
        df.attrs.update(synthetic=True, kind="synthetic", seed=seed, event=event, value="mid_1min")
        return df

    out: dict[str, Any] = {"joint": frame(truth.q_fair, truth.pi, BOP_LABELS, "SYNTHETIC_bop4_joint")}
    for event, groups, labels in (("house", ((0, 1), (2, 3)), ("Democratic", "Republican")),
                                  ("senate", ((0, 2), (1, 3)), ("Democratic", "Republican"))):
        pi_m = np.column_stack([truth.pi[:, list(g)].sum(axis=1) for g in groups])
        fair = np.column_stack([base[:, list(g)].sum(axis=1) for g in groups])
        rng_d, rng_s = rng.spawn(2)
        d, *_ = _dislocation(p, T, pi_m, rng_d, rng_s)
        q = fair + d + rng.normal(0.0, p.noise_sigma, fair.shape)
        out[event] = frame(q, pi_m, labels, f"SYNTHETIC_{event}_marginal")
    out["params"] = p.to_dict()
    out["synthetic"] = True
    return out


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.synthetic --name SYNTHETIC_demo5 --days 3 --seed 7 [--root PATH]``.

    Parameters come from ``config/markets.json`` when ``--name`` is a configured synthetic
    basket, otherwise from the defaults; ``--days``/``--seed`` override them.
    """
    ap = argparse.ArgumentParser(prog="python -m src.synthetic",
                                 description="Write a seeded SYNTHETIC recording (never real market data).")
    ap.add_argument("--name", default="SYNTHETIC_demo5", help="basket id (SYNTHETIC_ prefix added if missing)")
    ap.add_argument("--days", type=float, default=None, help="duration in days (default: config or 3)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: config or 7)")
    ap.add_argument("--root", type=Path, default=SYNTH_ROOT, help="output root (never data/historical_books)")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH, help="markets.json with synthetic_baskets")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    name = args.name if args.name.startswith(SYNTHETIC_PREFIX) else SYNTHETIC_PREFIX + args.name
    params: dict[str, Any] = {}
    seed = args.seed
    try:
        entry = load_markets(args.config).synthetic_baskets.get(name)
    except (OSError, ValueError) as exc:
        logger.warning("could not read %s (%s); using default parameters", args.config, exc)
        entry = None
    if entry:
        params = dict(entry.get("params", {}))
        seed = seed if seed is not None else entry.get("seed")
    if args.days is not None:
        params["duration_s"] = args.days * 86400.0
    p = SyntheticParams.from_dict(params, name=name, seed=seed)
    print(write_synthetic_recording(p, args.root))
    return 0


__all__ = [
    "BOP_LABELS", "SyntheticParams", "SyntheticTruth", "build_ladder", "calibration_summary", "flb",
    "generate_pairing_demo", "iter_raw_lines", "iter_wire_messages", "main", "select_tick",
    "simulate_latent", "synthetic_basket", "synthetic_tob", "write_synthetic_recording",
]

if __name__ == "__main__":
    raise SystemExit(main())
