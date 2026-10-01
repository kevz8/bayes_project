"""Friction-aware execution: depth walking, fees, latency, legging, convert and exact P&L attribution.

Core economics (why the simulator looks the way it does)
--------------------------------------------------------
A *basket* is a mutually exclusive and exhaustive set of binary outcomes (a Polymarket negRisk
event): exactly one leg resolves YES. With YES best bids ``b_i`` and asks ``a_i``::

    S_mid = sum(mid_i)    S_bid = sum(b_i)    S_ask = sum(a_i)    no-arbitrage band: S_bid <= 1 <= S_ask

* **Long basket** = buy one YES of every leg. It costs ``S_ask`` (the VWAP after walking depth)
  plus fees and pays exactly 1 at resolution.
* **Short basket** = BUY one NO of every leg. Polymarket's YES/NO books are mirrored
  (``ask_NO = 1 - bid_YES``), so the set costs ``n - S_bid`` plus fees and pays exactly ``n - 1``.
  The optional NegRisk *convert* turns a full NO set into ``n - 1`` collateral immediately.
* **Taker fee** per order is ``shares * rate * (p (1 - p)) ** exponent``, rounded to 5 dp per
  order. Makers pay 0 and the curve is symmetric in ``p``. On buys the fee is collected in shares
  (``fee_mode="shares"``, the default: buying C shares delivers ``C - fee/p``) or in USD (``"usd"``,
  a sensitivity option). Sells always pay in USD.

A taker round trip therefore earns ``S_bid_entry - min(1, S_ask_exit) - fees`` on the short side
(with convert) and ``max(S_bid_exit - fees, PV(1)) - S_ask_entry - fees`` on the long side.
Reversion of ``S_mid`` *inside* the band cannot be monetised with taker orders. The z-score rule
is statistical convergence, not arbitrage: it pays only when it enters outside the band or when the
sum swings across the whole band. Every trade therefore records ``outside_band_at_entry`` and an
exact P&L attribution, so the results show which of the two happened.

Timeline
--------
A decision at ``t`` uses the book at ``t`` only. Leg ``i`` reaches the book at ``t + latency_i``
and fills against the latest book state at or before that time. The caller runs
:meth:`ExecutionSimulator.on_time` *before* applying the frame stamped ``t``. On polled data, set
``require_newer_book`` so a fill can never see the decision snapshot.

This module is on the hot path. It imports only the standard library and NumPy.
"""
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .config import Basket, Defaults, FeeSchedule
from .events import ASK, BID, BookView, Side

log = logging.getLogger(__name__)

FeeMode = Literal["shares", "usd"]
ExitPolicy = Literal["z_exit", "hold", "hybrid"]
GateMode = Literal["none", "edge", "arb_only"]

FEE_MODES = ("shares", "usd")
EXIT_POLICIES = ("z_exit", "hold", "hybrid")
GATES = ("none", "edge", "arb_only")
TOKENS = ("YES", "NO")
# How a trade closed. "unwind" is used only for failed entries whose fills were all sold back.
EXIT_METHODS = ("sell", "convert", "resolution", "mark_end", "unwind")
# pnl_net == gross_mid + latency_drift + half_spread + depth_slippage + payoff_adjustment
#            - fees - gas - legging_cost      (exact by construction; see TradeRecord)
ATTRIBUTION_KEYS = ("gross_mid", "latency_drift", "half_spread", "depth_slippage",
                    "payoff_adjustment", "fees", "gas", "legging_cost")
COUNTER_KEYS = ("entries", "entries_rejected", "entry_failed", "leg_failures", "legging_events",
                "deferred_fills", "exits", "partial_exits", "converts", "holds", "resolutions")

# Rough, UNVERIFIED gas units for direct (non-relayed) transactions; see config notes.
DEFAULT_GAS_UNITS: Mapping[str, int] = {
    "clob_fill": 0, "split": 200_000, "merge": 200_000, "redeem": 150_000,
    "convert_base": 100_000, "convert_per_leg": 80_000,
}

QTY_EPS = 1e-9      # share quantities below this are floating-point noise
LOT = 0.01          # share granularity of planned orders
LEGGING_TOL = 0.01  # a legging event needs an excess of at least max(LOT, 1% of the target)
MAX_LEVELS = 200    # depth levels read from a BookView
_PX_EPS = 1e-9      # tolerance when comparing a level price with a limit price
_NS = 1_000_000_000
_YEAR_S = 365.25 * 86_400.0


# --------------------------------------------------------------------------- book walking
@dataclass(frozen=True, slots=True)
class LevelFill:
    price: float
    qty: float
    fee_unrounded: float


@dataclass(frozen=True, slots=True)
class FillResult:
    """One taker order walked through one price ladder.

    ``asset_id`` is always the leg's **YES** token id (the book that was walked); ``token`` says
    whether YES or NO was traded, so a portfolio key is ``(asset_id, token)``. ``fee`` is the
    USD fee rounded to 5 dp per order. ``shares_delivered`` is what a buy adds to holdings: it
    equals ``sum_k(take_k - fee_k / p_k)`` for shares-mode buys and ``filled`` otherwise. For
    sells it equals ``filled``, the number of shares given up.
    """

    asset_id: str
    token: str
    side: str
    requested: float
    filled: float
    notional: float
    vwap: float
    top_price: float
    worst_price: float
    fee: float
    shares_delivered: float
    levels: tuple[LevelFill, ...]
    insufficient: bool
    t_ns: int
    fee_mode: str = "shares"

    @property
    def fee_shares(self) -> float:
        """Shares withheld as the fee (only shares-mode buys)."""
        return self.filled - self.shares_delivered if self.side == "buy" else 0.0

    @property
    def fee_cost(self) -> float:
        """Fee in USD terms. For shares-mode buys the withheld shares are valued at the VWAP."""
        if self.side == "buy" and self.fee_mode == "shares":
            return self.fee_shares * self.vwap if self.filled > 0.0 else 0.0
        return self.fee

    @property
    def cash_delta(self) -> float:
        """Signed change in cash: buys pay the notional (+ the USD fee in usd mode); sells
        receive the notional minus the USD fee."""
        if self.side == "buy":
            return -(self.notional + (self.fee if self.fee_mode == "usd" else 0.0))
        return self.notional - self.fee

    @property
    def holdings_delta(self) -> float:
        return self.shares_delivered if self.side == "buy" else -self.filled


def _ladder(prices: Any, sizes: Any) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(prices, dtype=float).ravel()
    s = np.asarray(sizes, dtype=float).ravel()
    if p.shape != s.shape:
        raise ValueError(f"prices and sizes differ in length: {p.size} != {s.size}")
    return p, s


def _check_modes(side: str, fee_mode: str) -> None:
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if fee_mode not in FEE_MODES:
        raise ValueError(f"fee_mode must be one of {FEE_MODES}, got {fee_mode!r}")


def _eligible(p: np.ndarray, s: np.ndarray, limit_price: float | None, side: str) -> np.ndarray:
    """Sizes we may take: valid levels within the limit (buys: p <= limit, sells: p >= limit).
    Prices are best-first, so the levels beyond the limit form a suffix."""
    ok = np.isfinite(p) & np.isfinite(s) & (s > 0.0)
    if limit_price is not None:
        lim = float(limit_price)
        ok &= (p <= lim + _PX_EPS) if side == "buy" else (p >= lim - _PX_EPS)
    return np.where(ok, s, 0.0)


def walk_book(prices: Any, sizes: Any, qty: float, *, limit_price: float | None = None,
              side: str = "buy", fee: FeeSchedule | None = None, fee_mode: FeeMode = "shares",
              asset_id: str = "", token: str = "YES", t_ns: int = 0) -> FillResult:
    """Take ``qty`` shares from a best-first ladder (asks ascending for buys, bids descending
    for sells). This is a vectorised marketable FAK order.

    ``take_k = clip(qty - (cum_k - size_k), 0, size_k)`` over the levels inside the limit. The fee
    is computed per level and rounded once per order. ``insufficient`` is True when the eligible
    depth runs out before ``qty`` is filled.
    """
    _check_modes(side, fee_mode)
    p, s = _ladder(prices, sizes)
    want = float(qty)
    if not (math.isfinite(want) and want >= 0.0):
        raise ValueError(f"qty must be finite and >= 0, got {qty!r}")
    avail = _eligible(p, s, limit_price, side)
    cum = np.cumsum(avail)
    take = np.clip(want - (cum - avail), 0.0, avail)
    hit = take > 0.0
    px, tk = p[hit], take[hit]
    filled = float(tk.sum())
    notional = float(np.dot(tk, px))
    lvl_fee = fee.taker_fee_vec(tk, px) if fee is not None and fee.rate != 0.0 else np.zeros_like(tk)
    delivered = filled
    if side == "buy" and fee_mode == "shares":
        delivered = filled - float(np.divide(lvl_fee, px, out=np.zeros_like(px), where=px > 0.0).sum())
    return FillResult(
        asset_id=asset_id, token=token, side=side, requested=want, filled=filled, notional=notional,
        vwap=notional / filled if filled > 0.0 else math.nan,
        top_price=float(p[0]) if p.size else math.nan,
        worst_price=float(px[-1]) if px.size else math.nan,
        fee=round(float(lvl_fee.sum()), 5), shares_delivered=delivered,
        levels=tuple(LevelFill(float(a), float(b), float(c)) for a, b, c in zip(px, tk, lvl_fee)),
        insufficient=filled < want - QTY_EPS, t_ns=int(t_ns), fee_mode=fee_mode,
    )


def walk_book_notional(prices: Any, sizes: Any, usd: float, **kw: Any) -> FillResult:
    """Market BUY sized in USDC, the way Polymarket market orders are: spend ``usd`` of notional.

    Fees follow ``fee_mode`` on top of that notional: withheld shares, or extra USD.
    ``insufficient`` means the eligible depth costs less than ``usd``.
    """
    if kw.get("side", "buy") != "buy":
        raise ValueError("a notional walk is a market BUY; sells are sized in shares")
    p, s = _ladder(prices, sizes)
    budget = float(usd)
    if not (math.isfinite(budget) and budget >= 0.0):
        raise ValueError(f"usd must be finite and >= 0, got {usd!r}")
    avail = _eligible(p, s, kw.get("limit_price"), "buy")
    pz = np.where(avail > 0.0, p, 0.0)
    cost = avail * pz
    before = np.cumsum(cost) - cost
    shares = np.divide(budget - before, pz, out=np.zeros_like(pz), where=pz > 0.0)
    qty = float(np.clip(shares, 0.0, avail).sum())
    res = walk_book(p, s, qty, **kw)
    return replace(res, insufficient=res.notional < budget - 1e-9)


def mirror_to_no(yes_prices: Any, yes_sizes: Any) -> tuple[np.ndarray, np.ndarray]:
    """YES ladder -> the NO ladder on the other side of the same book: ``p_NO = 1 - p_YES``.

    The best-first order is preserved. YES bids (descending) become NO asks (ascending), and YES
    asks become NO bids.
    """
    p, s = _ladder(yes_prices, yes_sizes)
    return 1.0 - p, s.copy()


def token_ladder(view: BookView, yes_id: str, token: str, side: str,
                 max_levels: int = MAX_LEVELS) -> tuple[np.ndarray, np.ndarray]:
    """The best-first ladder a ``side`` order in ``token`` walks. Every leg has one book:
    buy YES -> YES asks, sell YES -> YES bids, buy NO -> 1 - YES bids, sell NO -> 1 - YES asks."""
    if token not in TOKENS:
        raise ValueError(f"token must be one of {TOKENS}, got {token!r}")
    _check_modes(side, "shares")
    book_side = ASK if (side == "buy") == (token == "YES") else BID
    px, sz = view.depth(yes_id, book_side, max_levels)
    return mirror_to_no(px, sz) if token == "NO" else _ladder(px, sz)


def take_liquidity(view: BookView, yes_id: str, token: str, side: str, qty: float, *,
                   limit_price: float | None = None, fee: FeeSchedule | None = None,
                   fee_mode: FeeMode = "shares", t_ns: int = 0) -> FillResult:
    """Buy or sell ``qty`` of ``token`` on leg ``yes_id`` against the current view."""
    px, sz = token_ladder(view, yes_id, token, side)
    return walk_book(px, sz, qty, limit_price=limit_price, side=side, fee=fee, fee_mode=fee_mode,
                     asset_id=yes_id, token=token, t_ns=t_ns)


def short_leg_via_no(view: BookView, yes_id: str, qty: float, *, limit_price: float | None = None,
                     fee: FeeSchedule | None = None, fee_mode: FeeMode = "shares", t_ns: int = 0) -> FillResult:
    """Short YES on one leg by **buying NO**. This is pro-tip 1 in code.

    Polymarket has no naked shorting: you can only sell tokens you hold. Buying NO needs no
    inventory and is fully collateralised at ``1 - b``. The CLOB *mint-matches* a BUY YES at ``p``
    against a BUY NO at ``1 - p`` by splitting collateral into a YES+NO pair. A resting YES bid at
    ``b`` is therefore a NO ask at ``1 - b``, so this walks the mirrored YES **bids**. The fee is the
    same as for selling YES at ``b``, because the fee curve is symmetric. A full NO set across a
    basket pays exactly ``n - 1``, and a NegRisk convert turns it into ``n - 1`` collateral at once.
    """
    return take_liquidity(view, yes_id, "NO", "buy", qty, limit_price=limit_price, fee=fee,
                          fee_mode=fee_mode, t_ns=t_ns)


def gross_up_factor(fee: FeeSchedule, price: float, fee_mode: FeeMode = "shares") -> float:
    """Order-size multiplier that makes a shares-mode buy deliver the target quantity.

    Buying Q shares at ``p`` delivers ``Q (1 - phi(p)/p)``, where ``phi(p) = r (p(1-p))^e``. For
    ``e = 1`` that is ``Q (1 - r (1 - p))``. Grossing every leg up by the inverse keeps the
    delivered quantities of a basket equal, so the set stays complete.
    """
    if fee_mode != "shares" or fee.rate == 0.0 or not price > 0.0:
        return 1.0
    keep = 1.0 - fee.fee_per_unit(price) / price
    return 1.0 / keep if keep > 0.0 else math.inf


def order_size_for_delivery(prices: Any, sizes: Any, target: float, *, fee: FeeSchedule | None,
                            fee_mode: FeeMode = "shares", limit_price: float | None = None) -> float:
    """Buy size whose shares-mode walk of this ladder delivers exactly ``target`` shares.

    Level ``k`` delivers ``1 - phi(p_k)/p_k`` shares per share bought, so ``delivered(Q)`` is
    piecewise linear and can be inverted level by level. With a single level this is the
    ``gross_up_factor``. Past the eligible depth, the remainder is grossed up at the top price.
    """
    if fee_mode != "shares" or fee is None or fee.rate == 0.0:
        return float(target)
    p, s = _ladder(prices, sizes)
    avail = _eligible(p, s, limit_price, "buy")
    pz = np.where(avail > 0.0, p, 1.0)
    keep = 1.0 - fee.taker_fee_vec(np.ones_like(pz), pz) / pz
    cum = np.cumsum(avail * keep)
    k = int(np.searchsorted(cum, target))
    if k >= cum.size:
        top = float(p[0]) if p.size else math.nan
        return float(avail.sum()) + (target - (float(cum[-1]) if cum.size else 0.0)) * gross_up_factor(fee, top)
    before = float(cum[k - 1]) if k > 0 else 0.0
    return float(avail[:k].sum()) + (target - before) / float(keep[k])


def _yes_mid(bid: float, ask: float) -> float:
    """Midpoint. A one-sided book falls back to its only quote; an empty book gives nan."""
    hb, ha = math.isfinite(bid), math.isfinite(ask)
    if hb and ha:
        return 0.5 * (bid + ask)
    return bid if hb else (ask if ha else math.nan)


def _clean(x: float) -> float:
    return 0.0 if x <= QTY_EPS else x


# --------------------------------------------------------------------------- cost models
@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Signal-to-fill delay per leg: ``base_ms + U(0, jitter_ms)``, drawn independently per leg."""

    base_ms: float = 500.0
    jitter_ms: float = 250.0

    def __post_init__(self) -> None:
        if not (self.base_ms >= 0.0 and self.jitter_ms >= 0.0):
            raise ValueError("latency base_ms and jitter_ms must be >= 0")

    def sample_ns(self, n_legs: int, rng: np.random.Generator) -> np.ndarray:
        ms = self.base_ms + rng.uniform(0.0, self.jitter_ms, size=int(n_legs))
        return np.rint(ms * 1e6).astype(np.int64)


@dataclass(frozen=True)
class GasModel:
    """On-chain cost in USD: ``units * gwei * 1e-9 * pol_usd``.

    The cost is 0 whenever the relayer pays, which is the default: CLOB fills, split, merge and
    redeem are gasless for users. Convert costs ``convert_base + convert_per_leg * n``. All inputs
    are UNVERIFIED estimates.
    """

    relayer_pays: bool = True
    gwei: float = 600.0
    pol_usd: float = 0.11
    units: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_GAS_UNITS))

    def usd(self, op: str, n_legs: int = 0) -> float:
        if op == "convert":
            u = self.units["convert_base"] + self.units["convert_per_leg"] * int(n_legs)
        elif op in self.units:
            u = self.units[op]
        else:
            raise ValueError(f"unknown gas operation {op!r}; known: {sorted(self.units)} + ['convert']")
        return 0.0 if self.relayer_pays else float(u) * self.gwei * 1e-9 * self.pol_usd


@dataclass(frozen=True)
class ExecConfig:
    """Execution assumptions. The defaults mirror ``Defaults`` in ``config/markets.json``.

    * ``gate``: ``none`` is the pure blueprint z rule. ``edge`` requires the expected exit-formula
      edge times q to exceed ``min_edge_usd``. ``arb_only`` requires the riskless edge (a NO set
      cheaper than ``n - 1``, or a YES set cheaper than 1) times q to exceed ``min_edge_usd``.
    * ``exit_policy``: ``z_exit`` sells, or converts a short. ``hold`` waits for resolution, or
      converts a short. ``hybrid`` sells only when that beats the discounted hold value.
    * ``convert_fee_bips``: ``None`` means use ``basket.convert_fee_bips``.
    * ``require_newer_book``: on polled data, an order fills only against a book whose version is
      newer than its decision book.
    """

    latency: LatencyModel = field(default_factory=LatencyModel)
    gas: GasModel = field(default_factory=GasModel)
    max_slip_ticks: int = 3
    participation_cap: float = 0.5
    max_position_frac: float = 0.2
    min_order_size: float = 5.0
    min_notional_usd: float = 1.0
    fee_mode: FeeMode = "shares"
    convert_enabled: bool = False
    convert_fee_bips: float | None = None
    exit_policy: ExitPolicy = "z_exit"
    gate: GateMode = "none"
    min_edge_usd: float = 0.0
    short_mode: str = "full_basket"
    leg_fail_prob: float = 0.0
    mark_interval_s: float = 60.0
    require_newer_book: bool = False
    rf_annual: float = 0.042

    def __post_init__(self) -> None:
        if self.short_mode == "overvalued_legs":
            raise NotImplementedError(
                "short_mode='overvalued_legs' buys NO only on the legs that look rich. That position "
                "does not pay a fixed n-1, so it is a directional bet, not a hedged basket trade. "
                "Only 'full_basket' is implemented.")
        if self.short_mode != "full_basket":
            raise ValueError(f"unknown short_mode {self.short_mode!r}")
        if self.fee_mode not in FEE_MODES:
            raise ValueError(f"fee_mode must be one of {FEE_MODES}, got {self.fee_mode!r}")
        if self.exit_policy not in EXIT_POLICIES:
            raise ValueError(f"exit_policy must be one of {EXIT_POLICIES}, got {self.exit_policy!r}")
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")
        if self.max_slip_ticks < 0 or not 0.0 < self.participation_cap:
            raise ValueError("max_slip_ticks must be >= 0 and participation_cap > 0")
        if not 0.0 < self.max_position_frac <= 1.0:
            raise ValueError("max_position_frac must be in (0, 1]")
        if not 0.0 <= self.leg_fail_prob <= 1.0:
            raise ValueError("leg_fail_prob must be in [0, 1]")
        if self.mark_interval_s < 0.0 or self.min_order_size < 0.0 or self.min_notional_usd < 0.0:
            raise ValueError("mark_interval_s, min_order_size and min_notional_usd must be >= 0")

    @classmethod
    def from_defaults(cls, defaults: Defaults, **overrides: Any) -> ExecConfig:
        g = dict(defaults.gas)
        units = {**DEFAULT_GAS_UNITS, **dict(g.get("units", {}))}
        cfg = cls(
            latency=LatencyModel(float(defaults.latency_ms), float(defaults.latency_jitter_ms)),
            gas=GasModel(relayer_pays=bool(g.get("relayer_pays", True)), gwei=float(g.get("gwei", 600.0)),
                         pol_usd=float(g.get("pol_usd", 0.11)), units=units),
            max_slip_ticks=int(defaults.max_slip_ticks), participation_cap=float(defaults.participation_cap),
            max_position_frac=float(defaults.max_position_frac), min_order_size=float(defaults.min_order_size),
            min_notional_usd=float(defaults.min_notional_usd), fee_mode=str(defaults.fee_mode),
            convert_enabled=bool(defaults.convert_enabled), rf_annual=float(defaults.risk_free_rate),
        )
        return replace(cfg, **overrides) if overrides else cfg


# --------------------------------------------------------------------------- portfolio
def _sell_value(view: BookView, yes_id: str, token: str, qty: float, fee: FeeSchedule) -> float:
    if qty <= QTY_EPS:
        return 0.0
    px, sz = token_ladder(view, yes_id, token, "sell")
    return walk_book(px, sz, qty, side="sell", fee=fee).cash_delta


def liquidation_value_of(view: BookView, basket: Basket, yes_qty: Mapping[str, float],
                         no_qty: Mapping[str, float], *, convert_enabled: bool = False,
                         convert_fee_bips: float | None = None, convert_gas_usd: float = 0.0) -> float:
    """USD you would get by liquidating these holdings now.

    YES holdings walk the YES bids. NO holdings walk the mirrored YES asks; all sells are net of
    fees. When convert applies (``convert_enabled`` and a complete partition), the hedged complete
    NO set ``h = min_i NO_i`` is worth at least ``(n-1) h (1 - bips/1e4) - gas`` plus the sell-walk
    of the residual NOs, so a NO set is never marked below its convert value. A missing book side
    therefore counts as 0 only when convert does not apply.
    """
    legs = basket.legs
    value = sum(_sell_value(view, lg.yes_token_id, "YES", yes_qty.get(lg.yes_token_id, 0.0), lg.fee) for lg in legs)
    x = [no_qty.get(lg.yes_token_id, 0.0) for lg in legs]
    walk_all = sum(_sell_value(view, lg.yes_token_id, "NO", xi, lg.fee) for lg, xi in zip(legs, x))
    best = walk_all
    if convert_enabled and basket.is_complete_partition and len(legs) >= 2:
        h = min(x)
        if h > QTY_EPS:
            bips = basket.convert_fee_bips if convert_fee_bips is None else convert_fee_bips
            conv = (len(legs) - 1) * h * (1.0 - bips / 1e4) - convert_gas_usd
            conv += sum(_sell_value(view, lg.yes_token_id, "NO", xi - h, lg.fee) for lg, xi in zip(legs, x))
            best = max(best, conv)
    return value + best


class Portfolio:
    """Cash plus token holdings keyed by ``(yes_token_id, "YES"|"NO")``.

    Every cash movement goes through a method, so cash is conserved exactly:
    ``cash = initial + sum(fill.cash_delta) + convert proceeds + settlements - gas``.
    """

    def __init__(self, initial_cash: float = 10_000.0):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.holdings: dict[tuple[str, str], float] = {}

    def qty(self, yes_id: str, token: str) -> float:
        return self.holdings.get((yes_id, token), 0.0)

    def _add(self, key: tuple[str, str], delta: float) -> None:
        v = self.holdings.get(key, 0.0) + delta
        if abs(v) <= QTY_EPS:
            self.holdings.pop(key, None)
        else:
            self.holdings[key] = v

    def apply_fill(self, f: FillResult) -> None:
        if f.filled <= 0.0:
            return
        key = (f.asset_id, f.token)
        if f.side == "sell" and f.filled > self.qty(*key) + 1e-7:
            raise ValueError(f"cannot sell {f.filled} {f.token} of {f.asset_id}: holding {self.qty(*key)} "
                             "(Polymarket has no naked selling)")
        self.cash += f.cash_delta
        self._add(key, f.holdings_delta)

    def pay(self, usd: float) -> None:
        """Debit a non-trade cost (gas)."""
        self.cash -= float(usd)

    def apply_convert(self, basket: Basket, qty: float, fee_bips: float, gas_usd: float) -> float:
        """NegRisk convert: burn ``qty`` NO of every leg and receive ``(n-1) qty (1 - fee_bips/1e4)``
        collateral, less gas. Returns the net cash received."""
        n = basket.n_legs
        if n < 2:
            raise ValueError("convert needs at least two legs")
        for yid in basket.yes_ids:
            if self.qty(yid, "NO") < qty - 1e-7:
                raise ValueError(f"convert of {qty} needs {qty} NO on every leg; {yid} holds {self.qty(yid, 'NO')}")
        for yid in basket.yes_ids:
            self._add((yid, "NO"), -qty)
        cash = (n - 1) * qty * (1.0 - fee_bips / 1e4) - gas_usd
        self.cash += cash
        return cash

    def settle_leg(self, basket: Basket, yes_id: str, outcome: str, *, gas_usd: float = 0.0) -> float:
        """Redeem at resolution. ``"No"`` (leg eliminated): YES_i pays 0 and NO_i pays 1, and only that
        leg settles. ``"Yes"`` (leg won): YES_i pays 1 and NO_i pays 0; every other leg's YES pays 0
        and its NO pays 1. Gas is charged only if something was held. Returns the net cash received."""
        outcome = _norm_outcome(outcome)
        if yes_id not in basket.leg_index:
            raise KeyError(f"{yes_id} is not a leg of {basket.basket_id}")
        settled = [yes_id] if outcome == "No" else list(basket.yes_ids)
        payout, held = 0.0, False
        for yid in settled:
            y = self.holdings.pop((yid, "YES"), 0.0)
            x = self.holdings.pop((yid, "NO"), 0.0)
            held |= y > QTY_EPS or x > QTY_EPS
            payout += y if (outcome == "Yes" and yid == yes_id) else x
        net = payout - (gas_usd if held else 0.0)
        self.cash += net
        return net

    def settle(self, basket: Basket, winning_yes_id: str, *, gas_usd: float = 0.0) -> float:
        """The basket resolved: ``winning_yes_id`` won and every other leg lost."""
        return self.settle_leg(basket, winning_yes_id, "Yes", gas_usd=gas_usd)

    def _basket_holdings(self, basket: Basket) -> tuple[dict[str, float], dict[str, float]]:
        ids = basket.yes_ids
        return ({y: self.qty(y, "YES") for y in ids}, {y: self.qty(y, "NO") for y in ids})

    def liquidation_value(self, view: BookView, basket: Basket, convert_enabled: bool = False, *,
                          convert_fee_bips: float | None = None, convert_gas_usd: float = 0.0) -> float:
        """Exit value of this basket's tokens; see :func:`liquidation_value_of` (convert floor)."""
        yes, no = self._basket_holdings(basket)
        return liquidation_value_of(view, basket, yes, no, convert_enabled=convert_enabled,
                                    convert_fee_bips=convert_fee_bips, convert_gas_usd=convert_gas_usd)

    def mid_value(self, view: BookView, basket: Basket) -> float:
        """Holdings at mid: YES at ``m`` and NO at ``1 - m``. A one-sided book uses its only quote;
        a leg with no quotes contributes 0."""
        total = 0.0
        for yid in basket.yes_ids:
            m = _yes_mid(*view.top(yid))
            if math.isfinite(m):
                total += self.qty(yid, "YES") * m + self.qty(yid, "NO") * (1.0 - m)
        return total


def _norm_outcome(outcome: str) -> str:
    o = str(outcome).strip().capitalize()
    if o not in ("Yes", "No"):
        raise ValueError(f"outcome must be 'Yes' or 'No', got {outcome!r}")
    return o


# --------------------------------------------------------------------------- records
@dataclass(slots=True)
class TradeRecord:
    """One basket trade, from signal to close.

    Money fields are USD. ``entry_cost`` and ``entry_fees`` cover the hedged quantity only; any
    excess over it goes into ``legging_cost``. ``exit_proceeds`` covers sells (net of fee), convert
    (net of its fee, before gas), resolution payouts and the ``mark_end`` value. ``pnl_net`` comes
    from the trade's own cash ledger, and the attribution identity holds to float precision::

        pnl_net = gross_mid + latency_drift + half_spread + depth_slippage + payoff_adjustment
                  - fees - gas - legging_cost

    Token mids are ``m`` for YES and ``1 - m`` for NO. Per hedged share,
    ``gross_mid = sum(m_exit_decision - m_entry_decision)``. ``latency_drift`` is the mid move from
    decision to fill. ``half_spread`` is ``top_fill - m_fill``, signed so it is a cost.
    ``depth_slippage`` is ``vwap - top_fill``. ``payoff_adjustment`` is a convert, resolution or
    mark payout minus the exit-decision mids. Shares-mode fees are valued at the fill VWAP.
    """

    trade_id: int
    basket_id: str
    side: Side
    strategy: str
    t_signal_ns: int
    z_entry: float
    mu_entry: float
    sigma_entry: float
    s_mid_entry: float
    s_bid_entry: float
    s_ask_entry: float
    outside_band_at_entry: bool
    expected_edge: float
    n_legs_entry: int
    qty_target: float
    qty_hedged: float = 0.0
    entry_failed: bool = False
    entry_fills: list[FillResult] = field(default_factory=list)
    unwind_fills: list[FillResult] = field(default_factory=list)
    entry_cost: float = 0.0
    entry_fees: float = 0.0
    t_exit_signal_ns: int | None = None
    z_exit: float = math.nan
    mu_exit: float = math.nan
    s_mid_exit: float = math.nan
    s_bid_exit: float = math.nan
    s_ask_exit: float = math.nan
    exit_reason: str | None = None
    exit_method: str | None = None
    exit_fills: list[FillResult] = field(default_factory=list)
    exit_proceeds: float = 0.0
    exit_fees: float = 0.0
    gas_usd: float = 0.0
    legging_cost: float = 0.0
    hold_value: float = math.nan
    legs_resolved: list[str] = field(default_factory=list)
    t_close_ns: int | None = None
    pnl_net: float = 0.0
    baseline_drift: float = math.nan
    holding_s: float = math.nan
    attribution: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for DataFrames: the scalar fields plus ``attr_<term>``; fill lists are dropped."""
        skip = ("entry_fills", "unwind_fills", "exit_fills", "attribution", "legs_resolved")
        row = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}
        row["side"] = int(self.side)
        row["n_legs_resolved"] = len(self.legs_resolved)
        row.update({f"attr_{k}": v for k, v in self.attribution.items()})
        return row


def attribution_total(attribution: Mapping[str, float]) -> float:
    """Right-hand side of the attribution identity; it equals ``pnl_net``."""
    a = attribution
    return (a["gross_mid"] + a["latency_drift"] + a["half_spread"] + a["depth_slippage"]
            + a["payoff_adjustment"] - a["fees"] - a["gas"] - a["legging_cost"])


@dataclass(slots=True)
class EquityPoint:
    t_ns: int
    equity_liq: float
    equity_mid: float
    cash: float
    gross_exposure: float
    side: int


# --------------------------------------------------------------------------- simulator internals
@dataclass(slots=True)
class _LegSnap:
    bid_px: np.ndarray
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray
    clean: bool

    @property
    def bid(self) -> float:
        return float(self.bid_px[0]) if self.bid_px.size else math.nan

    @property
    def ask(self) -> float:
        return float(self.ask_px[0]) if self.ask_px.size else math.nan

    def token_mid(self, token: str) -> float:
        m = _yes_mid(self.bid, self.ask)
        return m if token == "YES" else 1.0 - m

    def ladder(self, token: str, side: str) -> tuple[np.ndarray, np.ndarray]:
        if (side == "buy") == (token == "YES"):
            px, sz = self.ask_px, self.ask_sz
        else:
            px, sz = self.bid_px, self.bid_sz
        return (1.0 - px, sz) if token == "NO" else (px, sz)


@dataclass(slots=True)
class _BasketSnap:
    """Decision-time copy of every leg's ladders. Sums are conservative: a missing bid counts 0
    and a missing ask counts 1."""

    t_ns: int
    legs: dict[str, _LegSnap]

    @property
    def s_bid(self) -> float:
        return sum(lg.bid if math.isfinite(lg.bid) else 0.0 for lg in self.legs.values())

    @property
    def s_ask(self) -> float:
        return sum(lg.ask if math.isfinite(lg.ask) else 1.0 for lg in self.legs.values())

    @property
    def s_mid(self) -> float:
        return sum(lg.token_mid("YES") for lg in self.legs.values())

    @property
    def spread_sum(self) -> float:
        return self.s_ask - self.s_bid

    def outside_band(self, side: Side) -> bool:
        return self.s_bid > 1.0 if side == Side.SHORT_BASKET else self.s_ask < 1.0


@dataclass(slots=True)
class _Order:
    kind: str                # entry | sell | convert
    yes_id: str
    token: str
    qty: float               # entry: grossed-up order size; sell: hedged shares; convert: sets
    residual_qty: float      # sell: unhedged shares appended to the same order
    limit: float | None
    decision_version: int
    batch: int
    fail: bool = False
    deferred: bool = False


@dataclass(slots=True)
class _LegPos:
    yes_id: str
    token: str
    fee: FeeSchedule
    m_dec: float             # token mid at the entry decision
    last_mid: float          # last finite token mid (fallback when a book is empty)
    fill: FillResult | None = None
    m_fill: float = math.nan
    rem: float = 0.0         # hedged shares still held
    residual: float = 0.0    # unhedged excess still held (its value flows into legging_cost)


@dataclass(slots=True)
class _ExitRequest:
    t_ns: int
    reason: str
    z: float
    mu: float
    s_mid: float
    book_version: int
    snap: _BasketSnap


@dataclass(slots=True)
class _Position:
    side: Side
    rec: TradeRecord
    legs: dict[str, _LegPos]
    batch: int
    pending: int
    state: str = "entering"            # entering | open | exiting
    batch_kind: str = "entry"          # entry | sell | convert | hold
    expect_rem: float = 0.0            # hedged shares per leg meant to stay after this exit batch
    exit_mids: dict[str, float] = field(default_factory=dict)
    pending_exit: _ExitRequest | None = None
    cash_flow: float = 0.0
    mark_value: float = 0.0
    attr: dict[str, float] = field(default_factory=lambda: dict.fromkeys(ATTRIBUTION_KEYS, 0.0))


def _token_for(side: Side) -> str:
    return "YES" if side == Side.LONG_BASKET else "NO"


def _parse_end_ns(end_date: str | None) -> int | None:
    if not end_date:
        return None
    try:
        dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * _NS)


# --------------------------------------------------------------------------- simulator
class ExecutionSimulator:
    """Event-driven taker execution of one basket, one position at a time.

    Typical driver loop, the same for replay and live paper trading:
    ``on_time(t)`` -> apply the frame -> engine and state machine (with ``gate_ok``) ->
    ``enter``/``exit`` -> ``mark(t)``. Then ``on_leg_resolved`` for ``market_resolved`` events and
    ``finalize`` at the end. ``enter`` returns False when it rejects a signal (busy, resolved, or
    zero size). An entry that fills no hedged quantity calls ``on_entry_failed(record)``. In both
    cases the caller should run ``state_machine.on_entry_failed()``.
    """

    def __init__(self, basket: Basket, view: BookView, cfg: ExecConfig, portfolio: Portfolio,
                 rng: np.random.Generator, *,
                 on_entry_failed: Callable[[TradeRecord], None] | None = None):
        if basket.n_legs < 2:
            raise ValueError(f"basket {basket.basket_id!r} needs at least two legs")
        self.basket = basket
        self.view = view
        self.cfg = cfg
        self.portfolio = portfolio
        self.rng = rng
        self.on_entry_failed = on_entry_failed
        self.trades: list[TradeRecord] = []
        self.failed_entries: list[TradeRecord] = []
        self.equity: list[EquityPoint] = []
        self.counters: dict[str, int] = dict.fromkeys(COUNTER_KEYS, 0)
        self.position: _Position | None = None
        self.resolved = False
        self._heap: list[tuple[int, int, _Order]] = []
        self._seq = 0
        self._batch = 0
        self._trade_id = 0
        self._last_mark_ns: int | None = None
        self._end_ns = _parse_end_ns(basket.end_date)

    # ------------------------------------------------------------------ properties
    @property
    def side(self) -> Side:
        return self.position.side if self.position is not None else Side.FLAT

    @property
    def convert_fee_bips(self) -> float:
        return self.basket.convert_fee_bips if self.cfg.convert_fee_bips is None else float(self.cfg.convert_fee_bips)

    @property
    def convert_available(self) -> bool:
        return self.cfg.convert_enabled and self.basket.is_complete_partition and self.basket.n_legs >= 2

    def discount_factor(self, t_ns: int) -> float:
        """``(1 + rf)^-tau`` to the basket's end date. It is 1 when the end date is unknown."""
        if self._end_ns is None or self.cfg.rf_annual == 0.0:
            return 1.0
        tau = max(self._end_ns - int(t_ns), 0) / _NS / _YEAR_S
        return (1.0 + self.cfg.rf_annual) ** (-tau)

    # ------------------------------------------------------------------ decision-time helpers
    def _snapshot(self, t_ns: int) -> _BasketSnap:
        legs: dict[str, _LegSnap] = {}
        for yid in self.basket.yes_ids:
            bp, bs = self.view.depth(yid, BID, MAX_LEVELS)
            ap, asz = self.view.depth(yid, ASK, MAX_LEVELS)
            legs[yid] = _LegSnap(np.array(bp, dtype=float), np.array(bs, dtype=float),
                                 np.array(ap, dtype=float), np.array(asz, dtype=float), self.view.is_clean(yid))
        return _BasketSnap(int(t_ns), legs)

    def planned_size(self, side: Side, snap: _BasketSnap | None = None) -> float:
        """Hedged basket quantity for an entry now, floored to 0.01 share::

            min(max_position_frac * equity / unit_cost, cash / unit_cash,
                participation_cap * min_i(depth_i / g_i))

        ``unit_cost`` is ``S_ask`` for a long and ``n - S_bid`` for a short, at the top of book.
        ``depth_i`` is the depth within the slippage limit. ``g_i`` is the shares-mode gross-up. The
        size is 0 if it falls below ``min_order_size``, if any leg's order notional is under
        ``min_notional_usd``, or if any leg is stale or empty.
        """
        side = Side(side)
        if side == Side.FLAT:
            raise ValueError("planned_size needs LONG_BASKET or SHORT_BASKET")
        if self.basket.n_legs < 2:
            return 0.0
        snap = snap if snap is not None else self._snapshot(0)
        token, cfg = _token_for(side), self.cfg
        unit_cost = unit_cash = 0.0
        q_depth = math.inf
        leg_notional: list[float] = []
        for leg in self.basket.legs:
            ls = snap.legs[leg.yes_token_id]
            px, sz = ls.ladder(token, "buy")
            if not ls.clean or px.size == 0 or not math.isfinite(px[0]):
                return 0.0
            top = float(px[0])
            g = gross_up_factor(leg.fee, top, cfg.fee_mode)
            depth = float(sz[px <= top + cfg.max_slip_ticks * leg.tick_size + _PX_EPS].sum())
            q_depth = min(q_depth, cfg.participation_cap * depth / g)
            unit_cost += top
            unit_cash += top * g if cfg.fee_mode == "shares" else top + leg.fee.fee_per_unit(top)
            leg_notional.append(top * g)
        if not unit_cost > 0.0:
            return 0.0
        cash = self.portfolio.cash
        equity = cash + self._portfolio_liquidation()
        q = min(cfg.max_position_frac * equity / unit_cost, cash / unit_cash, q_depth)
        if not q > 0.0:
            return 0.0
        q = round(math.floor(q / LOT + 1e-9) * LOT, 2)
        if q < cfg.min_order_size or any(q * c < cfg.min_notional_usd for c in leg_notional):
            return 0.0
        return q

    def _edges(self, side: Side, t_ns: int, mu: float, snap: _BasketSnap, q: float) -> tuple[float, float]:
        """(expected edge, riskless edge) per hedged unit, using the exit formulas on the decision book.

        Entry cost per unit is ``sum_i cash_i / delivered_i``, from walking the planned size with the
        slippage limit; it includes fees and the gross-up. Exit values per unit, offered as the exit
        policy allows:

        * short: convert ``(n-1)(1 - bips) - gas/q``; hold ``(n-1) * DF``; sell
          ``n - E[S_ask_exit] - fees_out`` with ``E[S_ask_exit] = mu + spread_sum/2``.
        * long: sell ``E[S_bid_exit] - fees_out`` with ``E[S_bid_exit] = mu - spread_sum/2``; hold ``DF``.

        The riskless edge is ``(n-1) - cost`` for a short and ``1 - cost`` for a long.
        """
        token, cfg = _token_for(side), self.cfg
        legs = self.basket.legs
        n = len(legs)
        q_eval = q if q > 0.0 else 1.0
        entry = 0.0
        for leg in legs:
            px, sz = snap.legs[leg.yes_token_id].ladder(token, "buy")
            if px.size == 0:
                return math.nan, math.nan
            limit = float(px[0]) + cfg.max_slip_ticks * leg.tick_size
            size = order_size_for_delivery(px, sz, q_eval, fee=leg.fee, fee_mode=cfg.fee_mode, limit_price=limit)
            f = walk_book(px, sz, size, limit_price=limit, side="buy", fee=leg.fee, fee_mode=cfg.fee_mode)
            if f.shares_delivered <= 0.0:
                return math.nan, math.nan
            entry += -f.cash_delta / f.shares_delivered
        ref = mu if math.isfinite(mu) else snap.s_mid
        spread, df, pol = snap.spread_sum, self.discount_factor(t_ns), cfg.exit_policy
        cands: list[float] = []
        if side == Side.SHORT_BASKET:
            if self.convert_available:
                cands.append((n - 1) * (1.0 - self.convert_fee_bips / 1e4) - cfg.gas.usd("convert", n) / q_eval)
            if pol in ("hold", "hybrid"):
                cands.append((n - 1) * df)
            if pol in ("z_exit", "hybrid"):
                exp_ask = ref + 0.5 * spread
                cands.append(n - exp_ask - self._exit_fees(snap, "ask", (exp_ask - snap.s_ask) / n))
            riskless = (n - 1) - entry
        else:
            if pol in ("z_exit", "hybrid"):
                exp_bid = ref - 0.5 * spread
                cands.append(exp_bid - self._exit_fees(snap, "bid", (exp_bid - snap.s_bid) / n))
            if pol in ("hold", "hybrid"):
                cands.append(df)
            riskless = 1.0 - entry
        cands = [c for c in cands if math.isfinite(c)]
        return (max(cands) - entry if cands else math.nan), riskless

    def _exit_fees(self, snap: _BasketSnap, quote: str, shift: float) -> float:
        """Per-unit taker fees of the expected exit: every leg's current ``quote`` ("bid"/"ask"; a
        missing bid counts 0, a missing ask 1) moved by ``shift`` so the sum hits its expected
        value. YES at ``p`` and NO at ``1 - p`` pay the same fee, so this serves both sides."""
        total = 0.0
        for leg in self.basket.legs:
            ls = snap.legs[leg.yes_token_id]
            p = ls.bid if quote == "bid" else ls.ask
            p = p if math.isfinite(p) else (0.0 if quote == "bid" else 1.0)
            total += leg.fee.fee_per_unit(min(max(p + shift, 0.0), 1.0))
        return total

    def gate_ok(self, side: Side, t_ns: int, mu: float, sigma: float = math.nan) -> tuple[bool, float, bool]:
        """Should a z signal be acted on? Returns ``(ok, edge_per_unit, outside_band)``.

        * ``none``: always True, the pure blueprint rule. The expected edge is still returned.
        * ``edge``: ``expected_edge * q > min_edge_usd``.
        * ``arb_only``: ``riskless_edge * q > min_edge_usd``, ignoring z. The returned edge is the
          riskless one.

        ``outside_band`` is ``S_bid > 1`` for a short or ``S_ask < 1`` for a long, at the top of
        book. ``sigma`` is accepted for parity with the z-engine state; the exit formulas need only
        ``mu``.
        """
        side = Side(side)
        snap = self._snapshot(t_ns)
        q = self.planned_size(side, snap)
        edge, riskless = self._edges(side, t_ns, mu, snap, q)
        outside = snap.outside_band(side)
        if self.cfg.gate == "none":
            return True, edge, outside
        if self.cfg.gate == "edge":
            return bool(q > 0.0 and edge * q > self.cfg.min_edge_usd), edge, outside
        return bool(q > 0.0 and riskless * q > self.cfg.min_edge_usd), riskless, outside

    # ------------------------------------------------------------------ signals
    def enter(self, t_ns: int, side: Side, *, z: float = math.nan, mu: float = math.nan,
              sigma: float = math.nan, s_mid: float = math.nan, strategy: str = "z",
              book_version: int = 0) -> bool:
        """Schedule one taker order per leg at ``t + latency_i``. Long buys the YES asks; short buys
        NO, i.e. walks the mirrored YES bids. Each order has limit ``top + max_slip_ticks * tick``
        and, in shares mode, is grossed up so every leg delivers the same quantity."""
        side = Side(side)
        if side == Side.FLAT:
            raise ValueError("enter() needs LONG_BASKET or SHORT_BASKET")
        if self.position is not None or self.resolved or self.basket.n_legs < 2:
            self.counters["entries_rejected"] += 1
            return False
        snap = self._snapshot(t_ns)
        q = self.planned_size(side, snap)
        if q <= 0.0:
            self.counters["entries_rejected"] += 1
            return False
        edge, _ = self._edges(side, t_ns, mu, snap, q)
        token, n = _token_for(side), self.basket.n_legs
        self._trade_id += 1
        rec = TradeRecord(
            trade_id=self._trade_id, basket_id=self.basket.basket_id, side=side, strategy=strategy,
            t_signal_ns=int(t_ns), z_entry=z, mu_entry=mu, sigma_entry=sigma,
            s_mid_entry=s_mid if math.isfinite(s_mid) else snap.s_mid, s_bid_entry=snap.s_bid,
            s_ask_entry=snap.s_ask, outside_band_at_entry=snap.outside_band(side), expected_edge=edge,
            n_legs_entry=n, qty_target=q,
        )
        offsets = self.cfg.latency.sample_ns(n, self.rng)
        fails = self.rng.random(n) < self.cfg.leg_fail_prob
        self._batch += 1
        legs: dict[str, _LegPos] = {}
        for i, leg in enumerate(self.basket.legs):
            yid = leg.yes_token_id
            ls = snap.legs[yid]
            px, sz = ls.ladder(token, "buy")
            m = ls.token_mid(token)
            legs[yid] = _LegPos(yid, token, leg.fee, m_dec=m, last_mid=m)
            limit = float(px[0]) + self.cfg.max_slip_ticks * leg.tick_size
            qty = order_size_for_delivery(px, sz, q, fee=leg.fee, fee_mode=self.cfg.fee_mode, limit_price=limit)
            self._push(int(t_ns) + int(offsets[i]),
                       _Order("entry", yid, token, qty, 0.0, limit, book_version, self._batch, fail=bool(fails[i])))
        self.position = _Position(side=side, rec=rec, legs=legs, batch=self._batch, pending=n)
        self.counters["entries"] += 1
        return True

    def exit(self, t_ns: int, reason: str, *, z: float = math.nan, mu: float = math.nan,
             s_mid: float = math.nan, book_version: int = 0) -> str | None:
        """Decide how to close on the current book and schedule it. Returns the method chosen
        (``sell``, ``convert`` or ``hold``), ``"pending"`` while an order batch is in flight (an exit
        during entry runs when the entry completes), or None when flat.

        Every plan sells the unbalanced extras (``rem_i - h + residual_i``, with ``h = min_i rem_i``).
        The plans differ only in the hedged core ``h``, chosen by ``exit_policy``:
        long: z_exit -> sell; hold -> hold to resolution; hybrid -> sell only if proceeds >= h * PV(1).
        short: the best of convert (if enabled on a complete partition), sell NOs, and hold, as the
        policy allows. Exit walks are market orders; unfilled shares stay and are retried on the
        next ``exit`` or at ``finalize``.
        """
        pos = self.position
        if pos is None:
            return None
        req = _ExitRequest(int(t_ns), reason, z, mu, s_mid, book_version, self._snapshot(t_ns))
        if pos.state == "entering":
            pos.pending_exit = req
            return "pending"
        if pos.state == "exiting":
            return "pending"
        return self._decide_exit(req)

    def on_time(self, t_ns: int, book_version: int = 0) -> list[TradeRecord]:
        """Execute orders due at or before ``t_ns`` against the current view. Call this *before*
        applying the frame stamped ``t_ns``. Returns the trades that closed, including failed entries."""
        out: list[TradeRecord] = []
        deferred: list[tuple[int, int, _Order]] = []
        executed = False
        while self._heap and self._heap[0][0] <= t_ns:
            item = heapq.heappop(self._heap)
            due, _, order = item
            pos = self.position
            if pos is None or order.batch != pos.batch:
                continue
            if self.cfg.require_newer_book and book_version <= order.decision_version:
                if not order.deferred:
                    order.deferred = True
                    self.counters["deferred_fills"] += 1
                deferred.append(item)
                continue
            out.extend(self._execute(order, int(t_ns) if order.deferred else due))
            executed = True
        for item in deferred:
            heapq.heappush(self._heap, item)
        if executed:
            self.mark(t_ns, force=True)
        return out

    def on_leg_resolved(self, yes_id: str, outcome: str, t_ns: int) -> list[TradeRecord]:
        """Settle one leg's resolution. ``"No"``: that leg's YES pays 0 and its NO pays 1. The basket
        shrinks to n-1 legs and an open position continues on the rest. ``"Yes"``: the basket settles
        in full, the position closes with ``exit_method="resolution"`` and no more entries are taken.
        Any order batch in flight is cancelled first; a partial entry completes without unwinding."""
        outcome = _norm_outcome(outcome)
        if yes_id not in self.basket.leg_index:
            log.warning("resolution for %s ignored: not a leg of %s", yes_id, self.basket.basket_id)
            return []
        out = self._abort_batch(t_ns)
        pos = self.position
        snap = self._snapshot(t_ns)
        won = outcome == "Yes"
        affected = list(self.basket.yes_ids) if won else [yes_id]
        held = any(self.portfolio.qty(a, tok) > QTY_EPS for a in affected for tok in TOKENS)
        gas = self.cfg.gas.usd("redeem") if held else 0.0
        if pos is not None:
            a = pos.attr
            for aid in affected:
                leg = pos.legs.pop(aid, None)
                if leg is None:
                    continue
                pi = 1.0 if (leg.token == "YES") == (won and aid == yes_id) else 0.0
                m = self._now_mid(leg)
                a["gross_mid"] += leg.rem * m
                a["payoff_adjustment"] += leg.rem * (pi - m)
                a["legging_cost"] -= leg.residual * pi
                pos.rec.exit_proceeds += leg.rem * pi
                pos.cash_flow += (leg.rem + leg.residual) * pi
            pos.rec.legs_resolved.append(yes_id)
            if gas:
                pos.cash_flow -= gas
                a["gas"] += gas
        self.portfolio.settle_leg(self.basket, yes_id, outcome, gas_usd=gas)
        self.counters["resolutions"] += 1
        if won:
            self.resolved = True
        else:
            self.basket = self.basket.without_leg(yes_id)
        if pos is not None and (won or self._position_empty(pos)):
            if pos.rec.t_exit_signal_ns is None:
                self._record_exit_book(pos.rec, snap, math.nan)
            out.append(self._close(t_ns, "resolution", "resolution"))
        self.mark(t_ns, force=True)
        return out

    def mark(self, t_ns: int, force: bool = False) -> EquityPoint | None:
        """Append an equity point every ``mark_interval_s``, or now if ``force`` is set. Liquidation
        equity uses the convert floor; mid equity is shown for the liquidity haircut. A second
        point at the same timestamp replaces the first."""
        if (not force and self._last_mark_ns is not None
                and t_ns - self._last_mark_ns < self.cfg.mark_interval_s * _NS):
            return None
        cash = self.portfolio.cash
        mid = self.portfolio.mid_value(self.view, self.basket)
        pt = EquityPoint(int(t_ns), cash + self._portfolio_liquidation(), cash + mid, cash, mid, int(self.side))
        if self.equity and self.equity[-1].t_ns == pt.t_ns:
            self.equity[-1] = pt
        else:
            self.equity.append(pt)
        self._last_mark_ns = int(t_ns)
        return pt

    def finalize(self, t_end_ns: int) -> list[TradeRecord]:
        """End of data: cancel orders still in flight and mark any open position at liquidation
        value (``exit_method="mark_end"``). ``hold_value`` records the discounted hold-to-resolution
        value for reference."""
        out = self._abort_batch(t_end_ns)
        self._heap.clear()
        if self.position is not None:
            out.append(self._mark_end(self.position, t_end_ns))
        self.mark(t_end_ns, force=True)
        return out

    # ------------------------------------------------------------------ execution internals
    def _push(self, due_ns: int, order: _Order) -> None:
        heapq.heappush(self._heap, (int(due_ns), self._seq, order))
        self._seq += 1

    def _portfolio_liquidation(self) -> float:
        return self.portfolio.liquidation_value(
            self.view, self.basket, self.cfg.convert_enabled, convert_fee_bips=self.convert_fee_bips,
            convert_gas_usd=self.cfg.gas.usd("convert", self.basket.n_legs))

    def _now_mid(self, leg: _LegPos) -> float:
        m = _yes_mid(*self.view.top(leg.yes_id))
        if math.isfinite(m):
            leg.last_mid = m if leg.token == "YES" else 1.0 - m
        return leg.last_mid

    @staticmethod
    def _position_empty(pos: _Position) -> bool:
        return all(lg.rem <= QTY_EPS and lg.residual <= QTY_EPS for lg in pos.legs.values())

    @staticmethod
    def _record_exit_book(rec: TradeRecord, snap: _BasketSnap, s_mid: float) -> None:
        rec.s_bid_exit, rec.s_ask_exit = snap.s_bid, snap.s_ask
        rec.s_mid_exit = s_mid if math.isfinite(s_mid) else snap.s_mid

    def _apply_fill(self, pos: _Position, fill: FillResult) -> None:
        self.portfolio.apply_fill(fill)
        pos.cash_flow += fill.cash_delta
        gas = self.cfg.gas.usd("clob_fill")
        if gas:
            self.portfolio.pay(gas)
            pos.cash_flow -= gas
            pos.attr["gas"] += gas

    def _execute(self, order: _Order, t: int) -> list[TradeRecord]:
        pos = self.position
        assert pos is not None
        leg = pos.legs.get(order.yes_id)
        if order.kind == "entry" and leg is not None:
            if order.fail:
                self.counters["leg_failures"] += 1
            else:
                fill = take_liquidity(self.view, leg.yes_id, leg.token, "buy", order.qty, limit_price=order.limit,
                                      fee=leg.fee, fee_mode=self.cfg.fee_mode, t_ns=t)
                pos.rec.entry_fills.append(fill)
                if fill.filled > 0.0:
                    self._apply_fill(pos, fill)
                    leg.fill = fill
                    leg.m_fill = self._now_mid(leg)
        elif order.kind == "sell" and leg is not None:
            self._exec_sell(pos, leg, order, t)
        elif order.kind == "convert":
            self._exec_convert(pos, order)
        pos.pending -= 1
        if pos.pending > 0:
            return []
        return self._complete_entry(t) if pos.batch_kind == "entry" else self._complete_exit(t)

    def _complete_entry(self, t: int, *, allow_unwind: bool = True,
                        decide_pending: bool = True) -> list[TradeRecord]:
        """Hedged quantity = min delivered across legs. The excess on each leg is booked at cost into
        ``legging_cost`` and sold straight back, walking that token's bids. Hedged 0 means the
        entry failed. Fee and rounding residues are unwound too, but only an excess of at least
        ``max(LOT, LEGGING_TOL * target)`` counts as a ``legging_events`` entry."""
        pos = self.position
        assert pos is not None
        rec, a = pos.rec, pos.attr
        pos.state, pos.pending = "open", 0
        legs = list(pos.legs.values())
        delivered = [lg.fill.shares_delivered if lg.fill is not None else 0.0 for lg in legs]
        q = min(delivered) if delivered else 0.0
        q = q if q > QTY_EPS else 0.0
        material = False
        for leg, d in zip(legs, delivered):
            leg.rem = q
            if d <= 0.0 or leg.fill is None:
                continue
            f = leg.fill
            cash = -f.cash_delta
            excess = d - q
            if excess > 0.0:
                a["legging_cost"] += cash * excess / d
                leg.residual = _clean(excess)
                material |= excess >= max(LOT, LEGGING_TOL * rec.qty_target)
            if q > 0.0:
                c = cash / d
                a["gross_mid"] -= q * leg.m_dec
                a["latency_drift"] -= q * (leg.m_fill - leg.m_dec)
                a["half_spread"] -= q * (f.top_price - leg.m_fill)
                a["depth_slippage"] -= q * (f.vwap - f.top_price)
                a["fees"] += q * (c - f.vwap)
                rec.entry_cost += q * c
                rec.entry_fees += q * (c - f.vwap)
        rec.qty_hedged = q
        if material:
            self.counters["legging_events"] += 1
        if allow_unwind:
            for leg in legs:
                if leg.residual > 0.0:
                    self._unwind(pos, leg, t)
        if q == 0.0:
            rec.entry_failed = True
            self.counters["entry_failed"] += 1
            if self.on_entry_failed is not None:
                self.on_entry_failed(rec)
            if self._position_empty(pos):
                return [self._close(t, "unwind", "entry_failed")]
        if decide_pending and pos.pending_exit is not None:
            req, pos.pending_exit = pos.pending_exit, None
            self._decide_exit(req)
        return []

    def _unwind(self, pos: _Position, leg: _LegPos, t: int) -> None:
        qty = min(leg.residual, self.portfolio.qty(leg.yes_id, leg.token))
        fill = take_liquidity(self.view, leg.yes_id, leg.token, "sell", qty, fee=leg.fee,
                              fee_mode=self.cfg.fee_mode, t_ns=t)
        pos.rec.unwind_fills.append(fill)
        if fill.filled > 0.0:
            self._apply_fill(pos, fill)
            pos.attr["legging_cost"] -= fill.cash_delta
            leg.residual = _clean(leg.residual - fill.filled)

    def _decide_exit(self, req: _ExitRequest) -> str:
        pos = self.position
        assert pos is not None
        rec, cfg = pos.rec, self.cfg
        if rec.t_exit_signal_ns is None:
            rec.t_exit_signal_ns, rec.exit_reason, rec.z_exit, rec.mu_exit = req.t_ns, req.reason, req.z, req.mu
            self._record_exit_book(rec, req.snap, req.s_mid)
        legs = list(pos.legs.values())
        n = len(legs)
        for leg in legs:
            ls = req.snap.legs.get(leg.yes_id)
            m = ls.token_mid(leg.token) if ls is not None else math.nan
            if math.isfinite(m):
                leg.last_mid = m
            pos.exit_mids[leg.yes_id] = leg.last_mid
        h = min(lg.rem for lg in legs) if legs else 0.0
        extras = {lg.yes_id: max(lg.rem - h, 0.0) + lg.residual for lg in legs}

        def walk_value(leg: _LegPos, qty: float) -> float:
            ls = req.snap.legs.get(leg.yes_id)
            if qty <= QTY_EPS or ls is None:
                return 0.0
            px, sz = ls.ladder(leg.token, "sell")
            return walk_book(px, sz, qty, side="sell", fee=leg.fee).cash_delta

        v_extra = sum(walk_value(lg, extras[lg.yes_id]) for lg in legs)
        values: dict[str, float] = {}  # insertion order breaks ties: convert, sell, hold
        if pos.side == Side.SHORT_BASKET and self.convert_available and h > QTY_EPS:
            values["convert"] = ((n - 1) * h * (1.0 - self.convert_fee_bips / 1e4)
                                 - cfg.gas.usd("convert", n) + v_extra)
        if cfg.exit_policy in ("z_exit", "hybrid"):
            values["sell"] = sum(walk_value(lg, h + extras[lg.yes_id]) for lg in legs)
        if cfg.exit_policy in ("hold", "hybrid"):
            payoff = h if pos.side == Side.LONG_BASKET else (n - 1) * h
            values["hold"] = payoff * self.discount_factor(req.t_ns) + v_extra
        method = max(values, key=values.__getitem__)

        offsets = cfg.latency.sample_ns(n, self.rng)
        self._batch += 1
        pos.batch, pos.batch_kind, pos.pending = self._batch, method, 0
        pos.expect_rem = h if method == "hold" else 0.0
        for i, leg in enumerate(legs):
            part = leg.rem if method == "sell" else max(leg.rem - h, 0.0)
            if part + leg.residual > QTY_EPS:
                self._push(req.t_ns + int(offsets[i]),
                           _Order("sell", leg.yes_id, leg.token, part, leg.residual, None, req.book_version, pos.batch))
                pos.pending += 1
        if method == "convert":
            self._push(req.t_ns + int(offsets.max()),
                       _Order("convert", "", "NO", h, 0.0, None, req.book_version, pos.batch))
            pos.pending += 1
        pos.state = "exiting" if pos.pending else "open"
        self.counters["exits"] += 1
        if method == "hold":
            self.counters["holds"] += 1
        return method

    def _exec_sell(self, pos: _Position, leg: _LegPos, order: _Order, t: int) -> None:
        want_pos = min(order.qty, leg.rem)
        want_res = min(order.residual_qty, leg.residual)
        total = min(want_pos + want_res, self.portfolio.qty(leg.yes_id, leg.token))
        if total <= QTY_EPS:
            return
        fill = take_liquidity(self.view, leg.yes_id, leg.token, "sell", total, fee=leg.fee,
                              fee_mode=self.cfg.fee_mode, t_ns=t)
        pos.rec.exit_fills.append(fill)
        if fill.filled <= 0.0:
            return
        self._apply_fill(pos, fill)
        x = min(fill.filled, want_pos)       # hedged shares first, then the residual
        frac = x / fill.filled
        if x > 0.0:
            a = pos.attr
            m_x = pos.exit_mids.get(leg.yes_id, leg.last_mid)
            m_f = self._now_mid(leg)
            a["gross_mid"] += x * m_x
            a["latency_drift"] += x * (m_f - m_x)
            a["half_spread"] += x * (fill.top_price - m_f)
            a["depth_slippage"] += x * (fill.vwap - fill.top_price)
            a["fees"] += fill.fee * frac
            pos.rec.exit_proceeds += fill.cash_delta * frac
            pos.rec.exit_fees += fill.fee * frac
            leg.rem = _clean(leg.rem - x)
        if fill.filled > x:
            pos.attr["legging_cost"] -= fill.cash_delta * (1.0 - frac)
            leg.residual = _clean(leg.residual - (fill.filled - x))

    def _exec_convert(self, pos: _Position, order: _Order) -> None:
        legs = list(pos.legs.values())
        n = len(legs)
        if n < 2:
            return
        h = min([order.qty] + [lg.rem for lg in legs] + [self.portfolio.qty(lg.yes_id, "NO") for lg in legs])
        if h <= QTY_EPS:
            return
        bips, gas = self.convert_fee_bips, self.cfg.gas.usd("convert", n)
        pos.cash_flow += self.portfolio.apply_convert(self.basket, h, bips, gas)
        gross = (n - 1) * h
        fee = gross * bips / 1e4
        mids = sum(pos.exit_mids.get(lg.yes_id, lg.last_mid) for lg in legs)
        a = pos.attr
        a["gross_mid"] += h * mids
        a["payoff_adjustment"] += gross - h * mids
        a["fees"] += fee
        a["gas"] += gas
        pos.rec.exit_proceeds += gross - fee
        pos.rec.exit_fees += fee
        for leg in legs:
            leg.rem = _clean(leg.rem - h)
        self.counters["converts"] += 1

    def _complete_exit(self, t: int) -> list[TradeRecord]:
        pos = self.position
        assert pos is not None
        pos.state = "open"
        if self._position_empty(pos):
            return [self._close(t, "sell" if pos.batch_kind == "hold" else pos.batch_kind)]
        if any(lg.rem > pos.expect_rem + QTY_EPS or lg.residual > QTY_EPS for lg in pos.legs.values()):
            self.counters["partial_exits"] += 1
        return []

    def _abort_batch(self, t: int) -> list[TradeRecord]:
        """Cancel the order batch in flight. A partial entry completes as filled, without unwinding."""
        pos = self.position
        if pos is None or pos.state == "open":
            return []
        entering = pos.state == "entering"
        self._heap.clear()
        pos.state, pos.pending = "open", 0
        if entering:
            pos.pending_exit = None
            return self._complete_entry(t, allow_unwind=False, decide_pending=False)
        return []

    def _mark_end(self, pos: _Position, t: int) -> TradeRecord:
        rec, a = pos.rec, pos.attr
        legs = list(pos.legs.values())
        n = len(legs)
        kw = dict(convert_enabled=self.cfg.convert_enabled, convert_fee_bips=self.convert_fee_bips,
                  convert_gas_usd=self.cfg.gas.usd("convert", n))

        def value(with_residual: bool) -> float:
            q = {lg.yes_id: lg.rem + (lg.residual if with_residual else 0.0) for lg in legs}
            yes = {k: v for k, v in q.items() if pos.legs[k].token == "YES"}
            no = {k: v for k, v in q.items() if pos.legs[k].token == "NO"}
            return liquidation_value_of(self.view, self.basket, yes, no, **kw)

        v_pos, v_all = value(False), value(True)
        mids = sum(lg.rem * self._now_mid(lg) for lg in legs)
        a["gross_mid"] += mids
        a["payoff_adjustment"] += v_pos - mids
        a["legging_cost"] -= v_all - v_pos
        rec.exit_proceeds += v_pos
        pos.mark_value = v_all
        h = min((lg.rem for lg in legs), default=0.0)
        rec.hold_value = (h if pos.side == Side.LONG_BASKET else (n - 1) * h) * self.discount_factor(t)
        if rec.t_exit_signal_ns is None:
            self._record_exit_book(rec, self._snapshot(t), math.nan)
        return self._close(t, "mark_end", "end_of_data")

    def _close(self, t: int, method: str, reason: str | None = None) -> TradeRecord:
        pos = self.position
        assert pos is not None
        rec, a = pos.rec, pos.attr
        rec.t_close_ns = int(t)
        rec.exit_method = method
        if rec.exit_reason is None:
            rec.exit_reason = reason or method
        rec.gas_usd = a["gas"]
        rec.legging_cost = a["legging_cost"]
        rec.attribution = dict(a)
        rec.pnl_net = pos.cash_flow + pos.mark_value
        rec.baseline_drift = rec.mu_exit - rec.mu_entry
        rec.holding_s = (int(t) - rec.t_signal_ns) / _NS
        self.position = None
        (self.failed_entries if rec.entry_failed else self.trades).append(rec)
        return rec
