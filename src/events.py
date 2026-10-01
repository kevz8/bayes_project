"""Normalised market-data events and the read-only book interface shared by all modules.

Everything that crosses a module boundary lives here so the live client, the order
book, the engine and the execution simulator agree on one vocabulary:

* Prices inside order books are **integer micro-units** (``PRICE_SCALE = 1_000_000``).
  Polymarket's finest tick is 0.0001, so integer keys are exact and the YES/NO mirror
  ``p_no = PRICE_SCALE - p_yes`` is exact as well.
* ``ts_ms`` is the server timestamp (epoch milliseconds); ``t_recv_ns`` is our local
  receive time (``time.time_ns()``). Replays order by ``t_recv_ns``.
* Asset / token IDs are 77-78 digit decimal strings and are **never** converted to int.

This module imports only the standard library and NumPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

import numpy as np

PRICE_SCALE = 1_000_000
BID, ASK = 0, 1
SIDE_NAMES = {BID: "BUY", ASK: "SELL"}

# Long-format top-of-book table shared by recorder exports, synthetic data, data_io and
# stats_tools: one row per top-of-book change of one leg.
#   t_ns:int64 (local receive time, epoch ns) | leg:str (leg_id) | bid,ask:float (nan = empty side)
#   bid_sz,ask_sz:float | clean:bool (False while the book is known stale)
TOB_COLUMNS = ("t_ns", "leg", "bid", "ask", "bid_sz", "ask_sz", "clean")

# Raw recording line format (gzip JSONL, one object per line), written by the recorder
# and by synthetic.write_synthetic_recording:
#   {"t": <t_recv_ns int>, "src": "ws"|"rest_book"|"poll"|"synthetic", "conn": <int>, "msg": <server JSON>}
#   {"t": <t_recv_ns int>, "src": "meta", "kind": "<session_start|connect|disconnect|gap|resync|stop>", "data": {...}}
RAW_SOURCES = ("ws", "rest_book", "poll", "synthetic", "meta")


class Side(IntEnum):
    """Basket position direction. SHORT_BASKET is implemented by BUYING NO on every leg."""

    SHORT_BASKET = -1
    FLAT = 0
    LONG_BASKET = 1


def px_to_int(value: str | float | int) -> int:
    """Decimal price (``"0.48"``, ``".48"``, ``0.48``) -> integer micro-units.

    Raises ``ValueError`` for ``""``/``None``/non-finite input.
    """
    if value is None or value == "":
        raise ValueError("empty price")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"non-finite price {value!r}")
    return int(round(f * PRICE_SCALE))


def px_to_float(p: int) -> float:
    """Integer micro-units -> float price."""
    return p / PRICE_SCALE


def opt_px(value: Any) -> int | None:
    """Optional decimal string -> micro-units; ``""``/``None`` -> ``None``."""
    if value is None or value == "":
        return None
    try:
        return px_to_int(value)
    except ValueError:
        return None


def opt_float(value: Any) -> float | None:
    """Optional decimal string -> float; ``""``/``None``/garbage -> ``None``."""
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# --------------------------------------------------------------------------- events
@dataclass(slots=True)
class BookEvent:
    """Full L2 snapshot for one asset (WS ``book`` or REST ``/book``). Replaces both sides."""

    asset_id: str
    market: str
    bids: list[tuple[int, float]]  # (price micro-units, size) - any order
    asks: list[tuple[int, float]]
    ts_ms: int
    t_recv_ns: int
    source: str = "ws"  # ws | rest | poll | synthetic
    tick_size: int | None = None  # micro-units
    hash: str | None = None


@dataclass(slots=True)
class LevelChange:
    """One entry of a ``price_change`` message. ``size`` is the NEW absolute level size
    (0 removes the level); ``None`` means the field was absent (top-of-book info only)."""

    asset_id: str
    side: int  # BID | ASK
    price: int
    size: float | None
    best_bid: int | None = None  # server's top of book after this entry (v2 schema only)
    best_ask: int | None = None


@dataclass(slots=True)
class PriceChangeEvent:
    market: str
    changes: list[LevelChange]
    ts_ms: int
    t_recv_ns: int
    schema: str = "v2"  # "v2" (price_changes[], since 2025-09-15) | "legacy" (changes[])


@dataclass(slots=True)
class BestBidAskEvent:
    asset_id: str
    market: str
    best_bid: int | None
    best_ask: int | None
    ts_ms: int
    t_recv_ns: int


@dataclass(slots=True)
class LastTradeEvent:
    asset_id: str
    market: str
    price: float
    size: float | None
    side: str | None
    fee_rate_bps: float | None
    ts_ms: int
    t_recv_ns: int


@dataclass(slots=True)
class TickSizeEvent:
    asset_id: str
    market: str
    old_tick: int | None
    new_tick: int
    ts_ms: int
    t_recv_ns: int


@dataclass(slots=True)
class MarketResolvedEvent:
    """One binary leg resolved. In a negRisk basket ``winning_outcome == "No"`` means the
    leg was eliminated (only that leg settles); ``"Yes"`` means that leg won the basket."""

    market: str
    asset_ids: tuple[str, ...]
    winning_asset_id: str | None
    winning_outcome: str | None
    ts_ms: int
    t_recv_ns: int


@dataclass(slots=True)
class NewMarketEvent:
    market: str
    asset_ids: tuple[str, ...]
    event_slug: str | None
    event_id: str | None
    question: str | None
    ts_ms: int
    t_recv_ns: int


@dataclass(slots=True)
class MetaEvent:
    """Local bookkeeping event: session_start | connect | disconnect | gap | resync | stop."""

    kind: str
    data: dict = field(default_factory=dict)
    t_recv_ns: int = 0


Event = Union[
    BookEvent,
    PriceChangeEvent,
    BestBidAskEvent,
    LastTradeEvent,
    TickSizeEvent,
    MarketResolvedEvent,
    NewMarketEvent,
    MetaEvent,
]

CONTROL_EVENT_TYPES = (TickSizeEvent, MarketResolvedEvent, NewMarketEvent, MetaEvent)


# --------------------------------------------------------------------------- book view
@runtime_checkable
class BookView(Protocol):
    """Read-only access to current books, implemented by ``orderbook.BookManager`` and
    ``DictBookView``. All prices returned here are floats in [0, 1]."""

    def top(self, asset_id: str) -> tuple[float, float]:
        """(best_bid, best_ask); ``nan`` for an empty side."""

    def depth(self, asset_id: str, side: int, max_levels: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """(prices, sizes) best-first: bids descending, asks ascending."""

    def is_clean(self, asset_id: str) -> bool:
        """False while the book is known to be stale (after a reconnect/gap, before resync)."""


class DictBookView:
    """Minimal in-memory ``BookView`` for tests, synthetic replays and modelled books.

    ``books[asset_id] = (bids, asks)`` where each side is a sequence of
    ``(price_float, size)`` in any order.
    """

    def __init__(self, books: Mapping[str, tuple[Sequence[tuple[float, float]], Sequence[tuple[float, float]]]] | None = None):
        self._books: dict[str, tuple[list[tuple[float, float]], list[tuple[float, float]]]] = {}
        self._dirty: set[str] = set()
        for aid, (bids, asks) in (books or {}).items():
            self.set_book(aid, bids, asks)

    def set_book(self, asset_id: str, bids: Sequence[tuple[float, float]], asks: Sequence[tuple[float, float]]) -> None:
        b = sorted(((float(p), float(q)) for p, q in bids if q > 0), key=lambda x: -x[0])
        a = sorted(((float(p), float(q)) for p, q in asks if q > 0), key=lambda x: x[0])
        self._books[asset_id] = (b, a)

    def set_dirty(self, asset_id: str, dirty: bool = True) -> None:
        (self._dirty.add if dirty else self._dirty.discard)(asset_id)

    def top(self, asset_id: str) -> tuple[float, float]:
        b, a = self._books.get(asset_id, ([], []))
        return (b[0][0] if b else math.nan, a[0][0] if a else math.nan)

    def depth(self, asset_id: str, side: int, max_levels: int = 50) -> tuple[np.ndarray, np.ndarray]:
        b, a = self._books.get(asset_id, ([], []))
        lv = (b if side == BID else a)[:max_levels]
        return (np.array([p for p, _ in lv], dtype=float), np.array([q for _, q in lv], dtype=float))

    def is_clean(self, asset_id: str) -> bool:
        return asset_id in self._books and asset_id not in self._dirty
