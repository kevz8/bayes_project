"""L2 order books with integer micro-unit prices, and the multi-asset ``BookManager``.

Why integers: Polymarket's finest tick is 0.0001, so micro-units (``PRICE_SCALE = 1e6``)
represent every price exactly. Dict keys never alias, and the YES/NO mirror
``p_no = PRICE_SCALE - p_yes`` is exact.

Update semantics (CLOB market channel):

* ``book`` (WS) and ``/book`` (REST) are full snapshots that replace both sides. Wire
  arrays are not reliably sorted (WS bids ascend and asks descend; REST varies), so the
  best prices come from ``max``/``min``, never from an array position.
* Each ``price_change`` entry carries the NEW ABSOLUTE resting size at one price level.
  It is not an increment, and size 0 removes the level.
* Per asset, an event older than the last applied server timestamp is dropped. An
  event with an equal timestamp is applied, which is safe because sizes are absolute.

Recovery. A book is *dirty* (not trusted) until a snapshot arrives: at start-up, after
a reconnect or recording gap, and after a detected desync. While dirty, level updates
are applied **and** buffered. The next snapshot replaces the book and replays the
buffered updates whose ``ts_ms >= snapshot ts`` (the classic snapshot + delta-replay
merge), so updates that raced the snapshot are not lost.

This module imports only the standard library, NumPy and ``src.events``.
"""
from __future__ import annotations

import heapq
import logging
import math
from collections import deque
from typing import Callable, Iterable

import numpy as np

from .events import (
    ASK,
    BID,
    PRICE_SCALE,
    BestBidAskEvent,
    BookEvent,
    Event,
    PriceChangeEvent,
    TickSizeEvent,
)

logger = logging.getLogger(__name__)

DEFAULT_TICK = 10_000  # 0.01 in micro-units; tick_size_change switches to 0.001 near 0/1
_NAN = math.nan
_EMPTY = np.empty(0, dtype=float)

_STAT_KEYS = (
    "snapshots", "levels", "stale_dropped", "unknown_asset", "pending_replayed",
    "pending_overflow", "desync_mismatch", "desync", "crossed", "dirty_marks",
)


class OrderBook:
    """One asset's L2 book: ``{price_int: size}`` per side, with the best prices cached.

    The ``apply_*`` methods return ``True`` if the top of book (best price or the size
    resting there) changed, ``False`` if the update was applied without touching the
    top, and ``None`` if it was dropped as stale.
    """

    __slots__ = (
        "asset_id", "bids", "asks", "best_bid", "best_ask", "tick", "last_ts_ms",
        "dirty", "pending", "snapshot_ts_ms", "n_updates", "n_replayed",
        "_dirty_floor_ts", "_lost_ts",
    )

    def __init__(self, asset_id: str, tick: int = DEFAULT_TICK, *, max_pending: int = 10_000) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self.asset_id = asset_id
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}
        self.best_bid: int | None = None
        self.best_ask: int | None = None
        self.tick = tick
        self.last_ts_ms = -1
        self.snapshot_ts_ms = -1
        self.n_updates = 0
        self.n_replayed = 0
        self.dirty = True  # unknown until the first snapshot
        self.pending: deque[tuple[int, int, float, int]] = deque(maxlen=max_pending)
        self._dirty_floor_ts = -1  # a snapshot older than this cannot heal the book
        self._lost_ts = -1  # newest ts of a buffered update evicted by the deque bound

    # ------------------------------------------------------------------ updates
    def _top(self) -> tuple[int | None, int | None, float, float]:
        bb, ba = self.best_bid, self.best_ask
        return (bb, ba, self.bids[bb] if bb is not None else 0.0, self.asks[ba] if ba is not None else 0.0)

    def _set_level(self, side: int, price: int, size: float) -> bool:
        before = self._top()
        self.n_updates += 1
        if side == BID:
            if size > 0:
                self.bids[price] = size
                if self.best_bid is None or price > self.best_bid:
                    self.best_bid = price
            elif self.bids.pop(price, None) is not None and price == self.best_bid:
                self.best_bid = max(self.bids) if self.bids else None  # rare O(levels) recompute
        else:
            if size > 0:
                self.asks[price] = size
                if self.best_ask is None or price < self.best_ask:
                    self.best_ask = price
            elif self.asks.pop(price, None) is not None and price == self.best_ask:
                self.best_ask = min(self.asks) if self.asks else None
        return self._top() != before

    def apply_level(self, side: int, price: int, size: float, ts_ms: int) -> bool | None:
        """Set one level to its new absolute ``size``; ``size <= 0`` removes the level."""
        if ts_ms < self.last_ts_ms:
            return None
        self.last_ts_ms = ts_ms
        if self.dirty:
            pending = self.pending
            if len(pending) == pending.maxlen:
                self._lost_ts = max(self._lost_ts, pending[0][3])
            pending.append((side, price, size, ts_ms))
        return self._set_level(side, price, size)

    def apply_snapshot(self, bids: Iterable[tuple[int, float]], asks: Iterable[tuple[int, float]],
                       ts_ms: int) -> bool | None:
        """Replace both sides, then (if dirty) replay buffered updates with ts >= ``ts_ms``.

        A clean book drops snapshots older than its last update. A dirty book accepts any
        snapshot no older than the moment it became dirty: newer updates are in
        ``pending`` and get replayed. If the bounded buffer evicted an update at or after
        the snapshot time, the merge would be incomplete, so the book stays dirty.
        """
        floor = self._dirty_floor_ts if self.dirty else self.last_ts_ms
        if ts_ms < floor:
            return None
        before = self._top()
        self.bids = {p: q for p, q in bids if q > 0}
        self.asks = {p: q for p, q in asks if q > 0}
        self.best_bid = max(self.bids) if self.bids else None
        self.best_ask = min(self.asks) if self.asks else None
        self.snapshot_ts_ms = ts_ms
        self.n_updates += 1
        if self.dirty:
            for side, price, size, t in self.pending:
                if t >= ts_ms:
                    self._set_level(side, price, size)
                    self.n_replayed += 1
            incomplete = self._lost_ts >= ts_ms
            self.pending.clear()
            self._dirty_floor_ts = self._lost_ts if incomplete else -1
            self._lost_ts = -1
            self.dirty = incomplete
        self.last_ts_ms = max(self.last_ts_ms, ts_ms)
        return self._top() != before

    def mark_dirty(self) -> bool:
        """Flag the book as untrusted; returns ``True`` if it was clean before."""
        if self.dirty:
            return False
        self.dirty = True
        self._dirty_floor_ts = self.last_ts_ms
        self.pending.clear()
        self._lost_ts = -1
        return True

    # ------------------------------------------------------------------ reads
    def top(self) -> tuple[int | None, int | None]:
        return self.best_bid, self.best_ask

    def top_sizes(self) -> tuple[float, float]:
        """Sizes resting at the best bid / ask (``nan`` for an empty side)."""
        bb, ba = self.best_bid, self.best_ask
        return (self.bids[bb] if bb is not None else _NAN, self.asks[ba] if ba is not None else _NAN)

    def levels(self, side: int, max_levels: int | None = None) -> list[tuple[int, float]]:
        """``(price_int, size)`` best-first: bids descending, asks ascending."""
        book = self.bids if side == BID else self.asks
        if max_levels is None:
            return sorted(book.items(), reverse=(side == BID))
        pick = heapq.nlargest if side == BID else heapq.nsmallest
        return pick(max_levels, book.items())

    def depth_arrays(self, side: int, max_levels: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Best-first ``(prices, sizes)`` as float arrays (prices in [0, 1])."""
        lv = self.levels(side, max_levels)
        if not lv:
            return _EMPTY.copy(), _EMPTY.copy()
        arr = np.array(lv, dtype=float)
        return arr[:, 0] / PRICE_SCALE, arr[:, 1].copy()

    def is_crossed(self) -> bool:
        return self.best_bid is not None and self.best_ask is not None and self.best_bid >= self.best_ask

    def mirrored(self, no_asset_id: str) -> OrderBook:
        """The complementary NO book, exact in integers.

        Polymarket matches BUY YES at p against BUY NO at 1 - p, so the books are
        mirrored: a YES ask at p is a NO bid at 1 - p and a YES bid at p is a NO ask at
        1 - p. Hence ``ask_NO = 1 - bid_YES``, which is how the basket is shorted.
        """
        s = PRICE_SCALE
        nb = OrderBook(no_asset_id, self.tick, max_pending=1)
        nb.bids = {s - p: q for p, q in self.asks.items()}
        nb.asks = {s - p: q for p, q in self.bids.items()}
        nb.best_bid = s - self.best_ask if self.best_ask is not None else None
        nb.best_ask = s - self.best_bid if self.best_bid is not None else None
        nb.last_ts_ms = self.last_ts_ms
        nb.snapshot_ts_ms = self.snapshot_ts_ms
        nb.dirty = self.dirty
        return nb

    def __repr__(self) -> str:
        return (f"OrderBook({self.asset_id[:12]}..., bid={self.best_bid}, ask={self.best_ask}, "
                f"levels={len(self.bids)}/{len(self.asks)}, dirty={self.dirty})")


def _norm_bid(p: int | None) -> int | None:
    return None if p is None or p <= 0 else p


def _norm_ask(p: int | None) -> int | None:
    return None if p is None or p >= PRICE_SCALE else p


class BookManager:
    """Books for a set of assets; implements ``events.BookView``.

    ``apply(events)`` consumes one whole wire frame and returns the assets whose
    observable state changed, meaning the best price or size on either side, or the
    clean flag. Consumers re-read those books, so a frame is never seen half-applied.

    The desync detector uses the v2 ``price_change`` entries, which carry the server's
    top of book *after that entry*. After the frame, our top is compared with the
    **last** such entry per asset. Sentinels are normalised on both sides: a bid of
    0/absent and an ask of 1/absent both mean an empty side. Only clean books are
    checked. After ``desync_tolerance`` consecutive mismatching frames, the book is
    marked dirty and ``on_resync_needed(asset_id, "desync")`` is called. A crossed book
    (bid >= ask) is flagged in the same way.
    """

    def __init__(self, asset_ids: Iterable[str] = (), *,
                 on_resync_needed: Callable[[str, str], None] | None = None,
                 max_pending: int = 10_000, desync_tolerance: int = 3) -> None:
        if desync_tolerance < 1:
            raise ValueError("desync_tolerance must be >= 1")
        self.on_resync_needed = on_resync_needed
        self.max_pending = max_pending
        self.desync_tolerance = desync_tolerance
        self.books: dict[str, OrderBook] = {}
        self.server_tops: dict[str, tuple[int | None, int | None, int]] = {}  # best_bid_ask events
        self.book_version = 0  # bumped by every apply() whose returned set is non-empty
        self.stats: dict[str, int] = dict.fromkeys(_STAT_KEYS, 0)
        self._mismatch: dict[str, int] = {}
        self.add_assets(asset_ids)

    # ------------------------------------------------------------------ membership
    @property
    def asset_ids(self) -> list[str]:
        return list(self.books)

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self.books

    def add_assets(self, asset_ids: Iterable[str]) -> list[str]:
        added = []
        for aid in asset_ids:
            if aid not in self.books:
                self.books[aid] = OrderBook(aid, max_pending=self.max_pending)
                added.append(aid)
        return added

    def remove_assets(self, asset_ids: Iterable[str]) -> None:
        for aid in asset_ids:
            self.books.pop(aid, None)
            self.server_tops.pop(aid, None)
            self._mismatch.pop(aid, None)

    # ------------------------------------------------------------------ updates
    def apply(self, events: Iterable[Event]) -> set[str]:
        """Apply one frame's events; return assets whose top of book or clean flag changed."""
        books, st = self.books, self.stats
        changed: set[str] = set()
        touched: set[str] = set()
        was_clean: dict[str, bool] = {}
        server_top: dict[str, tuple[int | None, int | None]] = {}
        for ev in events:
            if isinstance(ev, PriceChangeEvent):
                ts, v2 = ev.ts_ms, ev.schema == "v2"
                for ch in ev.changes:
                    aid = ch.asset_id
                    book = books.get(aid)
                    if book is None:
                        st["unknown_asset"] += 1
                        continue
                    was_clean.setdefault(aid, not book.dirty)
                    if ch.size is None:  # entry only carries the server's top of book
                        if ts < book.last_ts_ms:
                            st["stale_dropped"] += 1
                            continue
                    else:
                        res = book.apply_level(ch.side, ch.price, ch.size, ts)
                        if res is None:
                            st["stale_dropped"] += 1
                            continue
                        st["levels"] += 1
                        touched.add(aid)
                        if res:
                            changed.add(aid)
                    if v2:
                        server_top[aid] = (ch.best_bid, ch.best_ask)
            elif isinstance(ev, BookEvent):
                aid = ev.asset_id
                book = books.get(aid)
                if book is None:
                    st["unknown_asset"] += 1
                    continue
                was_clean.setdefault(aid, not book.dirty)
                replayed = book.n_replayed
                res = book.apply_snapshot(ev.bids, ev.asks, ev.ts_ms)
                if res is None:
                    st["stale_dropped"] += 1
                    continue
                if ev.tick_size:
                    book.tick = ev.tick_size
                st["snapshots"] += 1
                st["pending_replayed"] += book.n_replayed - replayed
                self._mismatch[aid] = 0
                server_top.pop(aid, None)  # the snapshot supersedes earlier entries' tops
                touched.add(aid)
                if res:
                    changed.add(aid)
                if book.dirty:
                    st["pending_overflow"] += 1
                    self._resync(aid, "pending_overflow")
            elif isinstance(ev, BestBidAskEvent):
                if ev.asset_id in books:
                    self.server_tops[ev.asset_id] = (ev.best_bid, ev.best_ask, ev.ts_ms)
            elif isinstance(ev, TickSizeEvent):
                self.set_tick(ev.asset_id, ev.new_tick)
        for aid in touched:
            book = books[aid]
            if not book.dirty and book.is_crossed():
                st["crossed"] += 1
                logger.warning("crossed book %s: bid=%s ask=%s", aid, book.best_bid, book.best_ask)
                self._flag(aid, "crossed")
        for aid, (sb, sa) in server_top.items():
            self._check_desync(books[aid], sb, sa)
        for aid, clean_before in was_clean.items():
            if clean_before == books[aid].dirty:
                changed.add(aid)
        if changed:
            self.book_version += 1
        return changed

    def _check_desync(self, book: OrderBook, server_bid: int | None, server_ask: int | None) -> None:
        aid = book.asset_id
        if book.dirty:
            return
        if (_norm_bid(book.best_bid), _norm_ask(book.best_ask)) == (_norm_bid(server_bid), _norm_ask(server_ask)):
            self._mismatch[aid] = 0
            return
        self.stats["desync_mismatch"] += 1
        n = self._mismatch.get(aid, 0) + 1
        if n < self.desync_tolerance:
            self._mismatch[aid] = n
            return
        self.stats["desync"] += 1
        logger.warning("desync %s: ours=(%s, %s) server=(%s, %s) after %d frames",
                       aid, book.best_bid, book.best_ask, server_bid, server_ask, n)
        self._flag(aid, "desync")

    def _flag(self, asset_id: str, reason: str) -> None:
        if self.books[asset_id].mark_dirty():
            self.stats["dirty_marks"] += 1
        self._mismatch[asset_id] = 0
        self._resync(asset_id, reason)

    def _resync(self, asset_id: str, reason: str) -> None:
        if self.on_resync_needed is not None:
            self.on_resync_needed(asset_id, reason)

    def mark_dirty(self, asset_ids: Iterable[str] | None = None, reason: str = "") -> set[str]:
        """Mark books untrusted until their next snapshot; returns the newly dirty assets."""
        ids = self.books if asset_ids is None else asset_ids
        newly = set()
        for aid in ids:
            book = self.books.get(aid)
            if book is not None and book.mark_dirty():
                self._mismatch[aid] = 0
                newly.add(aid)
        if newly:
            self.stats["dirty_marks"] += len(newly)
            logger.debug("marked %d book(s) dirty (%s)", len(newly), reason)
        return newly

    def set_tick(self, asset_id: str, tick: int) -> None:
        book = self.books.get(asset_id)
        if book is not None and tick > 0:
            book.tick = tick

    # ------------------------------------------------------------------ BookView
    def is_clean(self, asset_id: str) -> bool:
        book = self.books.get(asset_id)
        return book is not None and not book.dirty

    def all_clean(self, asset_ids: Iterable[str] | None = None) -> bool:
        return all(self.is_clean(a) for a in (self.books if asset_ids is None else asset_ids))

    def top_int(self, asset_id: str) -> tuple[int | None, int | None]:
        book = self.books.get(asset_id)
        return book.top() if book is not None else (None, None)

    def top(self, asset_id: str) -> tuple[float, float]:
        """``(best_bid, best_ask)`` as floats, ``nan`` for an empty side or unknown asset."""
        book = self.books.get(asset_id)
        if book is None:
            return _NAN, _NAN
        bb, ba = book.best_bid, book.best_ask
        return (bb / PRICE_SCALE if bb is not None else _NAN, ba / PRICE_SCALE if ba is not None else _NAN)

    def top_floats(self, asset_id: str) -> tuple[float, float, float, float]:
        """``(bid, ask, bid_size, ask_size)``; ``nan`` where absent."""
        book = self.books.get(asset_id)
        if book is None:
            return _NAN, _NAN, _NAN, _NAN
        bid, ask = self.top(asset_id)
        bsz, asz = book.top_sizes()
        return bid, ask, bsz, asz

    def depth(self, asset_id: str, side: int, max_levels: int = 50) -> tuple[np.ndarray, np.ndarray]:
        book = self.books.get(asset_id)
        if book is None:
            return _EMPTY.copy(), _EMPTY.copy()
        return book.depth_arrays(side, max_levels)


__all__ = ["ASK", "BID", "DEFAULT_TICK", "PRICE_SCALE", "BookManager", "OrderBook"]
