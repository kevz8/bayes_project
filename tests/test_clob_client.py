"""Order books, wire parsing, queues, REST client and the live feeds (all offline).

The WebSocket tests run a real ``websockets`` server on 127.0.0.1:0 in-process; the
client connects with ``proxy=None``. Backoff randomness and every interval are injected
so the tests are fast and deterministic.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import requests
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from src.clob_client import (
    BOOKS_CHUNK,
    ClobHTTPError,
    ClobRestClient,
    CoalescingQueue,
    DropQueue,
    Heartbeat,
    MarketDataFeed,
    ParseStats,
    RawRecord,
    RestPollingFeed,
    backoff_delay,
    make_feed,
    parse_frame,
    parse_message,
    parse_rest_book,
)
from src.events import (
    ASK,
    BID,
    PRICE_SCALE,
    BestBidAskEvent,
    BookEvent,
    BookView,
    LastTradeEvent,
    LevelChange,
    MarketResolvedEvent,
    MetaEvent,
    NewMarketEvent,
    PriceChangeEvent,
    TickSizeEvent,
    px_to_int,
)
from src.orderbook import BookManager, OrderBook
from tests.conftest import FakeResponse

REPO = Path(__file__).resolve().parents[1]
A_BOOK = "101007741586870489619361069512452187353898396425142157315847015703471254508752"
A_PC = "71321045679252212594626385532706912750332728571942532289631379312455583992563"
A_LEG = "39327269875426915204597944387916069897800289788920336317845465327697809453999"
I1 = "1" * 77
I2 = "2" * 77


def px(s: str) -> int:
    return px_to_int(s)


def fmt(p: int | None, empty: str) -> str:
    return empty if p is None else f"{p / PRICE_SCALE:.6f}"


def book_msg(asset: str, bids: dict[str, float], asks: dict[str, float], ts: int, **extra: Any) -> dict:
    return {"event_type": "book", "asset_id": asset, "market": "0xm", "timestamp": str(ts),
            "bids": [{"price": p, "size": str(q)} for p, q in bids.items()],
            "asks": [{"price": p, "size": str(q)} for p, q in asks.items()], **extra}


def v2(entries: list[tuple], ts: int) -> dict:
    """entries: (asset, side 'BUY'/'SELL', price, size|None, best_bid|None, best_ask|None)."""
    out = []
    for asset, side, price, size, bb, ba in entries:
        e = {"asset_id": asset, "side": side, "price": price}
        if size is not None:
            e["size"] = size
        if bb is not None:
            e["best_bid"] = bb
        if ba is not None:
            e["best_ask"] = ba
        out.append(e)
    return {"event_type": "price_change", "market": "0xm", "timestamp": str(ts), "price_changes": out}


def events_of(msg: Any, t: int = 1) -> list:
    return parse_frame(json.dumps(msg), t)


# =========================================================================== parsing
@pytest.mark.parametrize("name, types", [
    ("ws_book.json", [BookEvent]),
    ("ws_price_change_v2.json", [PriceChangeEvent]),
    ("ws_price_change_legacy.json", [PriceChangeEvent]),
    ("ws_last_trade.json", [LastTradeEvent]),
    ("ws_tick_size_change.json", [TickSizeEvent]),
    ("ws_best_bid_ask.json", [BestBidAskEvent]),
    ("ws_new_market.json", [NewMarketEvent]),
    ("ws_market_resolved.json", [MarketResolvedEvent]),
    ("ws_array_frame.json", [BookEvent, BookEvent]),
])
def test_every_fixture_parses(fixture_text, name, types):
    stats = ParseStats()
    evs = parse_frame(fixture_text(name), 123, stats)
    assert [type(e) for e in evs] == types
    assert all(e.t_recv_ns == 123 for e in evs)
    assert (stats.frames, stats.events, stats.non_json, stats.malformed) == (1, len(types), 0, 0)
    assert not stats.unknown


def test_book_fields_and_ids_stay_strings(fixture_text):
    (ev,) = parse_frame(fixture_text("ws_book.json").encode(), 7)  # bytes frames are fine
    assert ev.asset_id == A_BOOK and isinstance(ev.asset_id, str) and len(ev.asset_id) in (77, 78)
    assert ev.bids == [(10_000, 510000.0), (20_000, 3100.0)]
    assert ev.asks == [(990_000, 58.07), (970_000, 178.73)]
    assert (ev.ts_ms, ev.source, ev.hash) == (1740759191594, "ws", "c0e51b1cfdbcb1b2aec58feaf7b01004019a89c6")
    mgr = BookManager([A_BOOK])
    mgr.apply([ev])
    assert A_BOOK in mgr.books and mgr.top(A_BOOK) == (0.02, 0.97)  # best by max/min, not position


def test_price_change_schemas(fixture_text):
    (new,) = parse_frame(fixture_text("ws_price_change_v2.json"), 1)
    assert new.schema == "v2" and new.ts_ms == 1757908892351
    assert new.changes == [LevelChange(A_PC, BID, 500_000, 200.0, 500_000, PRICE_SCALE)]
    (old,) = parse_frame(fixture_text("ws_price_change_legacy.json"), 1)
    assert old.schema == "legacy"
    assert old.changes == [LevelChange(A_LEG, ASK, 44_000, 611.0, None, None)]


def test_control_and_trade_fields(fixture_json):
    (lt,) = parse_message(fixture_json("ws_last_trade.json"), 1)
    assert (lt.price, lt.size, lt.side, lt.fee_rate_bps, lt.ts_ms) == (0.12, 8.333332, "BUY", 0.0, 1740760245471)
    (tk,) = parse_message(fixture_json("ws_tick_size_change.json"), 1)
    assert (tk.old_tick, tk.new_tick) == (10_000, 1_000)
    (bba,) = parse_message(fixture_json("ws_best_bid_ask.json"), 1)
    assert (bba.best_bid, bba.best_ask) == (730_000, 770_000)
    (nm,) = parse_message(fixture_json("ws_new_market.json"), 1)
    assert (nm.event_slug, nm.event_id, nm.asset_ids) == ("illustrative-election", "90001", (I1, I2))
    (mr,) = parse_message(fixture_json("ws_market_resolved.json"), 1)
    assert (mr.winning_asset_id, mr.winning_outcome, mr.asset_ids) == (I2, "No", (I1, I2))


def test_pong_non_json_unknown_and_malformed_frames():
    stats = ParseStats()
    assert parse_frame("PONG", 1, stats) == [] and stats.pongs == 1
    assert parse_frame("INVALID OPERATION", 1, stats) == [] and stats.non_json == 1
    assert parse_frame(b"\xff\xfe", 1, stats) == [] and stats.non_json == 2
    assert parse_frame(json.dumps({"event_type": "brand_new_type"}), 1, stats) == []
    assert stats.unknown["brand_new_type"] == 1
    assert parse_frame("42", 1, stats) == [] and stats.malformed == 1
    good = book_msg(I1, {"0.4": 1}, {"0.6": 1}, 5)
    bad = {"event_type": "book", "bids": [], "asks": [], "timestamp": "5"}  # no asset_id
    evs = parse_frame(json.dumps([bad, good, "junk"]), 1, stats)
    assert [e.asset_id for e in evs] == [I1]
    assert stats.malformed == 3  # bad book + "junk" element
    no_ts = book_msg(I1, {}, {}, 0)
    del no_ts["timestamp"]
    assert parse_frame(json.dumps(no_ts), 1, stats) == [] and stats.malformed == 4


def test_optional_fields_and_short_prices():
    msg = {"event_type": "last_trade_price", "asset_id": I1, "market": "0xm", "price": ".48",
           "fee_rate_bps": "", "timestamp": "10"}
    (lt,) = events_of(msg)
    assert lt.price == 0.48 and lt.fee_rate_bps is None and lt.size is None and lt.side is None
    assert px_to_int(".48") == 480_000
    (pc,) = events_of(v2([(I1, "SELL", ".52", None, ".5", ".52")], 11))
    assert pc.changes == [LevelChange(I1, ASK, 520_000, None, 500_000, 520_000)]  # size absent -> None


def test_v2_message_with_interleaved_assets():
    msg = v2([(I1, "BUY", "0.40", "10", "0.40", "1"), (I2, "SELL", "0.70", "5", "0", "0.70"),
              (I1, "BUY", "0.41", "3", "0.41", "1")], 20)
    (pc,) = events_of(msg)
    assert [c.asset_id for c in pc.changes] == [I1, I2, I1]
    mgr = BookManager([I1, I2], desync_tolerance=1)
    mgr.apply([BookEvent(I1, "0xm", [], [], 10, 0), BookEvent(I2, "0xm", [], [], 10, 0)])
    assert mgr.apply([pc]) == {I1, I2}
    bid, ask = mgr.top(I1)
    assert bid == 0.41 and math.isnan(ask)
    assert mgr.top_int(I2) == (None, 700_000)
    assert mgr.stats["desync_mismatch"] == 0


def test_parse_rest_book_unsorted(fixture_json):
    ev = parse_rest_book(fixture_json("rest_book.json"), 99)
    assert (ev.source, ev.tick_size, ev.asset_id, ev.ts_ms) == ("rest", 10_000, I1, 1759363200123)
    ob = OrderBook(I1)
    ob.apply_snapshot(ev.bids, ev.asks, ev.ts_ms)
    assert ob.top() == (460_000, 470_000) and ob.top_sizes() == (35.5, 12.0)
    assert [p for p, _ in ob.levels(BID)] == [460_000, 450_000, 440_000]
    assert [p for p, _ in ob.levels(ASK, 2)] == [470_000, 480_000]
    assert parse_rest_book(fixture_json("rest_books.json")[1], 1, source="poll").source == "poll"
    with pytest.raises(ValueError):
        parse_rest_book({"asset_id": I1, "bids": [], "asks": []}, 1)


# =========================================================================== order book
def snap(ob: OrderBook, bids: dict[str, float], asks: dict[str, float], ts: int):
    return ob.apply_snapshot([(px(p), q) for p, q in bids.items()], [(px(p), q) for p, q in asks.items()], ts)


def test_snapshot_best_and_level_updates():
    ob = OrderBook(I1)
    assert ob.dirty and ob.top() == (None, None)
    assert snap(ob, {"0.01": 510000, "0.02": 3100}, {"0.99": 58.07, "0.97": 178.73}, 100) is True
    assert not ob.dirty and ob.top() == (20_000, 970_000)
    assert ob.apply_level(BID, px("0.03"), 5, 101) is True  # better price: O(1) new best
    assert ob.apply_level(BID, px("0.01"), 7, 101) is False  # deep level
    assert ob.apply_level(BID, px("0.03"), 0, 102) is True and ob.best_bid == 20_000  # best removed
    assert ob.apply_level(ASK, px("0.97"), 100, 102) is True and ob.top_sizes() == (3100, 100)
    assert ob.apply_level(ASK, px("0.98"), 0, 102) is False  # removing a missing level is a no-op
    assert ob.bids == {10_000: 7, 20_000: 3100}
    prices, sizes = ob.depth_arrays(BID)
    np.testing.assert_array_equal(prices, [0.02, 0.01])
    np.testing.assert_array_equal(sizes, [3100, 7])
    assert ob.depth_arrays(ASK, 1)[0].tolist() == [0.97]


def test_stale_dropped_equal_timestamp_applied():
    ob = OrderBook(I1)
    snap(ob, {"0.40": 1}, {"0.60": 1}, 100)
    assert ob.apply_level(BID, px("0.41"), 1, 99) is None and ob.best_bid == px("0.40")
    assert ob.apply_level(BID, px("0.41"), 1, 100) is True and ob.best_bid == px("0.41")
    assert snap(ob, {"0.30": 1}, {"0.70": 1}, 99) is None  # stale snapshot on a clean book
    assert ob.best_bid == px("0.41")


def test_dirty_buffer_replays_updates_after_snapshot():
    ob = OrderBook(I1)
    snap(ob, {"0.40": 1}, {"0.60": 1}, 100)
    assert ob.mark_dirty() and not ob.mark_dirty()
    ob.apply_level(BID, px("0.40"), 10, 105)  # older than the snapshot below: superseded
    ob.apply_level(ASK, px("0.60"), 5, 110)
    ob.apply_level(BID, px("0.40"), 0, 120)
    assert snap(ob, {"0.30": 1}, {"0.70": 1}, 99) is None and ob.dirty  # older than the dirty point
    snap(ob, {"0.40": 50, "0.39": 20}, {"0.60": 1, "0.61": 2}, 108)
    assert not ob.dirty and ob.n_replayed == 2
    assert ob.bids == {px("0.39"): 20} and ob.asks == {px("0.60"): 5, px("0.61"): 2}
    assert ob.last_ts_ms == 120 and not ob.pending


def test_pending_overflow_keeps_book_dirty():
    seen = []
    mgr = BookManager([I1], max_pending=2, on_resync_needed=lambda a, r: seen.append((a, r)))
    for ts, p in ((10, "0.40"), (11, "0.41"), (12, "0.42")):
        mgr.apply([PriceChangeEvent("0xm", [LevelChange(I1, BID, px(p), 1.0)], ts, 0, "legacy")])
    mgr.apply([BookEvent(I1, "0xm", [(px("0.30"), 1.0)], [], 10, 0)])  # update at ts 10 was evicted
    assert not mgr.is_clean(I1) and seen == [(I1, "pending_overflow")]
    assert mgr.stats["pending_overflow"] == 1 and mgr.stats["pending_replayed"] == 2
    mgr.apply([BookEvent(I1, "0xm", [(px("0.30"), 1.0)], [], 11, 0)])
    assert mgr.is_clean(I1)


def test_mirrored_book_is_exact():
    ob = OrderBook(I1)
    snap(ob, {"0.43": 100, "0.42": 50}, {"0.57": 10, "0.60": 30}, 5)
    no = ob.mirrored(I2)
    assert no.asset_id == I2
    assert no.bids == {px("0.43"): 10, px("0.40"): 30} and no.asks == {px("0.57"): 100, px("0.58"): 50}
    assert no.best_ask == PRICE_SCALE - ob.best_bid and no.best_bid == PRICE_SCALE - ob.best_ask
    back = no.mirrored(I1)
    assert (back.bids, back.asks, back.top()) == (ob.bids, ob.asks, ob.top())


def test_book_manager_view_and_stats():
    mgr = BookManager([I1, I2])
    assert isinstance(mgr, BookView)
    assert all(math.isnan(x) for x in mgr.top(I1)) and not mgr.is_clean(I1)
    assert all(math.isnan(x) for x in mgr.top("unknown")) and mgr.depth("unknown", BID)[0].size == 0
    v0 = mgr.book_version
    changed = mgr.apply([BookEvent(I1, "0xm", [(px("0.40"), 10.0)], [], 5, 0, tick_size=1_000),
                         BookEvent("nope", "0xm", [], [], 5, 0)])
    assert changed == {I1} and mgr.book_version == v0 + 1 and mgr.stats["unknown_asset"] == 1
    bid, ask = mgr.top(I1)
    assert bid == 0.40 and math.isnan(ask) and mgr.books[I1].tick == 1_000
    assert mgr.top_floats(I1)[2] == 10.0 and math.isnan(mgr.top_floats(I1)[3])
    assert mgr.apply([BestBidAskEvent(I1, "0xm", px("0.5"), px("0.6"), 6, 0)]) == set()
    assert mgr.server_tops[I1] == (px("0.5"), px("0.6"), 6) and mgr.top(I1)[0] == 0.40  # recorded only
    assert mgr.apply([TickSizeEvent(I1, "0xm", 1_000, 10_000, 7, 0)]) == set() and mgr.books[I1].tick == 10_000
    assert mgr.book_version == v0 + 1
    assert mgr.apply([BookEvent(I1, "0xm", [(px("0.40"), 10.0)], [], 8, 0)]) == set()  # identical, clean
    mgr.mark_dirty([I1], "test")
    assert mgr.apply([BookEvent(I1, "0xm", [(px("0.40"), 10.0)], [], 9, 0)]) == {I1}  # clean flag flipped
    assert mgr.all_clean([I1]) and not mgr.all_clean()


def test_crossed_book_flagged():
    seen = []
    mgr = BookManager([I1], on_resync_needed=lambda a, r: seen.append(r))
    mgr.apply([BookEvent(I1, "0xm", [(px("0.40"), 1.0)], [(px("0.50"), 1.0)], 1, 0)])
    mgr.apply([PriceChangeEvent("0xm", [LevelChange(I1, BID, px("0.55"), 1.0)], 2, 0, "legacy")])
    assert mgr.stats["crossed"] == 1 and not mgr.is_clean(I1) and seen == ["crossed"]


def _desync_mgr(tol: int):
    seen: list[tuple[str, str]] = []
    mgr = BookManager([I1], desync_tolerance=tol, on_resync_needed=lambda a, r: seen.append((a, r)))
    mgr.apply([BookEvent(I1, "0xm", [(px("0.40"), 10.0)], [(px("0.60"), 10.0)], 100, 0)])
    return mgr, seen


def test_desync_after_consecutive_mismatches():
    mgr, seen = _desync_mgr(2)
    wrong = lambda ts: events_of(v2([(I1, "BUY", "0.30", "5", "0.45", "0.60")], ts))  # noqa: E731
    right = lambda ts: events_of(v2([(I1, "BUY", "0.31", "5", "0.40", "0.60")], ts))  # noqa: E731
    mgr.apply(wrong(101))
    mgr.apply(right(102))  # resets the consecutive counter
    mgr.apply(wrong(103))
    assert mgr.is_clean(I1) and mgr.stats["desync_mismatch"] == 2 and not seen
    mgr.apply(wrong(104))
    assert not mgr.is_clean(I1) and seen == [(I1, "desync")] and mgr.stats["desync"] == 1
    mgr.apply(wrong(105))  # dirty books are not checked
    assert mgr.stats["desync_mismatch"] == 3


def test_desync_default_tolerance_is_three():
    mgr, seen = _desync_mgr(3)
    for ts in (101, 102):
        mgr.apply(events_of(v2([(I1, "BUY", "0.30", "5", "0.45", "0.60")], ts)))
    assert not seen
    mgr.apply(events_of(v2([(I1, "BUY", "0.30", "5", "0.45", "0.60")], 103)))
    assert seen == [(I1, "desync")]


def test_desync_sentinels_and_last_entry(fixture_json):
    mgr = BookManager([A_PC], desync_tolerance=1)
    mgr.apply([BookEvent(A_PC, "0xm", [], [], 1, 0)])
    for k in range(5):  # captured example: best_ask "1" with an empty ask side
        msg = fixture_json("ws_price_change_v2.json")
        msg["timestamp"] = str(1757908892351 + k)
        mgr.apply(parse_message(msg, 0))
    mgr.apply(events_of(v2([(A_PC, "BUY", "0.5", "0", "0", "1")], 1757908892400)))  # both sides empty
    assert mgr.is_clean(A_PC) and mgr.stats["desync_mismatch"] == 0
    # entry-level tops: only the LAST entry per asset describes the post-frame book
    msg = v2([(A_PC, "BUY", "0.50", "10", "0.50", ""), (A_PC, "BUY", "0.52", "5", "0.52", "")], 1757908892500)
    mgr.apply(events_of(msg))
    assert mgr.is_clean(A_PC) and mgr.stats["desync_mismatch"] == 0
    # a size-less entry carries only the server top and is still checked
    mgr.apply(events_of(v2([(A_PC, "BUY", "0.52", None, "0.53", "")], 1757908892600)))
    assert not mgr.is_clean(A_PC) and mgr.stats["desync"] == 1


def test_desync_skipped_for_legacy_and_dirty_books():
    mgr = BookManager([I1], desync_tolerance=1)
    mgr.apply(events_of(v2([(I1, "BUY", "0.30", "5", "0.99", "0.99")], 1)))  # never snapshotted: dirty
    assert mgr.stats["desync_mismatch"] == 0
    mgr.apply([BookEvent(I1, "0xm", [], [], 2, 0)])
    mgr.apply(events_of({"event_type": "price_change", "asset_id": I1, "market": "0xm", "timestamp": "3",
                         "changes": [{"price": "0.3", "side": "BUY", "size": "1"}]}))
    assert mgr.is_clean(I1) and mgr.stats["desync_mismatch"] == 0


def test_snapshot_in_frame_supersedes_earlier_entry_tops():
    mgr = BookManager([I1], desync_tolerance=1)
    frame = events_of([v2([(I1, "BUY", "0.30", "5", "0.30", "")], 10),
                       book_msg(I1, {"0.20": 1}, {}, 11)])
    mgr.apply(frame)
    assert mgr.is_clean(I1) and mgr.stats["desync_mismatch"] == 0 and mgr.top_int(I1)[0] == px("0.20")


def test_hot_path_modules_do_not_import_pandas():
    code = "import sys, src.orderbook, src.clob_client; sys.exit('pandas' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO, timeout=60).returncode == 0


# =========================================================================== queues
def test_coalescing_queue_semantics():
    assert not inspect.iscoroutinefunction(CoalescingQueue.offer)
    q = CoalescingQueue(maxsize=2)
    assert q.offer("a", ("a", 1)) is None
    q.offer("b", ("b", 1))
    q.offer("a", ("a", 2))  # replaces, keeps a's first-offer position
    assert q.stats["coalesced"] == 1 and len(q) == 2
    q.offer("c", ("c", 1))  # full + new key: evicts the oldest (a)
    assert q.stats["dropped"] == 1
    assert q.get_nowait_batch(1) == [("b", 1)]
    assert q.get_nowait_batch() == [("c", 1)] and q.get_nowait_batch() == []
    assert q.stats["max_depth"] == 2


@pytest.mark.timeout(10)
async def test_coalescing_queue_get_batch_waits():
    q = CoalescingQueue()
    task = asyncio.create_task(q.get_batch())
    await asyncio.sleep(0.01)
    assert not task.done()
    for i in range(5):
        q.offer("x", i)
    q.offer("y", 0)
    assert await asyncio.wait_for(task, 1) == [4, 0]
    q.offer("z", 1)
    assert await q.get_batch(max_items=5) == [1]


def _rec(t: int, text: str = "x") -> RawRecord:
    return RawRecord(t, "ws", 1, text)


def test_drop_queue_overflow_gap_marker_and_callback():
    gaps = []
    q = DropQueue(maxsize_items=2, on_gap=lambda: gaps.append(1))
    assert q.offer(_rec(1), 1) and q.offer(_rec(2), 1)
    assert not q.offer(_rec(3), 1) and not q.offer(_rec(4), 1)  # newest dropped
    assert gaps == [1] and q.stats["dropped"] == 2
    assert q.get_nowait().t_recv_ns == 1
    assert q.offer(_rec(5), 1)
    items = [q.get_nowait() for _ in range(q.qsize())]
    assert items[0].t_recv_ns == 2
    gap = items[1]
    assert isinstance(gap, MetaEvent) and gap.kind == "gap"
    assert gap.data == {"dropped": 2, "first_t": 3, "last_t": 4}
    assert items[2].t_recv_ns == 5
    with pytest.raises(asyncio.QueueEmpty):
        DropQueue().get_nowait()


def test_drop_queue_byte_bound():
    gaps = []
    q = DropQueue(maxsize_items=100, maxsize_bytes=100, on_gap=lambda: gaps.append(1))
    assert q.offer(_rec(1), 40) and q.offer(_rec(2), 40) and q.bytes == 80
    assert not q.offer(_rec(3), 40) and gaps == [1]
    q.get_nowait()
    assert q.bytes == 40 and q.offer(_rec(4), 40)
    assert [type(x).__name__ for x in (q.get_nowait(), q.get_nowait(), q.get_nowait())] == \
        ["RawRecord", "MetaEvent", "RawRecord"]
    assert q.bytes == 0
    assert q.offer(_rec(5), 10_000)  # oversized record still admitted into an empty queue
    assert not q.offer(_rec(6), 1)
    assert q.flush_gap() and q.qsize() == 2 and not q.flush_gap()


@pytest.mark.timeout(10)
async def test_drop_queue_async_get():
    q = DropQueue()
    task = asyncio.create_task(q.get())
    await asyncio.sleep(0.01)
    q.offer(_rec(9), 3)
    assert (await asyncio.wait_for(task, 1)).t_recv_ns == 9 and q.bytes == 0


# =========================================================================== backoff / heartbeat
def test_backoff_delay_bounds():
    rng = random.Random(42)
    for attempt in (0, 1, 2, 5, 7, 8, 30, 64, 1000):
        bound = min(0.25 * 2 ** min(attempt, 64), 30.0)
        draws = [backoff_delay(attempt, rng=rng) for _ in range(10_000 if attempt < 3 else 500)]
        assert all(0.0 <= d <= bound for d in draws)
        assert max(draws) > 0.9 * bound
    assert backoff_delay(3, rng=random.Random(1)) == backoff_delay(3, rng=random.Random(1))
    assert backoff_delay(4, base_s=1.0, cap_s=3.0, rng=_One()) == 3.0


class _One(random.Random):
    def random(self) -> float:
        return 1.0


def test_heartbeat_pong_and_staleness():
    now = [100.0]
    hb = Heartbeat(interval_s=10, stale_s=30, clock=lambda: now[0])
    assert not hb.on_text('{"event_type":"book"}') and hb.on_text("PONG") and hb.on_text(b"PONG")
    now[0] = 129.0
    assert not hb.is_stale()
    now[0] = 130.5
    assert hb.is_stale() and not hb.is_stale(now=120.0)
    hb.on_text("PONG")
    assert not hb.is_stale() and hb.pongs == 3


@pytest.mark.timeout(10)
async def test_heartbeat_run_pings_until_stale():
    sent, stale = [], []

    async def send(text: str) -> None:
        sent.append(text)

    async def on_stale() -> None:
        stale.append(time.monotonic())

    hb = Heartbeat(interval_s=0.01, stale_s=0.06)
    await asyncio.wait_for(hb.run(send, on_stale, watchdog_s=0.01), 2)
    assert len(stale) == 1 and len(sent) >= 4 and set(sent) == {"PING"} and hb.pings_sent == len(sent)


# =========================================================================== REST client
def rest_client(session, **kw) -> tuple[ClobRestClient, list[float]]:
    sleeps: list[float] = []
    params = dict(session=session, sleep=sleeps.append, min_interval_s=0.0, rng=random.Random(0),
                  base_url="https://clob.example")
    params.update(kw)
    return ClobRestClient(**params), sleeps


def test_rest_get_book_and_simple_endpoints(fake_session_factory, fixture_json):
    s = fake_session_factory({
        ("GET", "/book"): fixture_json("rest_book.json"),
        ("GET", "/tick-size"): {"minimum_tick_size": 0.001},
        ("GET", "/neg-risk"): {"neg_risk": True},
        ("GET", "/markets/0xabc"): {"condition_id": "0xabc"},
        ("GET", "/clob-markets/0xabc"): {"c": "0xabc", "fd": {"r": 0.04, "e": 1, "to": True}},
        ("POST", "/midpoints"): {I1: "0.455", I2: ""},
    })
    c, _ = rest_client(s)
    assert c.get_book(I1)["asset_id"] == I1
    assert s.calls[0] == ("GET", "https://clob.example/book", {"token_id": I1}, None)
    assert c.get_tick_size(I1) == 0.001 and c.get_neg_risk(I1) is True
    assert c.get_market("0xabc") == {"condition_id": "0xabc"}
    assert c.get_clob_market("0xabc")["fd"]["r"] == 0.04
    assert s.calls[-1][1] == "https://clob.example/clob-markets/0xabc"
    assert c.get_midpoints([I1, I2]) == {I1: 0.455}
    assert s.calls[-1][3] == [{"token_id": I1}, {"token_id": I2}]


def test_rest_get_books_chunks_of_500(fake_session_factory):
    s = fake_session_factory({("POST", "/books"): lambda m, u, p, body: [{"asset_id": x["token_id"]} for x in body]})
    c, _ = rest_client(s)
    ids = [str(10**76 + i) for i in range(2 * BOOKS_CHUNK + 1)]
    out = c.get_books(ids)
    assert [len(call[3]) for call in s.calls] == [500, 500, 1]
    assert [b["asset_id"] for b in out] == ids


def test_rest_retries_429_and_5xx_then_succeeds(fake_session_factory):
    r429 = FakeResponse({"error": "slow down"}, status=429)
    r429.headers["Retry-After"] = "2"
    s = fake_session_factory({("POST", "/books"): [r429, FakeResponse({}, status=503), []]})
    c, sleeps = rest_client(s, backoff_base_s=0.5)
    assert c.get_books([I1]) == []
    assert len(s.calls) == 3 and len(sleeps) == 2
    assert sleeps[0] >= 2.0  # Retry-After honoured
    assert 0.0 <= sleeps[1] <= 1.0  # full jitter: U(0,1) * 0.5 * 2**1
    assert c.stats["retries"] == 2


def test_rest_4xx_raises_without_retry(fake_session_factory):
    s = fake_session_factory({("GET", "/book"): FakeResponse({"error": "bad"}, status=400)})
    c, sleeps = rest_client(s)
    with pytest.raises(ClobHTTPError) as ei:
        c.get_book(I1)
    assert ei.value.status == 400 and len(s.calls) == 1 and not sleeps


def test_rest_retries_exhausted(fake_session_factory):
    s = fake_session_factory({("GET", "/book"): FakeResponse({}, status=502)})
    c, sleeps = rest_client(s, max_retries=2)
    with pytest.raises(ClobHTTPError):
        c.get_book(I1)
    assert len(s.calls) == 3 and len(sleeps) == 2
    s2 = fake_session_factory({("GET", "/book"): [requests.ConnectionError("reset"), {"asset_id": I1}]})
    c2, sleeps2 = rest_client(s2)
    assert c2.get_book(I1) == {"asset_id": I1} and len(sleeps2) == 1
    s3 = fake_session_factory({("GET", "/book"): requests.Timeout("slow")})
    c3, _ = rest_client(s3, max_retries=1)
    with pytest.raises(requests.Timeout):
        c3.get_book(I1)


def test_rest_prices_history(fake_session_factory):
    s = fake_session_factory({("GET", "/prices-history"): {"history": [{"t": 20, "p": 0.5}, {"t": 10, "p": 0.455}]}})
    c, _ = rest_client(s)
    assert c.get_prices_history(I1, start_ts=10, end_ts=20, fidelity=5) == [(10, 0.455), (20, 0.5)]
    assert s.calls[-1][2] == {"market": I1, "fidelity": 5, "startTs": 10, "endTs": 20}
    c.get_prices_history(I1, interval="1d")
    assert s.calls[-1][2] == {"market": I1, "fidelity": 1, "interval": "1d"}
    with pytest.raises(ValueError):
        c.get_prices_history(I1, interval="1d", start_ts=1)
    with pytest.raises(ValueError):
        c.get_prices_history(I1)


def test_rest_throttle_and_default_session():
    sleeps: list[float] = []
    s = type("S", (), {"request": lambda self, *a, **k: FakeResponse({"ok": 1})})()
    c = ClobRestClient(session=s, min_interval_s=0.5, sleep=sleeps.append, clock=lambda: 0.0)
    c.get_market("0x1")
    c.get_market("0x2")
    assert sleeps == [0.5]
    default = ClobRestClient()
    assert default.session.trust_env is True and default.session.verify is True  # env proxy + CA, TLS verified
    default.close()


# =========================================================================== WebSocket feed helpers
class FakeRest:
    """Thread-safe ``get_books`` stand-in; ``responder(call_index, ids)`` builds the reply."""

    def __init__(self, responder: Callable[[int, list[str]], list[dict]] | None = None):
        self.responder = responder or (lambda i, ids: [])
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def get_books(self, ids):
        with self._lock:
            idx = len(self.calls)
            self.calls.append(list(ids))
        return self.responder(idx, list(ids))


class FakeMarketServer:
    """In-process market-channel server. ``script(server, ws, conn_index)`` drives each
    connection; a reader task records every inbound frame and answers PING with PONG."""

    def __init__(self, script=None, *, pong: bool = True):
        self.script = script
        self.pong = pong
        self.conns: list[Any] = []
        self.inbox: list[list[str]] = []
        self.ping_times: list[float] = []
        self.errors: list[Exception] = []
        self.url = ""

    async def handler(self, ws) -> None:
        idx = len(self.conns)
        self.conns.append(ws)
        self.inbox.append([])

        async def reader() -> None:
            with contextlib.suppress(ConnectionClosed):
                async for msg in ws:
                    self.inbox[idx].append(msg)
                    if msg == "PING":
                        self.ping_times.append(time.monotonic())
                        if self.pong:
                            await ws.send("PONG")

        # The handler lives exactly as long as the connection: when the reader sees the
        # close, the script is cancelled (otherwise serve() would wait for it on exit).
        tasks = [asyncio.create_task(reader())]
        if self.script is not None:
            tasks.append(asyncio.create_task(self.script(self, ws, idx)))
        try:
            await tasks[0]
        finally:
            for t in tasks:
                t.cancel()
            for res in await asyncio.gather(*tasks, return_exceptions=True):
                if isinstance(res, Exception) and not isinstance(res, ConnectionClosed):
                    self.errors.append(res)

    def json_frames(self, idx: int) -> list[dict]:
        return [json.loads(m) for m in self.inbox[idx] if m != "PING"]


@contextlib.asynccontextmanager
async def fake_server(script=None, *, pong: bool = True):
    srv = FakeMarketServer(script, pong=pong)
    async with serve(srv.handler, "127.0.0.1", 0, compression=None) as server:
        srv.url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        yield srv
    assert not srv.errors, srv.errors


def ws_feed(url: str, ids: list[str], **kw: Any) -> MarketDataFeed:
    params: dict[str, Any] = dict(url=url, proxy=None, backoff_base_s=0.001, backoff_cap_s=0.005,
                                  rng=random.Random(7), heartbeat_interval_s=0.05, stale_after_s=5.0,
                                  watchdog_s=0.05, open_timeout_s=2.0, resync_min_interval_s=0.0,
                                  lag_monitor_interval_s=0.01)
    params.update(kw)
    return MarketDataFeed(ids, **params)


@contextlib.asynccontextmanager
async def running(feed):
    task = asyncio.create_task(feed.run())
    try:
        yield task
    finally:
        await feed.stop()
        assert task.done() and not task.cancelled()
        task.result()  # surface any exception from run()


async def until(pred: Callable[[], bool], timeout: float = 5.0, what: str = "condition") -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.005)


async def hold(server, ws, idx) -> None:
    await asyncio.sleep(3600)


def drain(q: DropQueue) -> list:
    return [q.get_nowait() for _ in range(q.qsize())]


# =========================================================================== WebSocket feed tests
def test_subscribe_frame_shape():
    feed = MarketDataFeed([I2, I1, I2], rest=None, extra_subscribe_fields={"level": 2})
    assert feed.asset_ids == (I2, I1)
    assert feed.subscribe_frame() == {"type": "market", "assets_ids": [I1, I2], "custom_feature_enabled": True,
                                      "initial_dump": True, "level": 2}


@pytest.mark.timeout(20)
async def test_ws_subscribe_ping_pong_book_and_deltas():
    async def script(server, ws, idx):
        await until(lambda: server.inbox[idx], what="subscribe")
        await ws.send(json.dumps([book_msg(I1, {"0.40": 10, "0.38": 5}, {"0.60": 7}, 1000),
                                  book_msg(I2, {"0.20": 1}, {"0.80": 1}, 1000)]))
        await ws.send(json.dumps(v2([(I1, "BUY", "0.41", "3", "0.41", "0.60"),
                                     (I1, "SELL", "0.60", "0", "0.41", "0.62"),
                                     (I1, "SELL", "0.62", "9", "0.41", "0.62")], 1001)))
        await ws.send(json.dumps({"event_type": "price_change", "asset_id": I2, "market": "0xm",
                                  "timestamp": "1002", "changes": [{"price": "0.21", "side": "BUY", "size": "4"}]}))
        await ws.send(json.dumps({"event_type": "tick_size_change", "asset_id": I1, "market": "0xm",
                                  "old_tick_size": "0.01", "new_tick_size": "0.001", "timestamp": "1003"}))
        await ws.send(json.dumps({"event_type": "last_trade_price", "asset_id": I1, "market": "0xm",
                                  "price": "0.41", "size": "3", "side": "BUY", "timestamp": "1004"}))
        await ws.send("INVALID OPERATION")
        await hold(server, ws, idx)

    async with fake_server(script) as srv:
        feed = ws_feed(srv.url, [I2, I1], heartbeat_interval_s=0.02)
        upd = feed.subscribe_updates()
        raw = feed.subscribe_raw()
        control: list = []
        feed.add_listener(control.append)
        feed.add_listener(lambda ev: 1 / 0)  # a broken listener must not hurt the feed
        async with running(feed):
            await until(lambda: feed.stats.parse.non_json == 1 and feed.stats.pongs >= 2, what="frames")
            sub = srv.json_frames(0)[0]
            assert sub == {"type": "market", "assets_ids": [I1, I2], "custom_feature_enabled": True,
                           "initial_dump": True}
            assert srv.inbox[0].count("PING") >= 2
            b = feed.books.books[I1]
            assert b.bids == {px("0.41"): 3, px("0.40"): 10, px("0.38"): 5} and b.asks == {px("0.62"): 9}
            assert feed.books.books[I2].bids == {px("0.21"): 4, px("0.20"): 1}
            assert feed.books.all_clean() and feed.books.stats["desync_mismatch"] == 0
            batch = upd.get_nowait_batch()
            assert {a for a, _ in batch} == {I1, I2} and all(isinstance(t, int) for _, t in batch)
            p = feed.stats.parse
            assert (p.non_json, p.malformed, sum(p.unknown.values())) == (1, 0, 0)
            assert feed.stats.errors >= 1  # the failing listener was counted, the feed lives on
            kinds = [type(e).__name__ for e in control]
            assert "TickSizeEvent" in kinds and "LastTradeEvent" not in kinds
            assert isinstance(control[0], MetaEvent) and control[0].kind == "connect"
            srcs = [r.src if isinstance(r, RawRecord) else r.kind for r in drain(raw)]
            assert srcs[0] == "connect" and srcs.count("ws") == 6  # every frame, PONGs excluded
            assert feed.stats.max_loop_lag_s < 1.0


@pytest.mark.timeout(20)
async def test_ws_reconnect_resubscribes_and_stays_dirty_until_snapshot():
    gate = threading.Event()

    def responder(i, ids):
        if i == 1:
            gate.wait(10)  # second connection's resync is held back
        return [book_msg(a, {"0.45": 1}, {"0.55": 1}, 5000 + i) for a in ids]

    rest = FakeRest(responder)
    abort = asyncio.Event()

    async def script(server, ws, idx):
        await until(lambda: server.inbox[idx], what="subscribe")
        if idx == 0:
            await abort.wait()
            ws.transport.abort()
            return
        await ws.send(json.dumps(v2([(I1, "BUY", "0.44", "2", "0.45", "0.55")], 6000)))
        await hold(server, ws, idx)

    sleeps: list[float] = []

    async def rec_sleep(d: float) -> None:
        sleeps.append(d)
        await asyncio.sleep(d)

    try:
        async with fake_server(script) as srv:
            feed = ws_feed(srv.url, [I1, I2], rest=rest, sleep=rec_sleep, rng=_One())
            raw = feed.subscribe_raw()
            metas: list = []
            feed.add_listener(lambda e: metas.append(e.kind) if isinstance(e, MetaEvent) else None)
            async with running(feed):
                await until(lambda: feed.books.all_clean() and feed.stats.resyncs == 1, what="first resync")
                abort.set()
                await until(lambda: len(rest.calls) == 2 and feed.books.books[I1].bids.get(px("0.44")) == 2,
                            what="reconnect + resync call + delta")
                assert not feed.books.is_clean(I1) and not feed.books.is_clean(I2)
                gate.set()
                await until(lambda: feed.books.all_clean(), what="snapshot after reconnect")
                assert feed.books.books[I1].bids == {px("0.45"): 1, px("0.44"): 2}  # delta replayed
                assert len(srv.conns) == 2 and srv.inbox[0][0] == srv.inbox[1][0]  # identical subscribe
                assert sorted(rest.calls[0]) == sorted(rest.calls[1]) == sorted([I1, I2])
                assert feed.stats.connects == 2 and feed.stats.reconnects == 1 and feed.conn_seq == 2
                assert sleeps == [0.001]  # attempt 0 after a connection that delivered data
                assert metas[:4] == ["connect", "resync", "disconnect", "connect"]
                recs = drain(raw)
                assert [r.conn for r in recs if isinstance(r, RawRecord) and r.src == "rest_book"] == [1, 1, 2, 2]
    finally:
        gate.set()


@pytest.mark.timeout(20)
async def test_ws_stale_resync_across_reconnect_is_discarded():
    gate_old, gate_new = threading.Event(), threading.Event()
    old = book_msg(I1, {"0.40": 100}, {"0.60": 100}, 1500)
    new = book_msg(I1, {"0.45": 10}, {"0.55": 10}, 3000)

    def responder(i, ids):
        if i == 0:
            gate_old.wait(10)
            return [old]
        gate_new.wait(10)
        return [new]

    rest = FakeRest(responder)
    abort = asyncio.Event()

    async def script(server, ws, idx):
        await until(lambda: server.inbox[idx], what="subscribe")
        if idx == 0:
            await abort.wait()
            ws.transport.abort()
            return
        await ws.send(json.dumps(v2([(I1, "BUY", "0.44", "7", "0.45", "0.55")], 3500)))
        await hold(server, ws, idx)

    try:
        async with fake_server(script) as srv:
            # A long rate limit: the discarded call must not delay the fresh resync.
            feed = ws_feed(srv.url, [I1], rest=rest, resync_min_interval_s=30.0)
            raw = feed.subscribe_raw()
            async with running(feed):
                await until(lambda: len(rest.calls) == 1, what="resync issued on connection 1")
                abort.set()
                await until(lambda: feed.conn_seq == 2 and feed.books.books[I1].bids.get(px("0.44")) == 7,
                            what="reconnect + delta on connection 2")
                gate_old.set()  # the connection-1 snapshot completes only now
                await until(lambda: feed.stats.resync_discarded == 1, what="stale resync discarded")
                book = feed.books.books[I1]
                assert not feed.books.is_clean(I1) and px("0.40") not in book.bids
                await until(lambda: len(rest.calls) == 2, what="fresh resync on connection 2")
                gate_new.set()
                await until(lambda: feed.books.is_clean(I1), what="healed by the fresh snapshot")
                assert book.bids == {px("0.45"): 10, px("0.44"): 7} and book.asks == {px("0.55"): 10}
                rest_recs = [r for r in drain(raw) if isinstance(r, RawRecord) and r.src == "rest_book"]
                assert [(r.conn, r.msg["timestamp"]) for r in rest_recs] == [(2, "3000")]
    finally:
        gate_old.set()
        gate_new.set()


@pytest.mark.timeout(20)
async def test_ws_attempt_counter_resets_only_after_data():
    async def script(server, ws, idx):
        await until(lambda: server.inbox[idx], what="subscribe")
        if idx < 3:
            await ws.close()  # accept-then-close: no data, backoff must keep growing
            return
        if idx == 3:
            await ws.send(json.dumps(book_msg(I1, {"0.40": 1}, {"0.60": 1}, 10)))
            await asyncio.sleep(0.05)
            ws.transport.abort()
            return
        await hold(server, ws, idx)

    sleeps: list[float] = []

    async def rec_sleep(d: float) -> None:
        sleeps.append(d)
        await asyncio.sleep(0)

    async with fake_server(script) as srv:
        feed = ws_feed(srv.url, [I1], rng=_One(), backoff_base_s=0.001, backoff_cap_s=1.0, sleep=rec_sleep)
        async with running(feed):
            await until(lambda: len(srv.conns) == 5, what="five connections")
            assert sleeps == [0.001, 0.002, 0.004, 0.001]


@pytest.mark.timeout(20)
async def test_ws_stale_heartbeat_closes_and_reconnects():
    async with fake_server(hold, pong=False) as srv:
        feed = ws_feed(srv.url, [I1], heartbeat_interval_s=0.05, stale_after_s=0.2, watchdog_s=0.05)
        async with running(feed):
            await until(lambda: feed.stats.stale_closes >= 1 and len(srv.inbox) >= 2 and srv.inbox[1],
                        what="stale reconnect")
            assert feed.stats.pings >= 2 and feed.stats.pongs == 0
            assert srv.inbox[1][0] == srv.inbox[0][0]


@pytest.mark.timeout(60)
async def test_ws_flood_with_slow_consumer_keeps_heartbeat_alive():
    n_frames, ts0, rng = 5_000, 10_000, random.Random(3)
    bids = {px("0.40"): 100.0, px("0.39"): 50.0}
    asks = {px("0.60"): 100.0, px("0.61"): 50.0}
    frames, states = [], []
    for i in range(n_frames):
        entries = []
        for _ in range(3):
            side = rng.choice((BID, ASK))
            book = bids if side == BID else asks
            price = rng.randrange(380_000, 450_001, 10_000) if side == BID else rng.randrange(550_000, 620_001, 10_000)
            size = 0.0 if rng.random() < 0.3 else float(rng.randint(1, 500))
            if size:
                book[price] = size
            else:
                book.pop(price, None)
            entries.append((I1, "BUY" if side == BID else "SELL", f"{price / PRICE_SCALE:.2f}", str(size),
                            fmt(max(bids) if bids else None, "0"), fmt(min(asks) if asks else None, "1")))
        frames.append(json.dumps(v2(entries, ts0 + 1 + i)))
        states.append((dict(bids), dict(asks)))
    progress = [-1]  # index of the last frame the server has sent (read by the REST thread)
    initial = ({px("0.40"): 100.0, px("0.39"): 50.0}, {px("0.60"): 100.0, px("0.61"): 50.0})

    def responder(i, ids):
        k = progress[0]
        b, a = states[k] if k >= 0 else initial
        return [{"asset_id": I1, "market": "0xm", "timestamp": str(ts0 + 1 + k),
                 "bids": [{"price": str(p / PRICE_SCALE), "size": str(q)} for p, q in b.items()],
                 "asks": [{"price": str(p / PRICE_SCALE), "size": str(q)} for p, q in a.items()]}]

    rest = FakeRest(responder)
    flood = {}

    async def script(server, ws, idx):
        await until(lambda: server.inbox[idx], what="subscribe")
        await ws.send(json.dumps(book_msg(I1, {"0.40": 100.0, "0.39": 50.0}, {"0.60": 100.0, "0.61": 50.0}, ts0)))
        flood["start"] = time.monotonic()
        for i, f in enumerate(frames):
            await ws.send(f)
            progress[0] = i
        await hold(server, ws, idx)

    async with fake_server(script) as srv:
        feed = ws_feed(srv.url, [I1], rest=rest, heartbeat_interval_s=0.02, stale_after_s=5.0)
        upd = feed.subscribe_updates()
        raw = feed.subscribe_raw(maxsize_items=20)  # nobody drains it: overflows immediately
        got: list = []

        async def slow_consumer() -> None:
            while True:
                got.extend(await upd.get_batch())
                await asyncio.sleep(0.05)

        consumer = asyncio.create_task(slow_consumer())
        try:
            async with running(feed):
                await until(lambda: feed.stats.data_frames >= n_frames + 1, timeout=45, what="flood processed")
                t_end = time.monotonic()
                book = feed.books.books[I1]
                assert (book.bids, book.asks) == states[-1]
                assert feed.books.is_clean(I1) and feed.books.stats["desync"] == 0
                assert upd.stats["coalesced"] > 0 and len(got) < n_frames
                pings = [t for t in srv.ping_times if flood["start"] <= t <= t_end]
                gaps = np.diff([flood["start"], *pings, t_end])
                assert len(pings) >= 3 and gaps.max() < 0.5, (len(pings), gaps.max(), t_end - flood["start"])
                assert feed.stats.reader_yields >= n_frames // 50
                assert raw.stats["gaps"] == 1 and raw.stats["dropped"] > 0
                assert feed.stats.resync_reasons["raw_gap"] == 1  # the gap requested a resync
                await until(lambda: feed.stats.resyncs >= 1, what="resync applied")
        finally:
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        markers = [r for r in drain(raw) if isinstance(r, MetaEvent) and r.kind == "gap"]
        assert len(markers) == 1 and markers[0].data["dropped"] == raw.stats["dropped"]


class BufferedFloodWS:
    """Injected connection whose iterator hands out a pre-buffered flood WITHOUT ever
    suspending: the worst case in which only the reader's explicit yields let the
    heartbeat and the consumers run."""

    def __init__(self, frames: list[str]):
        self.frames = frames
        self.sent: list[tuple[float, str]] = []
        self.window = (0.0, 0.0)
        self.close_code, self.close_reason = None, ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def send(self, text: str) -> None:
        self.sent.append((time.monotonic(), text))

    async def close(self) -> None:
        pass

    async def _frames(self):
        t0 = time.monotonic()
        for f in self.frames:
            yield f
        self.window = (t0, time.monotonic())
        await asyncio.Event().wait()

    def __aiter__(self):
        return self._frames()


@pytest.mark.timeout(30)
async def test_reader_yields_during_never_suspending_flood():
    frames = [json.dumps(book_msg(I1, {"0.40": 1}, {"0.60": 1}, 1))]
    frames += [json.dumps(v2([(I1, "BUY", f"0.4{(i + k) % 10}", str(1 + (i + k) % 7), None, None) for k in range(5)],
                             2 + i)) for i in range(5_000)]  # moves the best bid: notifications
    conn = BufferedFloodWS(frames)
    feed = ws_feed("ws://injected", [I1], connect=lambda url, **kw: conn, heartbeat_interval_s=0.005)
    upd = feed.subscribe_updates()
    batches: list[float] = []

    async def slow_consumer() -> None:
        while True:
            await upd.get_batch()
            batches.append(time.monotonic())
            await asyncio.sleep(0.01)

    consumer = asyncio.create_task(slow_consumer())
    try:
        async with running(feed):
            await until(lambda: conn.window[1] > 0, timeout=20, what="flood consumed")
            t0, t1 = conn.window
            pings = [t for t, text in conn.sent if text == "PING" and t0 <= t <= t1]
            gaps = np.diff([t0, *pings, t1])
            assert len(pings) >= 2 and gaps.max() < 0.25, (len(pings), gaps.max(), t1 - t0)
            assert sum(t0 <= t <= t1 for t in batches) >= 2  # the slow consumer was not starved
            assert upd.stats["coalesced"] > 0 and feed.stats.data_frames == len(frames)
            assert feed.stats.max_loop_lag_s < 0.25
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)


@pytest.mark.timeout(20)
async def test_ws_add_and_remove_assets():
    async with fake_server(hold) as srv:
        feed = ws_feed(srv.url, [I1], rest=FakeRest())
        async with running(feed):
            await until(lambda: srv.inbox and srv.inbox[0], what="subscribe")
            assert await feed.add_assets([I2, I1, I2]) == [I2]
            assert await feed.add_assets([I1, I2]) == []  # duplicates are never re-subscribed
            assert await feed.remove_assets([I1, "unknown"]) == [I1]
            assert await feed.remove_assets([I1]) == []
            await until(lambda: len(srv.json_frames(0)) == 3, what="dynamic frames")
            _, sub, unsub = srv.json_frames(0)
            assert sub == {"operation": "subscribe", "assets_ids": [I2], "custom_feature_enabled": True}
            assert unsub == {"operation": "unsubscribe", "assets_ids": [I1]}
            assert feed.asset_ids == (I2,) and I1 not in feed.books and I2 in feed.books
            assert feed.subscribe_frame()["assets_ids"] == [I2]


@pytest.mark.timeout(20)
async def test_ws_stop_leaves_no_tasks_even_during_backoff():
    async with fake_server(hold) as srv:
        feed = ws_feed(srv.url, [I1], rest=FakeRest())
        async with running(feed):
            await until(lambda: feed.stats.connects == 1, what="connect")
        assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("clob:")]
        assert feed.stats.disconnects == 1
    sock = __import__("socket").socket()
    sock.bind(("127.0.0.1", 0))
    url = f"ws://127.0.0.1:{sock.getsockname()[1]}"
    sock.close()  # nothing listens: every connect fails, then a long backoff sleep
    feed = ws_feed(url, [I1], backoff_base_s=30.0, backoff_cap_s=30.0, rng=_One())
    task = asyncio.create_task(feed.run())
    await until(lambda: feed.stats.reconnects == 1, what="first failure")
    t0 = time.monotonic()
    await feed.stop()
    assert time.monotonic() - t0 < 2.0 and task.done() and task.result() is None
    assert feed.stats.connects == 0 and feed.stats.last_error
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("clob:")]
    early = ws_feed(url, [I1])
    task = asyncio.create_task(early.run())
    await early.stop()  # before run() was even scheduled: stop is sticky
    await asyncio.wait_for(task, 1)
    assert early.stats.connects == 0 and early.stats.reconnects == 0


def test_make_feed_modes():
    rest = FakeRest()
    assert isinstance(make_feed("ws", [I1], rest=None), MarketDataFeed)
    assert isinstance(make_feed("poll", [I1], rest=rest, interval_s=0.5), RestPollingFeed)
    with pytest.raises(ValueError):
        make_feed("poll", [I1], rest=None)
    with pytest.raises(ValueError):
        make_feed("auto", [I1], rest=rest)


# =========================================================================== polling feed
@pytest.mark.timeout(20)
async def test_poll_feed_emits_only_changes_and_heals_after_errors(fixture_json):
    books = fixture_json("rest_books.json")
    changed_b = json.loads(json.dumps(books))
    changed_b[1]["bids"][1]["size"] = "301"  # size at the best bid
    changed_b[1]["hash"] = "h-new"
    changed_b[1]["timestamp"] = "1759363200999"
    same_levels_new_hash = json.loads(json.dumps(changed_b))
    same_levels_new_hash[0]["hash"] = "h-other"
    replies: list[Any] = [books, books, changed_b, requests.ConnectionError("down"), changed_b,
                          same_levels_new_hash]

    def responder(i, ids):
        r = replies[i]
        if isinstance(r, BaseException):
            raise r
        return r

    feed = RestPollingFeed([I1, I2], rest=FakeRest(responder))
    upd = feed.subscribe_updates()
    raw = feed.subscribe_raw()
    metas: list = []
    feed.add_listener(lambda e: metas.append(e.kind) if isinstance(e, MetaEvent) else None)

    assert await feed.poll_once() == {I1, I2}
    assert feed.books.all_clean() and feed.books.top(I2) == (0.53, 0.54)
    assert [r.src for r in drain(raw) if isinstance(r, RawRecord)] == ["poll", "poll"]
    assert await feed.poll_once() == set() and not drain(raw)  # unchanged
    assert await feed.poll_once() == {I2}  # only the changed book is emitted
    recs = drain(raw)
    assert [r.msg["asset_id"] for r in recs] == [I2] and recs[0].conn == 0
    assert await feed.poll_once() == set()  # failure: books dirty until the next success
    assert not feed.books.is_clean(I1) and not feed.books.is_clean(I2) and feed.stats.poll_failures == 1
    assert await feed.poll_once() == {I1, I2}  # re-emitted (dirty) even though levels match
    assert feed.books.all_clean()
    assert await feed.poll_once() == set()  # new hash but identical levels: not a change
    assert metas == ["connect", "disconnect", "connect"]
    assert {a for a, _ in upd.get_nowait_batch()} == {I1, I2}


@pytest.mark.timeout(20)
async def test_poll_feed_run_tick_change_gap_and_stop(fixture_json):
    base = fixture_json("rest_book.json")
    ticks = ["0.01", "0.01", "0.001"]

    def responder(i, ids):
        b = dict(base)
        b["tick_size"] = ticks[min(i, 2)]  # the tick changes while the levels stay the same
        b["timestamp"] = str(int(base["timestamp"]) + i)
        b["hash"] = f"h{min(i, 3)}"
        if i >= 3:
            b["bids"] = [{"price": "0.461", "size": "1"}]
        return [b]

    sleeps: list[float] = []

    async def rec_sleep(d: float) -> None:
        sleeps.append(d)
        await asyncio.sleep(0)

    feed = RestPollingFeed([I1], rest=FakeRest(responder), interval_s=0.5, jitter_s=0.1, sleep=rec_sleep,
                           rng=random.Random(1))
    raw = feed.subscribe_raw(maxsize_items=1)
    control: list = []
    feed.add_listener(control.append)
    async with running(feed):
        await until(lambda: feed.stats.polls >= 5, what="polls")
    assert all(0.0 <= d <= 0.6 for d in sleeps)
    ticks_seen = [e for e in control if isinstance(e, TickSizeEvent)]
    assert len(ticks_seen) == 1 and (ticks_seen[0].old_tick, ticks_seen[0].new_tick) == (10_000, 1_000)
    assert feed.books.books[I1].tick == 1_000 and feed.books.top(I1)[0] == 0.461
    assert raw.stats["gaps"] == 1 and feed.stats.resync_reasons["raw_gap"] == 1
    assert [e.kind for e in control if isinstance(e, MetaEvent)] == ["connect", "disconnect"]
    assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("clob:")]
