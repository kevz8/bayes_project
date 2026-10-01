"""Polymarket CLOB market data: wire parsing, WebSocket and REST-polling feeds, REST client.

Data flow. The socket reader parses each frame and applies it to the shared
``BookManager`` right away, at a cost of O(levels in the frame). It then fans out
without ever awaiting a consumer:

* ``CoalescingQueue``: asset-level "re-read this book" notifications for the strategy.
  Conflated per asset, so a slow consumer sees the latest state, never a backlog.
* ``DropQueue``: the lossless raw stream for the recorder, bounded by items and by
  bytes. On overflow it drops the newest record, writes a gap marker, and triggers a
  REST resync so the recording heals itself.

Reliability rules:

* A text ``PING`` goes out every 10 s, and the connection counts as stale after 30 s
  without a ``PONG``.
* Reconnects use full-jitter backoff ``U(0,1)·min(0.25·2^a, 30)``, matching the official
  py-sdk. The attempt counter resets only after real data arrives, so a server that
  accepts and then closes cannot cause a hot reconnect loop.
* Every (re)connect resends the full subscribe frame, marks every book dirty, and
  requests a REST ``POST /books`` resync. Each resync is tagged with the connection
  sequence number it was issued on. A late result from an earlier connection is
  discarded, because it cannot contain the updates missed during the outage.
* The reader yields to the event loop every 50 frames or 5 ms, so a flood of buffered
  frames cannot starve the heartbeat or the consumers.

This container's egress proxy does not support WebSocket upgrades, so real data is
recorded here with ``RestPollingFeed`` (``POST /books`` about once a second).
``MarketDataFeed`` is the low-latency path for users' own machines.

Hot-path module: no pandas import.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Hashable, Iterable, Mapping, Sequence

import requests
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException

from .events import (
    ASK,
    BID,
    CONTROL_EVENT_TYPES,
    BestBidAskEvent,
    BookEvent,
    Event,
    LastTradeEvent,
    LevelChange,
    MarketResolvedEvent,
    MetaEvent,
    NewMarketEvent,
    PriceChangeEvent,
    TickSizeEvent,
    opt_float,
    opt_px,
    px_to_int,
)
from .orderbook import BookManager

try:  # optional fast path; the stdlib parser is the reference
    import orjson

    _loads: Callable[[str | bytes], Any] = orjson.loads
except ImportError:  # pragma: no cover - depends on the environment
    _loads = json.loads

logger = logging.getLogger(__name__)

WS_MARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB_REST_URL = "https://clob.polymarket.com"
HEARTBEAT_INTERVAL_S = 10.0
HEARTBEAT_STALE_S = 30.0
WATCHDOG_INTERVAL_S = 5.0
BACKOFF_BASE_S = 0.25
BACKOFF_CAP_S = 30.0
BOOKS_CHUNK = 500  # POST /books accepts at most 500 token ids per call

_NETWORK_ERRORS = (WebSocketException, OSError, TimeoutError)


# =========================================================================== parsing
@dataclass(slots=True)
class ParseStats:
    frames: int = 0
    events: int = 0
    pongs: int = 0
    non_json: int = 0
    malformed: int = 0
    unknown: Counter = field(default_factory=Counter)  # event_type -> count


def _ms(value: Any) -> int:
    """Epoch-ms timestamp (sent as a decimal string) -> int; raises if absent."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if value is None or value == "":
        raise ValueError("missing timestamp")
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _opt_ms(value: Any) -> int:
    try:
        return _ms(value)
    except (TypeError, ValueError):
        return 0


def _side(value: Any) -> int:
    s = str(value).upper()
    if s == "BUY":
        return BID
    if s == "SELL":
        return ASK
    raise ValueError(f"unknown side {value!r}")


def _asset(value: Any) -> str:
    """Token ids are 77-78 digit decimal strings; they stay ``str`` (never int/float)."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"bad asset id {value!r}")
    return value


def _levels(arr: Any) -> list[tuple[int, float]]:
    return [(px_to_int(lv["price"]), float(lv["size"])) for lv in arr or ()]


def _book_event(m: Mapping[str, Any], t_recv_ns: int, source: str) -> BookEvent:
    # "buys"/"sells" are the field names in older docs; captured payloads use bids/asks.
    return BookEvent(
        asset_id=_asset(m["asset_id"]),
        market=str(m.get("market") or ""),
        bids=_levels(m.get("bids", m.get("buys"))),
        asks=_levels(m.get("asks", m.get("sells"))),
        ts_ms=_ms(m.get("timestamp")),
        t_recv_ns=t_recv_ns,
        source=source,
        tick_size=opt_px(m.get("tick_size")),
        hash=m.get("hash") or None,
    )


def _on_book(m: Mapping[str, Any], t: int) -> list[Event]:
    return [_book_event(m, t, "ws")]


def _on_price_change(m: Mapping[str, Any], t: int) -> list[Event]:
    market, ts = str(m.get("market") or ""), _ms(m.get("timestamp"))
    if "price_changes" in m:  # v2 schema (since 2025-09-15): per-entry asset id and server top
        changes = [
            LevelChange(_asset(e["asset_id"]), _side(e["side"]), px_to_int(e["price"]), opt_float(e.get("size")),
                        opt_px(e.get("best_bid")), opt_px(e.get("best_ask")))
            for e in m["price_changes"]
        ]
        return [PriceChangeEvent(market, changes, ts, t, "v2")]
    if "changes" in m:  # legacy schema: one asset per message, no server top
        aid = _asset(m["asset_id"])
        changes = [LevelChange(aid, _side(e["side"]), px_to_int(e["price"]), opt_float(e.get("size")))
                   for e in m["changes"]]
        return [PriceChangeEvent(market, changes, ts, t, "legacy")]
    raise KeyError("price_change without price_changes/changes")


def _on_best_bid_ask(m: Mapping[str, Any], t: int) -> list[Event]:
    return [BestBidAskEvent(_asset(m["asset_id"]), str(m.get("market") or ""), opt_px(m.get("best_bid")),
                            opt_px(m.get("best_ask")), _ms(m.get("timestamp")), t)]


def _on_last_trade(m: Mapping[str, Any], t: int) -> list[Event]:
    return [LastTradeEvent(_asset(m["asset_id"]), str(m.get("market") or ""), float(m["price"]),
                           opt_float(m.get("size")), m.get("side") or None, opt_float(m.get("fee_rate_bps")),
                           _ms(m.get("timestamp")), t)]


def _on_tick_size(m: Mapping[str, Any], t: int) -> list[Event]:
    return [TickSizeEvent(_asset(m["asset_id"]), str(m.get("market") or ""), opt_px(m.get("old_tick_size")),
                          px_to_int(m["new_tick_size"]), _ms(m.get("timestamp")), t)]


def _ids(value: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in value or ())


def _on_market_resolved(m: Mapping[str, Any], t: int) -> list[Event]:
    win = m.get("winning_asset_id")
    return [MarketResolvedEvent(str(m.get("market") or ""), _ids(m.get("assets_ids")), str(win) if win else None,
                                m.get("winning_outcome") or None, _opt_ms(m.get("timestamp")), t)]


def _on_new_market(m: Mapping[str, Any], t: int) -> list[Event]:
    em = m.get("event_message") or {}
    eid = em.get("id")
    return [NewMarketEvent(str(m.get("market") or m.get("condition_id") or ""), _ids(m.get("assets_ids")),
                           em.get("slug") or None, str(eid) if eid not in (None, "") else None,
                           m.get("question") or None, _opt_ms(m.get("timestamp")), t)]


_HANDLERS: dict[str, Callable[[Mapping[str, Any], int], list[Event]]] = {
    "book": _on_book,
    "price_change": _on_price_change,
    "best_bid_ask": _on_best_bid_ask,
    "last_trade_price": _on_last_trade,
    "tick_size_change": _on_tick_size,
    "market_resolved": _on_market_resolved,
    "new_market": _on_new_market,
}


def parse_message(obj: Mapping[str, Any], t_recv_ns: int, stats: ParseStats | None = None) -> list[Event]:
    """One decoded server message -> normalised events (unknown keys are ignored)."""
    et = obj.get("event_type")
    handler = _HANDLERS.get(et) if isinstance(et, str) else None
    if handler is None:
        if stats is not None:
            stats.unknown[str(et)] += 1
        return []
    try:
        events = handler(obj, t_recv_ns)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if stats is not None:
            stats.malformed += 1
        logger.debug("malformed %s message: %r", et, exc)
        return []
    if stats is not None:
        stats.events += len(events)
    return events


def parse_frame(raw: str | bytes, t_recv_ns: int, stats: ParseStats | None = None) -> list[Event]:
    """One WebSocket frame -> events. A frame is a JSON object or a JSON array of them.

    ``PONG`` and non-JSON server texts (e.g. ``INVALID OPERATION``) yield ``[]`` and
    never raise; malformed or unknown messages are counted and dropped individually.
    """
    if stats is not None:
        stats.frames += 1
    if raw == "PONG" or raw == b"PONG":
        if stats is not None:
            stats.pongs += 1
        return []
    try:
        obj = _loads(raw)
    except ValueError:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        if stats is not None:
            stats.non_json += 1
        return []
    if isinstance(obj, dict):
        return parse_message(obj, t_recv_ns, stats)
    if not isinstance(obj, list):
        if stats is not None:
            stats.malformed += 1
        return []
    out: list[Event] = []
    for m in obj:
        if isinstance(m, dict):
            out.extend(parse_message(m, t_recv_ns, stats))
        elif stats is not None:
            stats.malformed += 1
    return out


def parse_rest_book(obj: Mapping[str, Any], t_recv_ns: int, source: str = "rest") -> BookEvent:
    """REST ``/book`` (or one ``/books`` element) -> ``BookEvent``. Raises if malformed."""
    return _book_event(obj, t_recv_ns, source)


# =========================================================================== backoff / heartbeat
def backoff_delay(attempt: int, *, base_s: float = BACKOFF_BASE_S, cap_s: float = BACKOFF_CAP_S,
                  rng: random.Random | None = None) -> float:
    """Full-jitter exponential backoff ``U(0,1)·min(base·2^attempt, cap)`` (as the py-sdk)."""
    u = (rng or random).random()
    return u * min(base_s * 2.0 ** min(max(attempt, 0), 64), cap_s)


class Heartbeat:
    """Application-level keep-alive: the market channel expects a text ``PING`` about
    every 10 s and answers ``PONG``. Without PINGs the server drops the socket after
    about 10 s. Without PONGs for ``stale_s`` the connection is presumed dead."""

    def __init__(self, interval_s: float = HEARTBEAT_INTERVAL_S, stale_s: float = HEARTBEAT_STALE_S,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.interval_s = interval_s
        self.stale_s = stale_s
        self._clock = clock
        self.last_pong = clock()
        self.pings_sent = 0
        self.pongs = 0

    def reset(self) -> None:
        self.last_pong = self._clock()

    def on_text(self, text: str | bytes) -> bool:
        """``True`` (and liveness refreshed) if the frame is a ``PONG``."""
        if text == "PONG" or text == b"PONG":
            self.last_pong = self._clock()
            self.pongs += 1
            return True
        return False

    def is_stale(self, now: float | None = None) -> bool:
        return (self._clock() if now is None else now) - self.last_pong > self.stale_s

    async def run(self, send: Callable[[str], Awaitable[None]], on_stale: Callable[[], Awaitable[None]],
                  watchdog_s: float = WATCHDOG_INTERVAL_S) -> None:
        """Send ``PING`` every ``interval_s``; check staleness at least every ``watchdog_s``."""
        self.reset()
        next_ping = self._clock()
        while True:
            now = self._clock()
            if now >= next_ping:
                await send("PING")
                self.pings_sent += 1
                next_ping = now + self.interval_s
            if self.is_stale(now):
                await on_stale()
                return
            await asyncio.sleep(max(0.0, min(next_ping - now, watchdog_s)))


# =========================================================================== queues
class CoalescingQueue:
    """Bounded, conflating notification queue: at most one pending item per key.

    ``offer`` is synchronous and O(1), so the socket reader never waits on a slow
    consumer. A newer item replaces the pending one for its key (``coalesced``). If the
    queue is full, a new key evicts the oldest pending key (``dropped``). Items come out
    FIFO by first-offer order. Single event loop, so no locks are needed.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._items: dict[Hashable, Any] = {}
        self._event = asyncio.Event()
        self.stats = {"offered": 0, "coalesced": 0, "dropped": 0, "max_depth": 0, "batches": 0}

    def __len__(self) -> int:
        return len(self._items)

    def offer(self, key: Hashable, item: Any) -> None:
        st = self.stats
        st["offered"] += 1
        items = self._items
        if key in items:
            items[key] = item  # keeps the key's original position
            st["coalesced"] += 1
        else:
            if len(items) >= self.maxsize:
                del items[next(iter(items))]
                st["dropped"] += 1
            items[key] = item
            if len(items) > st["max_depth"]:
                st["max_depth"] = len(items)
        self._event.set()

    def get_nowait_batch(self, max_items: int | None = None) -> list[Any]:
        items = self._items
        if max_items is None or max_items >= len(items):
            batch = list(items.values())
            items.clear()
        else:
            keys = [k for k, _ in zip(items, range(max_items))]
            batch = [items.pop(k) for k in keys]
        if not items:
            self._event.clear()
        if batch:
            self.stats["batches"] += 1
        return batch

    async def get_batch(self, max_items: int | None = None) -> list[Any]:
        """Wait until at least one item is pending, then return up to ``max_items``."""
        while not self._items:
            self._event.clear()
            await self._event.wait()
        return self.get_nowait_batch(max_items)


class DropQueue:
    """Lossless-path queue for the recorder, bounded by item count **and** bytes.

    ``offer`` never awaits. On overflow it drops the NEWEST record (what is queued
    stays contiguous) and opens a gap. The next accepted offer first enqueues
    ``MetaEvent('gap', {dropped, first_t, last_t})``, so a replay knows exactly where
    data is missing. ``on_gap`` fires once per gap episode (the feeds use it to force a
    resync). The gap marker itself is not counted against the bounds. A single record
    larger than the byte budget is admitted only into an empty queue, so that progress
    is always possible.
    """

    def __init__(self, maxsize_items: int = 100_000, maxsize_bytes: int = 256 * 2**20,
                 on_gap: Callable[[], None] | None = None, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self.maxsize_items = maxsize_items
        self.maxsize_bytes = maxsize_bytes
        self.on_gap = on_gap
        self._clock_ns = clock_ns
        self._q: deque[tuple[Any, int]] = deque()
        self._bytes = 0
        self._event = asyncio.Event()
        self._gap: dict[str, int] | None = None
        self.stats = {"offered": 0, "enqueued": 0, "dropped": 0, "gaps": 0, "max_depth": 0, "max_bytes": 0}

    @property
    def bytes(self) -> int:
        return self._bytes

    def qsize(self) -> int:
        return len(self._q)

    def _t_of(self, item: Any) -> int:
        t = getattr(item, "t_recv_ns", 0)
        return t if isinstance(t, int) and t > 0 else self._clock_ns()

    def _push(self, item: Any, nbytes: int) -> None:
        self._q.append((item, nbytes))
        self._bytes += nbytes
        st = self.stats
        st["max_depth"] = max(st["max_depth"], len(self._q))
        st["max_bytes"] = max(st["max_bytes"], self._bytes)
        self._event.set()

    def flush_gap(self) -> bool:
        """Enqueue the pending gap marker now (e.g. at shutdown); ``True`` if one was pending."""
        gap = self._gap
        if gap is None:
            return False
        self._gap = None
        self._push(MetaEvent("gap", dict(gap), self._clock_ns()), 0)
        return True

    def offer(self, item: Any, nbytes: int = 0) -> bool:
        self.stats["offered"] += 1
        q = self._q
        if len(q) >= self.maxsize_items or (len(q) > 0 and self._bytes + nbytes > self.maxsize_bytes):
            t = self._t_of(item)
            self.stats["dropped"] += 1
            if self._gap is None:
                self._gap = {"dropped": 1, "first_t": t, "last_t": t}
                self.stats["gaps"] += 1
                logger.warning("raw queue full (%d items, %d bytes): dropping records", len(q), self._bytes)
                if self.on_gap is not None:
                    self.on_gap()
            else:
                self._gap["dropped"] += 1
                self._gap["last_t"] = t
            return False
        self.flush_gap()
        self._push(item, nbytes)
        self.stats["enqueued"] += 1
        return True

    def get_nowait(self) -> Any:
        if not self._q:
            raise asyncio.QueueEmpty
        item, nbytes = self._q.popleft()
        self._bytes -= nbytes
        if not self._q:
            self._event.clear()
        return item

    async def get(self) -> Any:
        while not self._q:
            self._event.clear()
            await self._event.wait()
        return self.get_nowait()


@dataclass(slots=True)
class RawRecord:
    """One line of the raw recording: the WS frame text, or a REST book / meta dict."""

    t_recv_ns: int
    src: str  # ws | rest_book | poll
    conn: int
    msg: str | dict


# =========================================================================== REST
class ClobHTTPError(RuntimeError):
    def __init__(self, status: int | None, url: str, body: str = "") -> None:
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body


class ClobRestClient:
    """Synchronous public CLOB REST client (``requests``). Call it from async code via
    ``asyncio.to_thread``.

    Rate limits are IP-based and mostly delay rather than reject requests. 429, 5xx,
    connection errors and timeouts are retried with full-jitter backoff (``Retry-After``
    honoured); any other 4xx raises ``ClobHTTPError`` immediately. The session keeps
    ``trust_env`` so the environment's proxy and CA bundle apply; TLS verification is
    never disabled. ``/price`` is deliberately absent: its BUY/SELL semantics are
    ambiguous (UNVERIFIED), so bid and ask always come from ``/book``.
    """

    def __init__(self, base_url: str = CLOB_REST_URL, session: requests.Session | None = None,
                 timeout_s: float = 10.0, max_retries: int = 4, backoff_base_s: float = 0.5,
                 min_interval_s: float = 0.02, sleep: Callable[[float], None] = time.sleep, *,
                 backoff_cap_s: float = BACKOFF_CAP_S, rng: random.Random | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.min_interval_s = min_interval_s
        self._sleep = sleep
        self._rng = rng
        self._clock = clock
        self._lock = threading.Lock()
        self._next_ok = 0.0
        self.stats = {"requests": 0, "retries": 0, "errors": 0}

    def _throttle(self) -> None:
        with self._lock:
            now = self._clock()
            wait = self._next_ok - now
            self._next_ok = max(now, self._next_ok) + self.min_interval_s
        if wait > 0:
            self._sleep(wait)

    def _retry_wait(self, attempt: int, resp: Any = None) -> None:
        delay = backoff_delay(attempt, base_s=self.backoff_base_s, cap_s=self.backoff_cap_s, rng=self._rng)
        retry_after = getattr(resp, "headers", {}).get("Retry-After") if resp is not None else None
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = max(delay, min(float(retry_after), self.backoff_cap_s))
        self.stats["retries"] += 1
        self._sleep(delay)

    def _request(self, method: str, path: str, *, params: Mapping[str, Any] | None = None,
                 json_body: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.stats["requests"] += 1
            try:
                resp = self.session.request(method, url, params=params, json=json_body, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    self.stats["errors"] += 1
                    raise
                logger.info("%s %s failed (%s); retrying", method, path, type(exc).__name__)
                self._retry_wait(attempt)
                continue
            status = resp.status_code
            if status == 429 or 500 <= status < 600:
                if attempt >= self.max_retries:
                    self.stats["errors"] += 1
                    raise ClobHTTPError(status, url, resp.text)
                logger.info("%s %s -> HTTP %d; retrying", method, path, status)
                self._retry_wait(attempt, resp)
                continue
            if status >= 400:
                self.stats["errors"] += 1
                raise ClobHTTPError(status, url, resp.text)
            try:
                return resp.json()
            except ValueError as exc:
                self.stats["errors"] += 1
                raise ClobHTTPError(status, url, f"invalid JSON: {exc}") from exc
        raise AssertionError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ endpoints
    def get_book(self, token_id: str) -> dict:
        return self._request("GET", "/book", params={"token_id": token_id})

    def get_books(self, token_ids: Sequence[str]) -> list[dict]:
        """``POST /books`` in chunks of 500 ids; returns the concatenated book objects."""
        out: list[dict] = []
        ids = list(token_ids)
        for i in range(0, len(ids), BOOKS_CHUNK):
            data = self._request("POST", "/books", json_body=[{"token_id": t} for t in ids[i:i + BOOKS_CHUNK]])
            if not isinstance(data, list):
                raise ClobHTTPError(200, f"{self.base_url}/books", f"expected a list, got {type(data).__name__}")
            out.extend(data)
        return out

    def get_midpoints(self, token_ids: Sequence[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        ids = list(token_ids)
        for i in range(0, len(ids), BOOKS_CHUNK):
            data = self._request("POST", "/midpoints", json_body=[{"token_id": t} for t in ids[i:i + BOOKS_CHUNK]])
            out.update({str(k): float(v) for k, v in (data or {}).items() if opt_float(v) is not None})
        return out

    def get_tick_size(self, token_id: str) -> float:
        return float(self._request("GET", "/tick-size", params={"token_id": token_id})["minimum_tick_size"])

    def get_neg_risk(self, token_id: str) -> bool:
        return bool(self._request("GET", "/neg-risk", params={"token_id": token_id})["neg_risk"])

    def get_market(self, condition_id: str) -> dict:
        return self._request("GET", f"/markets/{condition_id}")

    def get_clob_market(self, condition_id: str) -> dict:
        """Compact market info: ``t`` tokens, ``mts`` tick, ``nr`` negRisk, ``fd{r,e,to}`` fees."""
        return self._request("GET", f"/clob-markets/{condition_id}")

    def get_prices_history(self, token_id: str, *, start_ts: int | None = None, end_ts: int | None = None,
                           interval: str | None = None, fidelity: int = 1) -> list[tuple[int, float]]:
        """``GET /prices-history`` -> sorted ``[(t_seconds, price)]``.

        Pass either ``interval`` (``max``/``1w``/``1d``/``6h``/``1h``) or a
        ``start_ts``/``end_ts`` window in unix seconds, never both. ``fidelity`` is the
        bucket size in minutes. Windows longer than about 15 days, and fine fidelity on
        resolved markets, are known to come back empty; chunking and the fidelity
        fallback are the caller's job.
        """
        if (interval is None) == (start_ts is None and end_ts is None):
            raise ValueError("pass exactly one of interval or start_ts/end_ts")
        params: dict[str, Any] = {"market": token_id, "fidelity": int(fidelity)}
        if interval is not None:
            params["interval"] = interval
        if start_ts is not None:
            params["startTs"] = int(start_ts)
        if end_ts is not None:
            params["endTs"] = int(end_ts)
        data = self._request("GET", "/prices-history", params=params)
        hist = (data or {}).get("history") or []
        return sorted((int(h["t"]), float(h["p"])) for h in hist)

    def close(self) -> None:
        self.session.close()


# =========================================================================== feeds
@dataclass(slots=True)
class FeedStats:
    mode: str = "ws"
    connects: int = 0
    reconnects: int = 0
    disconnects: int = 0
    frames: int = 0
    data_frames: int = 0
    bytes: int = 0
    pings: int = 0
    pongs: int = 0
    stale_closes: int = 0
    errors: int = 0
    resync_requests: int = 0
    resyncs: int = 0
    resync_failures: int = 0
    resync_discarded: int = 0
    polls: int = 0
    poll_failures: int = 0
    reader_yields: int = 0
    max_loop_lag_s: float = 0.0
    last_loop_lag_s: float = 0.0
    last_frame_ns: int = 0
    last_error: str = ""
    resync_reasons: Counter = field(default_factory=Counter)
    parse: ParseStats = field(default_factory=ParseStats)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_size(obj: Any) -> int:
    return len(json.dumps(obj, separators=(",", ":")))


class _FeedBase:
    """Plumbing shared by both feeds: books, fan-out queues, listeners, lifecycle."""

    def __init__(self, asset_ids: Iterable[str], *, books: BookManager | None, mode: str,
                 clock_ns: Callable[[], int], lag_monitor_interval_s: float) -> None:
        self._asset_ids: list[str] = list(dict.fromkeys(str(a) for a in asset_ids))
        self._asset_set = set(self._asset_ids)
        self.books = books if books is not None else BookManager()
        self.books.add_assets(self._asset_ids)
        user_cb = self.books.on_resync_needed

        def on_resync_needed(asset_id: str, reason: str) -> None:
            self._on_book_resync(asset_id, reason)
            if user_cb is not None:
                user_cb(asset_id, reason)

        self.books.on_resync_needed = on_resync_needed
        self.stats = FeedStats(mode=mode)
        self._clock_ns = clock_ns
        self.lag_monitor_interval_s = lag_monitor_interval_s
        self._upd_queues: list[CoalescingQueue] = []
        self._raw_queues: list[DropQueue] = []
        self._listeners: list[Callable[[Event], None]] = []
        self._tasks: set[asyncio.Task] = set()
        self._run_task: asyncio.Task | None = None
        self._stopping = False

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(self._asset_ids)

    # ------------------------------------------------------------------ fan-out
    def subscribe_updates(self, maxsize: int = 4096) -> CoalescingQueue:
        """Queue of ``(asset_id, t_recv_ns)`` items keyed by asset id: "re-read this book".

        ``t_recv_ns`` is the receive time of the latest frame that changed the asset's top
        of book or clean flag.
        """
        q = CoalescingQueue(maxsize)
        self._upd_queues.append(q)
        return q

    def subscribe_raw(self, maxsize_items: int = 100_000, maxsize_bytes: int = 256 * 2**20) -> DropQueue:
        """Lossless queue of ``RawRecord`` and ``MetaEvent`` items for the recorder."""
        q = DropQueue(maxsize_items, maxsize_bytes, on_gap=self._on_raw_gap, clock_ns=self._clock_ns)
        self._raw_queues.append(q)
        return q

    def add_listener(self, callback: Callable[[Event], None]) -> None:
        """Called synchronously with control events: tick size, resolution, new market, meta."""
        self._listeners.append(callback)

    def _publish_raw(self, item: Any, nbytes: int) -> None:
        for q in self._raw_queues:
            q.offer(item, nbytes)

    def _notify(self, asset_ids: Iterable[str], t_recv_ns: int) -> None:
        for q in self._upd_queues:
            for aid in asset_ids:
                q.offer(aid, (aid, t_recv_ns))

    def _emit(self, ev: Event) -> None:
        for cb in self._listeners:
            try:
                cb(ev)
            except Exception:  # noqa: BLE001 - a broken listener must not kill the feed
                self.stats.errors += 1
                logger.exception("feed listener failed on %s", type(ev).__name__)

    def _emit_meta(self, kind: str, data: dict) -> None:
        ev = MetaEvent(kind, data, self._clock_ns())
        self._publish_raw(ev, 128)
        self._emit(ev)

    def _dispatch(self, events: Sequence[Event], t_recv_ns: int) -> set[str]:
        changed = self.books.apply(events)
        if changed:
            self._notify(changed, t_recv_ns)
        if self._listeners:
            for ev in events:
                if isinstance(ev, CONTROL_EVENT_TYPES):
                    self._emit(ev)
        return changed

    def _mark_dirty(self, asset_ids: Iterable[str] | None, reason: str) -> None:
        newly = self.books.mark_dirty(asset_ids, reason)
        if newly:
            self._notify(newly, self._clock_ns())

    def _add_ids(self, asset_ids: Iterable[str]) -> list[str]:
        new = [a for a in dict.fromkeys(str(x) for x in asset_ids) if a not in self._asset_set]
        self._asset_ids.extend(new)
        self._asset_set.update(new)
        self.books.add_assets(new)
        return new

    def _remove_ids(self, asset_ids: Iterable[str]) -> list[str]:
        gone = [a for a in dict.fromkeys(str(x) for x in asset_ids) if a in self._asset_set]
        if gone:
            drop = set(gone)
            self._asset_ids = [a for a in self._asset_ids if a not in drop]
            self._asset_set -= drop
            self.books.remove_assets(gone)
        return gone

    # ------------------------------------------------------------------ hooks
    def _on_book_resync(self, asset_id: str, reason: str) -> None:  # pragma: no cover - overridden
        pass

    def _on_raw_gap(self) -> None:  # pragma: no cover - overridden
        pass

    async def _run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    # ------------------------------------------------------------------ lifecycle
    def _spawn(self, coro: Awaitable[Any], name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=f"clob:{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _lag_monitor(self) -> None:
        """Event-loop health: how late a short sleep wakes up is the scheduling delay."""
        loop = asyncio.get_running_loop()
        interval = self.lag_monitor_interval_s
        while True:
            t0 = loop.time()
            await asyncio.sleep(interval)
            lag = max(loop.time() - t0 - interval, 0.0)
            self.stats.last_loop_lag_s = lag
            if lag > self.stats.max_loop_lag_s:
                self.stats.max_loop_lag_s = lag

    async def run(self) -> None:
        """Run until ``stop()``, then return normally. A feed is one-shot: once stopped
        (even before ``run`` got scheduled) it does not start again."""
        if self._run_task is not None:
            raise RuntimeError("feed.run() was already called")
        self._run_task = asyncio.current_task()
        if self._stopping:
            return
        if self.lag_monitor_interval_s > 0:
            self._spawn(self._lag_monitor(), "lag-monitor")
        try:
            await self._run()
        except asyncio.CancelledError:
            if not self._stopping:
                raise
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
        finally:
            tasks = list(self._tasks)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop ``run()`` from another task; leaves no feed task pending."""
        self._stopping = True
        task = self._run_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()
            await asyncio.wait({task})
        for q in self._raw_queues:
            q.flush_gap()


class MarketDataFeed(_FeedBase):
    """Polymarket market-channel WebSocket feed with books, resync and fan-out."""

    def __init__(self, asset_ids: Sequence[str], *, url: str = WS_MARKET_URL,
                 rest: ClobRestClient | None = None, books: BookManager | None = None,
                 custom_feature_enabled: bool = True, initial_dump: bool = True,
                 extra_subscribe_fields: Mapping[str, Any] | None = None,
                 heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S, stale_after_s: float = HEARTBEAT_STALE_S,
                 watchdog_s: float = WATCHDOG_INTERVAL_S, open_timeout_s: float = 10.0,
                 close_timeout_s: float = 0.5, max_size: int = 16 * 2**20, max_queue: int = 64,
                 proxy: str | bool | None = True, backoff_base_s: float = BACKOFF_BASE_S,
                 backoff_cap_s: float = BACKOFF_CAP_S, resync_on_connect: bool = True,
                 resync_min_interval_s: float = 10.0, yield_every_frames: int = 50, yield_every_s: float = 0.005,
                 lag_monitor_interval_s: float = 0.1, connect: Callable[..., Any] | None = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, rng: random.Random | None = None,
                 clock_ns: Callable[[], int] = time.time_ns) -> None:
        super().__init__(asset_ids, books=books, mode="ws", clock_ns=clock_ns,
                         lag_monitor_interval_s=lag_monitor_interval_s)
        self.url = url
        self.rest = rest
        self.custom_feature_enabled = custom_feature_enabled
        self.initial_dump = initial_dump
        self.extra_subscribe_fields = dict(extra_subscribe_fields or {})
        self.heartbeat = Heartbeat(heartbeat_interval_s, stale_after_s)
        self.watchdog_s = watchdog_s
        self.open_timeout_s = open_timeout_s
        self.close_timeout_s = close_timeout_s
        self.max_size = max_size
        self.max_queue = max_queue
        self.proxy = proxy
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.resync_on_connect = resync_on_connect
        self.resync_min_interval_s = resync_min_interval_s
        self.yield_every_frames = max(1, yield_every_frames)
        self.yield_every_s = yield_every_s
        self._connect = connect if connect is not None else ws_connect
        self._sleep = sleep
        self._rng = rng
        self._ws: Any = None
        self._conn_seq = 0
        self._connected = asyncio.Event()
        self._attempt = 0
        self._got_data = False
        self._resync_pending: dict[str, str] = {}  # asset -> first reason
        self._last_resync: dict[str, float] = {}
        self._resync_wakeup = asyncio.Event()

    @property
    def conn_seq(self) -> int:
        return self._conn_seq

    def subscribe_frame(self) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "type": "market",
            "assets_ids": sorted(self._asset_ids),  # "assets_ids" (sic) is the wire name
            "custom_feature_enabled": self.custom_feature_enabled,
            "initial_dump": self.initial_dump,
        }
        frame.update(self.extra_subscribe_fields)
        return frame

    # ------------------------------------------------------------------ dynamic subscriptions
    async def _send(self, frame: Mapping[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            return  # the next connect sends the full subscribe frame
        with contextlib.suppress(ConnectionClosed):
            await ws.send(json.dumps(frame))

    async def add_assets(self, asset_ids: Iterable[str]) -> list[str]:
        """Subscribe more assets on the open socket. Ids already subscribed are skipped,
        because the server rejects duplicates. Returns the newly added ids."""
        new = self._add_ids(asset_ids)
        if new:
            frame: dict[str, Any] = {"operation": "subscribe", "assets_ids": new}
            if self.custom_feature_enabled:
                frame["custom_feature_enabled"] = True
            await self._send(frame)
            self.request_resync(new, "add_assets")
        return new

    async def remove_assets(self, asset_ids: Iterable[str]) -> list[str]:
        gone = self._remove_ids(asset_ids)
        if gone:
            for aid in gone:
                self._resync_pending.pop(aid, None)
                self._last_resync.pop(aid, None)
            await self._send({"operation": "unsubscribe", "assets_ids": gone})
        return gone

    # ------------------------------------------------------------------ resync
    def request_resync(self, asset_ids: Iterable[str] | None = None, reason: str = "") -> None:
        """Queue a REST ``POST /books`` snapshot. Requests coalesce per asset, and each
        asset is resynced at most once per ``resync_min_interval_s`` (later requests are
        deferred, never dropped)."""
        if self.rest is None:
            return
        ids = self._asset_ids if asset_ids is None else [a for a in asset_ids if a in self._asset_set]
        if not ids:
            return
        self.stats.resync_requests += 1
        self.stats.resync_reasons[reason or "unspecified"] += 1
        for aid in ids:
            self._resync_pending.setdefault(aid, reason)
        self._resync_wakeup.set()

    def _on_book_resync(self, asset_id: str, reason: str) -> None:
        self.request_resync([asset_id], reason)

    def _on_raw_gap(self) -> None:
        self.request_resync(None, "raw_gap")

    async def _resync_worker(self) -> None:
        while True:
            if not self._resync_pending:
                self._resync_wakeup.clear()
                await self._resync_wakeup.wait()
                continue
            if not self._connected.is_set():
                await self._connected.wait()
                continue
            now = time.monotonic()
            interval = self.resync_min_interval_s
            due = [a for a in self._resync_pending if now - self._last_resync.get(a, -1e18) >= interval]
            if not due:
                wait = min(self._last_resync[a] for a in self._resync_pending) + interval - now
                self._resync_wakeup.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._resync_wakeup.wait(), max(wait, 0.0))
                continue
            await self._resync_once(due, now)

    async def _resync_once(self, due: list[str], now: float) -> None:
        reasons = sorted({self._resync_pending.pop(a) or "unspecified" for a in due})
        for aid in due:
            self._last_resync[aid] = now
        seq = self._conn_seq  # tag: only a snapshot issued on the current connection may heal it
        try:
            payload = await asyncio.to_thread(self.rest.get_books, due)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - network/HTTP failures; retried after min interval
            self.stats.resync_failures += 1
            self.stats.last_error = f"resync: {type(exc).__name__}: {exc}"
            logger.warning("resync of %d asset(s) failed: %s", len(due), exc)
            for aid in due:
                if aid in self._asset_set:
                    self._resync_pending.setdefault(aid, "retry")
            return
        if seq != self._conn_seq or not self._connected.is_set():
            # Issued before the current connection: it cannot contain the updates missed
            # during the outage, so it must not clear 'dirty'. Ask again on this connection.
            self.stats.resync_discarded += 1
            logger.info("discarding resync from connection %d (now %d)", seq, self._conn_seq)
            for aid in due:
                self._last_resync.pop(aid, None)  # a wasted call must not delay the fresh one
                if aid in self._asset_set:
                    self._resync_pending.setdefault(aid, "stale_resync")
            return
        t = self._clock_ns()
        events: list[Event] = []
        records: list[RawRecord] = []
        for obj in payload:
            try:
                ev = parse_rest_book(obj, t, "rest")
            except (KeyError, TypeError, ValueError) as exc:
                self.stats.parse.malformed += 1
                logger.debug("malformed REST book: %r", exc)
                continue
            events.append(ev)
            records.append(RawRecord(t, "rest_book", seq, obj))
        self.stats.resyncs += 1
        self._emit_meta("resync", {"conn": seq, "assets": len(events), "requested": len(due), "reasons": reasons})
        for rec in records:
            self._publish_raw(rec, _json_size(rec.msg))
        self._dispatch(events, t)
        missing = set(due) - {ev.asset_id for ev in events}
        if missing:
            logger.warning("resync returned no book for %d asset(s); they stay dirty", len(missing))

    # ------------------------------------------------------------------ connection loop
    async def _run(self) -> None:
        if self.rest is not None:
            self._spawn(self._resync_worker(), "resync")
        while not self._stopping:
            await self._session()
            if self._stopping:
                break
            delay = backoff_delay(self._attempt, base_s=self.backoff_base_s, cap_s=self.backoff_cap_s, rng=self._rng)
            self._attempt += 1
            self.stats.reconnects += 1
            await self._sleep(delay)

    async def _session(self) -> None:
        hb_task: asyncio.Task | None = None
        ws: Any = None
        seq = 0
        error = ""
        try:
            async with self._connect(self.url, open_timeout=self.open_timeout_s, close_timeout=self.close_timeout_s,
                                     max_size=self.max_size, max_queue=self.max_queue, proxy=self.proxy) as ws:
                self._conn_seq += 1
                seq = self._conn_seq
                self._ws = ws
                self._got_data = False
                self.stats.connects += 1
                self._mark_dirty(None, "connect")
                self._emit_meta("connect", {"conn": seq, "url": self.url, "attempt": self._attempt})
                await ws.send(json.dumps(self.subscribe_frame()))
                self._connected.set()
                if self.resync_on_connect:
                    self.request_resync(None, "connect")
                hb_task = self._spawn(self.heartbeat.run(self._pinger(ws), self._stale_closer(ws), self.watchdog_s),
                                      f"heartbeat-{seq}")
                await self._read(ws, seq)
        except InvalidURI:
            raise
        except _NETWORK_ERRORS as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("market websocket error: %s", error)
        except Exception as exc:  # noqa: BLE001 - keep a long-running recorder alive; counted
            error = f"{type(exc).__name__}: {exc}"
            self.stats.errors += 1
            logger.exception("market websocket session failed")
        finally:
            self._connected.clear()
            if hb_task is not None:
                hb_task.cancel()
                await asyncio.gather(hb_task, return_exceptions=True)
            if error:
                self.stats.last_error = error
            if seq:
                self._ws = None
                self.stats.disconnects += 1
                self._emit_meta("disconnect", {"conn": seq, "code": getattr(ws, "close_code", None),
                                               "reason": getattr(ws, "close_reason", None) or "", "error": error})
                self._mark_dirty(None, "disconnect")

    def _pinger(self, ws: Any) -> Callable[[str], Awaitable[None]]:
        async def send(text: str) -> None:
            with contextlib.suppress(ConnectionClosed):
                await ws.send(text)
                self.stats.pings += 1
        return send

    def _stale_closer(self, ws: Any) -> Callable[[], Awaitable[None]]:
        async def close() -> None:
            self.stats.stale_closes += 1
            logger.warning("no PONG for %.1f s; closing the market websocket", self.heartbeat.stale_s)
            await ws.close()
        return close

    async def _read(self, ws: Any, seq: int) -> None:
        hb, st, clock, perf = self.heartbeat, self.stats, self._clock_ns, time.perf_counter
        n_since, last_yield = 0, perf()
        async for raw in ws:
            t = clock()
            st.frames += 1
            st.bytes += len(raw)
            st.last_frame_ns = t
            if hb.on_text(raw):
                st.pongs += 1
                continue
            if self._handle_frame(raw, t, seq) and not self._got_data:
                self._got_data = True
                self._attempt = 0  # reset only once the connection has delivered real data
            n_since += 1
            # A flood of already-buffered frames never suspends `async for`; yield explicitly
            # so the heartbeat, the resync worker and the consumers keep running.
            if n_since >= self.yield_every_frames or perf() - last_yield >= self.yield_every_s:
                await asyncio.sleep(0)
                st.reader_yields += 1
                n_since, last_yield = 0, perf()

    def _handle_frame(self, raw: str | bytes, t: int, seq: int) -> bool:
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8", "replace")
        if self._raw_queues:
            self._publish_raw(RawRecord(t, "ws", seq, text), len(raw))
        events = parse_frame(text, t, self.stats.parse)
        if not events:
            return False
        self.stats.data_frames += 1
        self._dispatch(events, t)
        return True


class RestPollingFeed(_FeedBase):
    """Polls ``POST /books`` every ``interval_s`` and offers the same API as
    ``MarketDataFeed``. A book is emitted (applied, recorded, notified) only when it
    changed: equal ``hash`` means unchanged, otherwise the level sets are compared. A
    failed poll marks the books dirty until the next successful poll. Dynamics faster
    than the poll interval are invisible in this mode.
    """

    def __init__(self, asset_ids: Sequence[str], *, rest: ClobRestClient, interval_s: float = 1.0,
                 jitter_s: float = 0.1, books: BookManager | None = None,
                 clock_ns: Callable[[], int] = time.time_ns, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 rng: random.Random | None = None, lag_monitor_interval_s: float = 0.1) -> None:
        super().__init__(asset_ids, books=books, mode="poll", clock_ns=clock_ns,
                         lag_monitor_interval_s=lag_monitor_interval_s)
        self.rest = rest
        self.interval_s = interval_s
        self.jitter_s = jitter_s
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._hashes: dict[str, str] = {}  # hash of the polled book known to equal our book
        self._force: set[str] = set()  # re-emit even if unchanged (resync / recording gap)
        self._healthy = False

    async def add_assets(self, asset_ids: Iterable[str]) -> list[str]:
        return self._add_ids(asset_ids)

    async def remove_assets(self, asset_ids: Iterable[str]) -> list[str]:
        gone = self._remove_ids(asset_ids)
        for aid in gone:
            self._hashes.pop(aid, None)
        return gone

    def request_resync(self, asset_ids: Iterable[str] | None = None, reason: str = "") -> None:
        """Every poll is a full snapshot, so a resync just re-emits the books next poll."""
        self.stats.resync_requests += 1
        self.stats.resync_reasons[reason or "unspecified"] += 1
        self._force.update(self._asset_ids if asset_ids is None else asset_ids)

    def _on_book_resync(self, asset_id: str, reason: str) -> None:
        self.request_resync([asset_id], reason)

    def _on_raw_gap(self) -> None:
        self.request_resync(None, "raw_gap")  # re-anchor the recording after dropped records

    def _is_new(self, ev: BookEvent) -> bool:
        """Changed relative to the book we hold (not merely to the previous poll, so a
        snapshot the manager dropped as stale can never mask a later change)."""
        aid = ev.asset_id
        book = self.books.books[aid]
        if book.dirty or aid in self._force:
            return True
        if ev.hash and self._hashes.get(aid) == ev.hash:
            return False
        if {p: q for p, q in ev.bids if q > 0} == book.bids and {p: q for p, q in ev.asks if q > 0} == book.asks:
            if ev.hash:
                self._hashes[aid] = ev.hash
            return False
        return True

    async def poll_once(self) -> set[str]:
        """One ``POST /books`` round; returns the assets whose state changed."""
        ids = list(self._asset_ids)
        if not ids:
            return set()
        self.stats.polls += 1
        try:
            payload = await asyncio.to_thread(self.rest.get_books, ids)
        except Exception as exc:  # noqa: BLE001 - network/HTTP failure: books are stale until next success
            self.stats.poll_failures += 1
            self.stats.last_error = f"poll: {type(exc).__name__}: {exc}"
            if self._healthy:
                self._healthy = False
                self.stats.disconnects += 1
                logger.warning("poll failed: %s", exc)
                self._emit_meta("disconnect", {"conn": 0, "mode": "poll", "reason": "poll_error",
                                               "error": self.stats.last_error})
            self._mark_dirty(ids, "poll_error")
            return set()
        t = self._clock_ns()
        if not self._healthy:
            self._healthy = True
            self.stats.connects += 1
            self._emit_meta("connect", {"conn": 0, "mode": "poll", "interval_s": self.interval_s})
        self.stats.frames += 1
        events: list[Event] = []
        seen: set[str] = set()
        for obj in payload:
            try:
                ev = parse_rest_book(obj, t, "poll")
            except (KeyError, TypeError, ValueError) as exc:
                self.stats.parse.malformed += 1
                logger.debug("malformed REST book: %r", exc)
                continue
            aid = ev.asset_id
            if aid not in self._asset_set:
                continue
            seen.add(aid)
            book = self.books.books[aid]
            if ev.tick_size and ev.tick_size != book.tick:  # polling has no tick_size_change event
                if book.snapshot_ts_ms >= 0:
                    self._emit(TickSizeEvent(aid, ev.market, book.tick, ev.tick_size, ev.ts_ms, t))
                self.books.set_tick(aid, ev.tick_size)
            if not self._is_new(ev):
                continue
            events.append(ev)
            self._publish_raw(RawRecord(t, "poll", 0, obj), _json_size(obj))
        self._force.clear()
        missing = set(ids) - seen
        if missing:
            self._mark_dirty(missing, "poll_missing")
        if not events:
            return set()
        self.stats.data_frames += 1
        return self._dispatch(events, t)

    async def _run(self) -> None:
        try:
            while not self._stopping:
                t0 = time.monotonic()
                await self.poll_once()
                elapsed = time.monotonic() - t0
                await self._sleep(max(0.0, self.interval_s - elapsed) + self._rng.uniform(0.0, self.jitter_s))
        finally:
            if self._healthy:
                self._healthy = False
                self._emit_meta("disconnect", {"conn": 0, "mode": "poll", "reason": "stop"})


def make_feed(mode: str, asset_ids: Sequence[str], *, rest: ClobRestClient | None,
              **kw: Any) -> MarketDataFeed | RestPollingFeed:
    """``"ws"`` -> ``MarketDataFeed``; ``"poll"`` -> ``RestPollingFeed`` (needs ``rest``)."""
    if mode == "ws":
        return MarketDataFeed(asset_ids, rest=rest, **kw)
    if mode == "poll":
        if rest is None:
            raise ValueError("poll mode needs a ClobRestClient")
        return RestPollingFeed(asset_ids, rest=rest, **kw)
    raise ValueError(f"unknown feed mode {mode!r} (expected 'ws' or 'poll')")


__all__ = [
    "BACKOFF_BASE_S", "BACKOFF_CAP_S", "CLOB_REST_URL", "HEARTBEAT_INTERVAL_S", "HEARTBEAT_STALE_S",
    "WATCHDOG_INTERVAL_S", "WS_MARKET_URL", "ClobHTTPError", "ClobRestClient", "CoalescingQueue", "DropQueue",
    "FeedStats", "Heartbeat", "MarketDataFeed", "ParseStats", "RawRecord", "RestPollingFeed", "backoff_delay",
    "make_feed", "parse_frame", "parse_message", "parse_rest_book",
]
