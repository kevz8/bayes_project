"""Record, replay and export live Polymarket order books; backfill price history.

Storage layout (``data/historical_books/<group>/``)::

    manifest.json                                   sessions, files, sha256, gaps
    raw/YYYY-MM-DD/HH-<session_id>.jsonl.gz         raw lines, rotated every UTC hour
    tob/<basket_id>.csv.gz                          exported top-of-book table (TOB_COLUMNS)

Raw lines (see ``src/events.py``) keep the server's WebSocket text verbatim, so a
replay is byte-for-byte deterministic and parser changes can be re-applied later::

    {"t": <recv ns>, "src": "ws", "conn": 3, "msg": <server JSON text>}
    {"t": ..., "src": "rest_book"|"poll", "conn": 3, "msg": {<REST /book object>}}
    {"t": ..., "src": "meta", "kind": "connect|disconnect|gap|resync|stop|...", "data": {...}}

Every ``record`` run is one *session* appended to the manifest. Background jobs in the
cloud container are capped at ~2 h and the container restarts often, so recordings are
deliberately chunked; the gaps between sessions are explicit, and ``export_tob`` writes a
``clean=False`` marker at every session end so nothing is ever forward-filled across one.

Price-history backfill (``data/prices_history/<basket_id>/``) pulls ``/prices-history``
in <=14-day windows (longer windows are rejected with HTTP 400): 1-minute data for the
recent window plus hourly data back to the market's creation.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import signal
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from . import HIST_ROOT, PRICES_ROOT, SYNTH_ROOT, code_version
from .clob_client import WS_MARKET_URL, ClobRestClient, RawRecord, make_feed, parse_frame, parse_message, parse_rest_book
from .config import Basket, MarketsConfig, get_basket, load_markets
from .events import TOB_COLUMNS, Event, MetaEvent
from .orderbook import BookManager

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
CHUNK_DAYS = 14
PRICES_SEMANTICS = ("p is Polymarket's price-history value (a midpoint / last-trade style display price; "
                    "the exact definition is UNVERIFIED). It is NOT executable: no bid, ask or depth.")


# --------------------------------------------------------------------------- helpers
def utc_iso(t_ns: int | None = None) -> str:
    t = time.time_ns() if t_ns is None else t_ns
    return datetime.fromtimestamp(t / 1e9, timezone.utc).isoformat()


def parse_duration(text: str) -> float:
    """``"118m"`` -> 7080.0; units s/m/h/d; a bare number is seconds."""
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([smhd]?)\s*", str(text))
    if not m:
        raise ValueError(f"bad duration {text!r}")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def _guard_root(root: Path) -> None:
    if Path(root).resolve() == Path(SYNTH_ROOT).resolve():
        raise ValueError("refusing to record real data into the synthetic data directory")


def load_manifest(group_dir: Path) -> dict:
    p = Path(group_dir) / MANIFEST
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {"group": Path(group_dir).name, "synthetic": False, "sessions": []}


def save_session(group_dir: Path, session: Mapping[str, Any]) -> None:
    """Insert or replace one session in the group manifest (atomic)."""
    man = load_manifest(group_dir)
    if man.get("synthetic"):
        raise ValueError("refusing to mix real sessions into a synthetic manifest")
    man["sessions"] = [s for s in man["sessions"] if s["session_id"] != session["session_id"]] + [dict(session)]
    man["sessions"].sort(key=lambda s: s.get("started") or "")
    _atomic_write_json(Path(group_dir) / MANIFEST, man)


# --------------------------------------------------------------------------- writer
@dataclass
class FileInfo:
    path: str  # relative to the group dir
    bytes: int
    lines: int
    sha256: str | None


class RawRecordWriter:
    """Hour-rotated gzip JSONL writer. Methods are blocking; call them via ``to_thread``."""

    def __init__(self, group_dir: Path, session_id: str, *, compresslevel: int = 6,
                 clock_ns: Callable[[], int] = time.time_ns) -> None:
        self.group_dir = Path(group_dir)
        self.session_id = session_id
        self.compresslevel = compresslevel
        self._clock_ns = clock_ns
        self._key: str | None = None
        self._fh: Any = None
        self.lines: dict[str, int] = {}
        self.closed_files: set[str] = set()

    def _file_for(self, t_ns: int) -> Any:
        key = datetime.fromtimestamp(t_ns / 1e9, timezone.utc).strftime("%Y-%m-%d/%H")
        if key != self._key:
            if self._fh is not None:
                self._fh.close()
                self.closed_files.add(self._rel)
            day, hour = key.split("/")
            self._rel = f"raw/{day}/{hour}-{self.session_id}.jsonl.gz"
            (self.group_dir / self._rel).parent.mkdir(parents=True, exist_ok=True)
            self._fh = gzip.open(self.group_dir / self._rel, "at", compresslevel=self.compresslevel)
            self._key = key
            self.lines.setdefault(self._rel, 0)
        return self._fh

    @staticmethod
    def encode(item: RawRecord | MetaEvent | Mapping[str, Any]) -> tuple[int, str]:
        """(t_ns, line) - WS text is spliced verbatim unless it contains a newline."""
        if isinstance(item, MetaEvent):
            return item.t_recv_ns, json.dumps({"t": item.t_recv_ns, "src": "meta", "kind": item.kind,
                                               "data": item.data}, separators=(",", ":"), default=str)
        if isinstance(item, Mapping):
            return int(item["t"]), json.dumps(item, separators=(",", ":"), default=str)
        msg = item.msg
        if isinstance(msg, str) and "\n" not in msg and "\r" not in msg:
            return item.t_recv_ns, '{"t":%d,"src":%s,"conn":%d,"msg":%s}' % (
                item.t_recv_ns, json.dumps(item.src), item.conn, msg)
        if isinstance(msg, str):
            msg = json.loads(msg)
        return item.t_recv_ns, json.dumps({"t": item.t_recv_ns, "src": item.src, "conn": item.conn, "msg": msg},
                                          separators=(",", ":"))

    def write(self, items: Iterable[RawRecord | MetaEvent | Mapping[str, Any]]) -> int:
        n = 0
        for item in items:
            t, line = self.encode(item)
            self._file_for(t).write(line + "\n")
            self.lines[self._rel] += 1
            n += 1
        return n

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def file_infos(self, *, with_hash: bool = False) -> list[FileInfo]:
        out = []
        for rel, n in self.lines.items():
            p = self.group_dir / rel
            out.append(FileInfo(rel, p.stat().st_size if p.exists() else 0, n,
                                sha256_file(p) if with_hash and p.exists() else None))
        return out

    def close(self) -> list[FileInfo]:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return self.file_infos(with_hash=True)


# --------------------------------------------------------------------------- record
def _new_session_id(clock_ns: Callable[[], int]) -> str:
    stamp = datetime.fromtimestamp(clock_ns() / 1e9, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{random.randrange(16 ** 4):04x}"


def resolve_assets(basket_ids: Sequence[str], cfg: MarketsConfig, *, subscribe_no: bool = False
                   ) -> tuple[list[str], dict[str, list[str]]]:
    assets: list[str] = []
    baskets: dict[str, list[str]] = {}
    for bid in basket_ids:
        b = get_basket(bid, cfg)
        if b.synthetic:
            raise ValueError(f"{bid} is synthetic; synthetic data is generated, never recorded")
        baskets[bid] = list(b.yes_ids)
        ids = list(b.yes_ids) + (list(b.no_ids) if subscribe_no else [])
        assets.extend(i for i in ids if i not in assets)
    return assets, baskets


async def record(group: str, basket_ids: Sequence[str], *, mode: str = "ws", duration_s: float | None = None,
                 poll_interval_s: float = 1.0, root: Path = HIST_ROOT, cfg: MarketsConfig | None = None,
                 subscribe_no: bool = False, feed_factory: Callable[..., Any] | None = None,
                 clock_ns: Callable[[], int] = time.time_ns, manifest_every_s: float = 60.0) -> dict:
    """Record one session; returns its manifest entry."""
    _guard_root(root)
    cfg = cfg or load_markets()
    assets, baskets = resolve_assets(basket_ids, cfg, subscribe_no=subscribe_no)
    group_dir = Path(root) / group
    session_id = _new_session_id(clock_ns)
    if feed_factory is not None:
        feed = feed_factory(assets)
    else:
        kw: dict[str, Any] = {"interval_s": poll_interval_s} if mode == "poll" else {}
        feed = make_feed(mode, assets, rest=ClobRestClient(), **kw)
    raw_q = feed.subscribe_raw()
    writer = RawRecordWriter(group_dir, session_id, clock_ns=clock_ns)
    import websockets  # version provenance only

    session: dict[str, Any] = {
        "session_id": session_id, "started": utc_iso(clock_ns()), "ended": None, "mode": mode,
        "url": WS_MARKET_URL if mode == "ws" else "https://clob.polymarket.com/books",
        "assets": assets, "baskets": baskets, "n_frames": 0, "n_dropped": 0, "reconnects": 0,
        "files": [], "code_version": code_version(), "websockets_version": websockets.__version__, "notes": "",
    }
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, stop.set)

    def update_session(final: bool = False) -> None:
        st = getattr(feed, "stats", None)
        session["reconnects"] = int(getattr(st, "reconnects", 0) or 0)
        session["n_dropped"] = int(raw_q.stats.get("dropped", 0))
        session["files"] = [vars(f) for f in (writer.close() if final else writer.file_infos())]
        if final:
            session["ended"] = utc_iso(clock_ns())
        save_session(group_dir, session)

    await asyncio.to_thread(writer.write, [MetaEvent("session_start", {"session_id": session_id, "mode": mode,
                                                                        "assets": assets, "baskets": baskets},
                                                     clock_ns())])
    save_session(group_dir, session)

    async def drain_forever() -> None:
        last_manifest = time.monotonic()
        while True:
            batch = [await raw_q.get()]
            with contextlib.suppress(asyncio.QueueEmpty):
                while len(batch) < 2000:
                    batch.append(raw_q.get_nowait())
            session["n_frames"] += sum(1 for x in batch if isinstance(x, RawRecord) and x.src == "ws")
            await asyncio.to_thread(writer.write, batch)
            if time.monotonic() - last_manifest >= manifest_every_s:
                await asyncio.to_thread(writer.flush)
                update_session()
                last_manifest = time.monotonic()

    feed_task = asyncio.create_task(feed.run(), name="feed")
    drain_task = asyncio.create_task(drain_forever(), name="writer")
    waiters = [asyncio.create_task(stop.wait(), name="stop")]
    if duration_s is not None:
        waiters.append(asyncio.create_task(asyncio.sleep(duration_s), name="duration"))
    try:
        await asyncio.wait([feed_task, *waiters], return_when=asyncio.FIRST_COMPLETED)
    finally:
        reason = "signal" if stop.is_set() else ("feed_exit" if feed_task.done() else "duration")
        for w in waiters:
            w.cancel()
        await feed.stop()
        if not feed_task.done():
            feed_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await feed_task
        raw_q.flush_gap()
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task
        rest = []
        with contextlib.suppress(asyncio.QueueEmpty):
            while True:
                rest.append(raw_q.get_nowait())
        session["n_frames"] += sum(1 for x in rest if isinstance(x, RawRecord) and x.src == "ws")
        rest.append(MetaEvent("stop", {"reason": reason}, clock_ns()))
        writer.write(rest)
        update_session(final=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
    logger.info("session %s: %d frames, %d dropped, %d reconnects", session_id, session["n_frames"],
                session["n_dropped"], session["reconnects"])
    return session


# --------------------------------------------------------------------------- replay
def iter_raw_records(paths: Iterable[Path]) -> Iterator[dict]:
    """Stream raw lines; a truncated gzip tail or partial last line ends that file with a warning."""
    for p in paths:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if not line.endswith("\n"):
                        logger.warning("%s: partial last line ignored", p)
                        break
                    try:
                        yield json.loads(line)
                    except ValueError:
                        logger.warning("%s: corrupt line ignored", p)
        except (EOFError, zlib.error, gzip.BadGzipFile, OSError) as exc:
            logger.warning("%s: truncated/corrupt gzip tail (%s); stopping this file", p, type(exc).__name__)


def list_sessions(group: str, root: Path = HIST_ROOT) -> list[dict]:
    return sorted(load_manifest(Path(root) / group)["sessions"], key=lambda s: s.get("started") or "")


def raw_paths(group: str, root: Path = HIST_ROOT, session_id: str | None = None) -> list[Path]:
    out = []
    for s in list_sessions(group, root):
        if session_id in (None, s["session_id"]):
            out.extend(Path(root) / group / f["path"] for f in sorted(s["files"], key=lambda f: f["path"]))
    return out


def record_to_events(rec: Mapping[str, Any]) -> list[Event]:
    t = int(rec["t"])
    src = rec.get("src")
    if src == "meta":
        return [MetaEvent(str(rec.get("kind")), dict(rec.get("data") or {}), t)]
    msg = rec.get("msg")
    if src in ("rest_book", "poll") and isinstance(msg, Mapping):
        return [parse_rest_book(msg, t, source="rest" if src == "rest_book" else "poll")]
    if isinstance(msg, (Mapping, list)):
        if isinstance(msg, list):
            return [e for m in msg if isinstance(m, Mapping) for e in parse_message(m, t)]
        return parse_message(msg, t)
    if isinstance(msg, str):
        return parse_frame(msg, t)
    return []


def replay_frames(group: str | None = None, *, paths: Sequence[Path] | None = None, root: Path = HIST_ROOT,
                  session_id: str | None = None) -> Iterator[tuple[int, list[Event]]]:
    """Yield ``(t_recv_ns, events)`` per raw line, bracketed per session by synthetic
    ``MetaEvent("replay_start"|"replay_end")`` boundaries so consumers can reset state
    (distinct from the recorder's own ``session_start`` meta line)."""
    if paths is not None:
        groups = [(None, list(paths))]
    else:
        groups = [(s["session_id"], [Path(root) / group / f["path"] for f in sorted(s["files"], key=lambda f: f["path"])])
                  for s in list_sessions(group, root) if session_id in (None, s["session_id"])]
    for sid, ps in groups:
        last_t = 0
        first = True
        for rec in iter_raw_records(ps):
            t = int(rec["t"])
            if first:
                yield t, [MetaEvent("replay_start", {"session_id": sid}, t)]
                first = False
            last_t = t
            evs = record_to_events(rec)
            if evs:
                yield t, evs
        if not first:
            yield last_t, [MetaEvent("replay_end", {"session_id": sid}, last_t)]


# --------------------------------------------------------------------------- top-of-book export
def tob_rows(frames: Iterable[tuple[int, list[Event]]], basket: Basket) -> Iterator[tuple]:
    """Replay frames through a fresh ``BookManager`` per session and emit one TOB row per
    top-of-book change of a basket leg, plus NaN/clean=False rows at each session end."""
    leg_of = {l.yes_token_id: l.leg_id for l in basket.legs}
    books: BookManager | None = None
    last: dict[str, tuple] = {}
    for t, events in frames:
        boundary = [e for e in events if isinstance(e, MetaEvent) and e.kind in ("replay_start", "replay_end")]
        if boundary and boundary[0].kind == "replay_start":
            books = BookManager(list(leg_of))
            last = {}
            continue
        if boundary and boundary[0].kind == "replay_end":
            for yes_id, leg in leg_of.items():
                if yes_id in last:
                    yield (t, leg, math.nan, math.nan, math.nan, math.nan, False)
            last = {}
            continue
        if books is None:
            books = BookManager(list(leg_of))
        for e in events:
            if isinstance(e, MetaEvent) and e.kind in ("disconnect", "gap"):
                books.mark_dirty(None, e.kind)
        books.apply([e for e in events if not isinstance(e, MetaEvent)])
        for yes_id, leg in leg_of.items():
            bid, ask, bsz, asz = books.top_floats(yes_id)
            row = (bid, ask, bsz, asz, books.is_clean(yes_id))
            prev = last.get(yes_id)
            if prev is None and all(math.isnan(x) for x in row[:4]):
                continue
            if prev is None or not _same(prev, row):
                last[yes_id] = row
                yield (t, leg, *row)


def _same(a: tuple, b: tuple) -> bool:
    return all((x == y) or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))
               for x, y in zip(a, b))


def write_tob_csv(rows: Iterable[tuple], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    n = 0
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TOB_COLUMNS)
        for r in rows:
            w.writerow([r[0], r[1], *("" if isinstance(x, float) and math.isnan(x) else repr(float(x)) for x in r[2:6]),
                        int(bool(r[6]))])
            n += 1
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return n


def read_tob_csv(path: Path) -> Any:
    """Load an exported TOB table as a pandas DataFrame (lazy pandas import)."""
    import pandas as pd

    df = pd.read_csv(path, dtype={"leg": str})
    df["clean"] = df["clean"].astype(bool)
    return df


def export_tob(group: str, basket_ids: Sequence[str] | None = None, *, root: Path = HIST_ROOT,
               cfg: MarketsConfig | None = None) -> dict[str, Path]:
    cfg = cfg or load_markets()
    sessions = list_sessions(group, root)
    recorded = {b for s in sessions for b in (s.get("baskets") or {})}
    out = {}
    for bid in basket_ids or sorted(recorded):
        basket = get_basket(bid, cfg)
        path = Path(root) / group / "tob" / f"{bid}.csv.gz"
        n = write_tob_csv(tob_rows(replay_frames(group, root=root), basket), path)
        logger.info("%s: %d top-of-book rows -> %s", bid, n, path)
        out[bid] = path
    return out


def status(group: str, root: Path = HIST_ROOT) -> dict:
    sessions = list_sessions(group, root)

    def ts(s: str | None) -> float | None:
        return datetime.fromisoformat(s).timestamp() if s else None

    hours = 0.0
    gaps = []
    prev_end = None
    for s in sessions:
        a, b = ts(s.get("started")), ts(s.get("ended"))
        if a and b:
            hours += (b - a) / 3600
        if prev_end and a:
            gaps.append(round(a - prev_end, 1))
        prev_end = b or prev_end
    return {
        "group": group, "sessions": len(sessions), "hours_recorded": round(hours, 3),
        "first": sessions[0]["started"] if sessions else None,
        "last": (sessions[-1].get("ended") or sessions[-1]["started"]) if sessions else None,
        "gaps_s": gaps, "frames": sum(int(s.get("n_frames") or 0) for s in sessions),
        "reconnects": sum(int(s.get("reconnects") or 0) for s in sessions),
        "dropped": sum(int(s.get("n_dropped") or 0) for s in sessions),
        "bytes": sum(int(f.get("bytes") or 0) for s in sessions for f in s.get("files", [])),
        "open_sessions": [s["session_id"] for s in sessions if not s.get("ended")],
    }


# --------------------------------------------------------------------------- price-history backfill
def _chunks(start: int, end: int, days: int = CHUNK_DAYS) -> list[tuple[int, int]]:
    step = days * 86400
    return [(a, min(a + step, end)) for a in range(int(start), int(end), step)]


def _created_ts(basket: Basket) -> int | None:
    stamps = [l.created_at for l in basket.legs if l.created_at]
    if not stamps:
        return None
    return int(min(datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps).timestamp())


def fetch_history(client: ClobRestClient, token_id: str, start: int, end: int, fidelity: int) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for a, b in _chunks(start, end):
        pts = client.get_prices_history(token_id, start_ts=a, end_ts=b, fidelity=fidelity)
        if not pts and fidelity < 720:
            pts = client.get_prices_history(token_id, start_ts=a, end_ts=b, fidelity=720)
        out.extend(pts)
    return out


def backfill(basket_id: str, *, end_ts: int | None = None, recent_days: float = 30, recent_fidelity: int = 1,
             full_history: bool = True, full_fidelity: int = 60, start_ts: int | None = None,
             root: Path = PRICES_ROOT, cfg: MarketsConfig | None = None, client: ClobRestClient | None = None) -> Path:
    """1-minute history for the last ``recent_days`` + hourly history back to market creation."""
    cfg = cfg or load_markets()
    basket = get_basket(basket_id, cfg)
    if basket.synthetic:
        raise ValueError("synthetic baskets have no price history")
    client = client or ClobRestClient()
    end = int(end_ts if end_ts is not None else time.time())
    recent_start = int(end - recent_days * 86400)
    start = int(start_ts if start_ts is not None else (_created_ts(basket) or end - 365 * 86400))
    out_dir = Path(root) / basket_id
    out_dir.mkdir(parents=True, exist_ok=True)
    legs_meta = {}
    for leg in basket.legs:
        rows: dict[int, tuple[float, int]] = {}
        if full_history and start < recent_start:
            for t, p in fetch_history(client, leg.yes_token_id, start, recent_start, full_fidelity):
                rows[t] = (p, full_fidelity * 60)
        for t, p in fetch_history(client, leg.yes_token_id, max(start, recent_start), end, recent_fidelity):
            rows[t] = (p, recent_fidelity * 60)
        path = out_dir / f"{leg.leg_id}.csv.gz"
        fd, tmp = tempfile.mkstemp(dir=out_dir, prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "p", "fidelity_s"])
            for t in sorted(rows):
                w.writerow([t, repr(rows[t][0]), rows[t][1]])
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        ts_sorted = sorted(rows)
        legs_meta[leg.leg_id] = {"yes_token_id": leg.yes_token_id, "label": leg.label, "rows": len(rows),
                                 "first_t": ts_sorted[0] if rows else None, "last_t": ts_sorted[-1] if rows else None,
                                 "sha256": sha256_file(path)}
        logger.info("%s/%s: %d points", basket_id, leg.leg_id, len(rows))
    _atomic_write_json(out_dir / MANIFEST, {
        "basket_id": basket_id, "synthetic": False, "fetched_at": utc_iso(),
        "endpoint": "GET https://clob.polymarket.com/prices-history",
        "params": {"start_ts": start, "end_ts": end, "recent_start_ts": recent_start, "recent_fidelity_min": recent_fidelity,
                   "full_history": full_history, "full_fidelity_min": full_fidelity, "chunk_days": CHUNK_DAYS},
        "legs": legs_meta, "semantics_note": PRICES_SEMANTICS,
    })
    return out_dir


def load_prices_history(basket_id: str, root: Path = PRICES_ROOT) -> Any:
    """Long DataFrame ``(t, leg, p, fidelity_s)`` for a backfilled basket (lazy pandas)."""
    import pandas as pd

    d = Path(root) / basket_id
    with open(d / MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)
    frames = []
    for leg_id in man["legs"]:
        df = pd.read_csv(d / f"{leg_id}.csv.gz")
        df.insert(1, "leg", leg_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["t", "leg", "p", "fidelity_s"])


# --------------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.recorder", description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--group", default="main")
    r.add_argument("--baskets", required=True)
    r.add_argument("--mode", choices=("ws", "poll"), default="ws")
    r.add_argument("--duration", default=None)
    r.add_argument("--poll-interval", type=float, default=1.0)
    r.add_argument("--subscribe-no", action="store_true")
    b = sub.add_parser("backfill")
    b.add_argument("--basket", action="append")
    b.add_argument("--all", action="store_true")
    b.add_argument("--recent-days", type=float, default=30)
    b.add_argument("--no-full-history", action="store_true")
    b.add_argument("--start")
    b.add_argument("--end")
    s = sub.add_parser("status")
    s.add_argument("--group", default="main")
    e = sub.add_parser("export-tob")
    e.add_argument("--group", default="main")
    e.add_argument("--baskets")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfg = load_markets(args.config) if args.config else load_markets()

    if args.cmd == "record":
        sess = asyncio.run(record(args.group, args.baskets.split(","), mode=args.mode,
                                  duration_s=parse_duration(args.duration) if args.duration else None,
                                  poll_interval_s=args.poll_interval, root=args.root or HIST_ROOT, cfg=cfg,
                                  subscribe_no=args.subscribe_no))
        print(json.dumps({k: sess[k] for k in ("session_id", "started", "ended", "n_frames", "n_dropped", "reconnects")}))
        return 0
    if args.cmd == "backfill":
        ids = args.basket or []
        if args.all:
            ids = [k for k, v in cfg.baskets.items() if v.is_resolved and not v.synthetic]
        to_ts = lambda s: int(datetime.fromisoformat(s.replace("Z", "+00:00")).replace(
            tzinfo=datetime.fromisoformat(s.replace("Z", "+00:00")).tzinfo or timezone.utc).timestamp())
        for bid in ids:
            print(backfill(bid, recent_days=args.recent_days, full_history=not args.no_full_history,
                           start_ts=to_ts(args.start) if args.start else None,
                           end_ts=to_ts(args.end) if args.end else None,
                           root=args.root or PRICES_ROOT, cfg=cfg))
        return 0
    if args.cmd == "status":
        print(json.dumps(status(args.group, args.root or HIST_ROOT), indent=1))
        return 0
    paths = export_tob(args.group, args.baskets.split(",") if args.baskets else None, root=args.root or HIST_ROOT, cfg=cfg)
    for k, v in paths.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
