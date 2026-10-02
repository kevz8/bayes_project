"""Recorder: raw format, rotation, manifests, replay, TOB export, backfill (all offline)."""
from __future__ import annotations

import asyncio
import gzip
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from src import SYNTH_ROOT
from src.clob_client import ClobRestClient, DropQueue, RawRecord
from src.config import Basket, Leg, MarketsConfig, Defaults
from src.events import MetaEvent
from src.orderbook import BookManager
from src.recorder import (
    RawRecordWriter,
    backfill,
    export_tob,
    iter_raw_records,
    list_sessions,
    load_prices_history,
    parse_duration,
    read_tob_csv,
    record,
    replay_frames,
    save_session,
    sha256_file,
    status,
)
from tests.conftest import FakeSession

YES = ["1" * 77, "2" * 77]
NO = ["3" * 77, "4" * 77]
HOUR_NS = 3600 * 10**9
T0 = 1_790_900_000 * 10**9  # 2026-10-02 ~ 00:13 UTC


def _cfg() -> MarketsConfig:
    legs = tuple(Leg(leg_id=f"leg{i}", label=f"Leg {i}", yes_token_id=YES[i], no_token_id=NO[i],
                     created_at="2026-09-01T00:00:00Z") for i in range(2))
    b = Basket(basket_id="b2", title="B", legs=legs, status="active", neg_risk=True)
    return MarketsConfig(defaults=Defaults(), baskets={"b2": b})


def _book(asset: str, bid: str, ask: str, ts: int) -> str:
    return json.dumps({"event_type": "book", "asset_id": asset, "market": "0xm", "timestamp": str(ts),
                       "bids": [{"price": bid, "size": "100"}], "asks": [{"price": ask, "size": "50"}]})


def _pc(asset: str, price: str, size: str, side: str, ts: int) -> str:
    return json.dumps({"event_type": "price_change", "market": "0xm", "timestamp": str(ts),
                       "price_changes": [{"asset_id": asset, "price": price, "size": size, "side": side}]})


def test_parse_duration():
    assert parse_duration("118m") == 7080 and parse_duration("2h") == 7200 and parse_duration("90") == 90
    with pytest.raises(ValueError):
        parse_duration("abc")


def test_encode_splices_verbatim_and_handles_newlines():
    t, line = RawRecordWriter.encode(RawRecord(5, "ws", 2, '{"a": 1}'))
    assert line == '{"t":5,"src":"ws","conn":2,"msg":{"a": 1}}' and json.loads(line)["msg"] == {"a": 1}
    _, line2 = RawRecordWriter.encode(RawRecord(5, "ws", 2, '{"a":\n1}'))
    assert "\n" not in line2 and json.loads(line2)["msg"] == {"a": 1}
    _, meta = RawRecordWriter.encode(MetaEvent("gap", {"dropped": 3}, 7))
    assert json.loads(meta) == {"t": 7, "src": "meta", "kind": "gap", "data": {"dropped": 3}}


def test_hourly_rotation_and_sha(tmp_path):
    w = RawRecordWriter(tmp_path, "sid")
    w.write([RawRecord(T0, "ws", 1, "{}"), RawRecord(T0 + HOUR_NS, "ws", 1, "{}"), RawRecord(T0 + HOUR_NS + 1, "ws", 1, "{}")])
    infos = w.close()
    assert len(infos) == 2 and sorted(i.lines for i in infos) == [1, 2]
    for i in infos:
        assert i.sha256 == sha256_file(tmp_path / i.path)


def test_truncated_gzip_tail_is_tolerated(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt") as fh:
        for i in range(200):
            fh.write(json.dumps({"t": i, "src": "meta", "kind": "k", "data": {}}) + "\n")
    data = p.read_bytes()
    p.write_bytes(data[: len(data) - 15])
    recs = list(iter_raw_records([p]))
    assert 0 < len(recs) <= 200


def _write_session(root: Path, group: str, sid: str, t0: int, lines: list[dict]) -> None:
    gdir = root / group
    w = RawRecordWriter(gdir, sid)
    w.write(lines)
    files = [vars(f) for f in w.close()]
    save_session(gdir, {"session_id": sid, "started": f"2026-10-02T00:{len(sid):02d}:00+00:00",
                        "ended": f"2026-10-02T00:{len(sid) + 1:02d}:00+00:00", "mode": "ws",
                        "assets": YES, "baskets": {"b2": YES}, "n_frames": len(lines), "n_dropped": 0,
                        "reconnects": 0, "files": files, "code_version": "test", "notes": ""})


def _stopgap_lines(t0: int) -> list[dict]:
    return [
        {"t": t0, "src": "meta", "kind": "session_start", "data": {}},
        {"t": t0 + 1, "src": "ws", "conn": 1, "msg": json.loads(_book(YES[0], "0.60", "0.62", 1000))},
        {"t": t0 + 2, "src": "rest_book", "conn": 1, "msg": {"market": "0xm", "asset_id": YES[1], "timestamp": "1001",
                                                             "bids": [{"price": "0.37", "size": "10"}],
                                                             "asks": [{"price": "0.39", "size": "20"}]}},
        {"t": t0 + 3, "src": "ws", "conn": 1, "msg": json.loads(_pc(YES[0], "0.61", "40", "BUY", 1002))},
        {"t": t0 + 4, "src": "ws", "conn": 1, "msg": json.loads(_pc(YES[1], "0.39", "0", "SELL", 1003))},
    ]


def test_replay_and_export_tob(tmp_path):
    _write_session(tmp_path, "g", "s1", T0, _stopgap_lines(T0))
    _write_session(tmp_path, "g", "s22", T0 + HOUR_NS, _stopgap_lines(T0 + HOUR_NS))
    frames = list(replay_frames("g", root=tmp_path))
    kinds = [e.kind for _, evs in frames for e in evs if isinstance(e, MetaEvent) and e.kind.startswith("replay_")]
    assert kinds == ["replay_start", "replay_end"] * 2

    bm = BookManager(YES)
    for _, evs in list(replay_frames("g", root=tmp_path, session_id="s1")):
        bm.apply([e for e in evs if not isinstance(e, MetaEvent)])
    assert bm.top(YES[0]) == (0.61, 0.62) and math.isnan(bm.top(YES[1])[1])

    paths = export_tob("g", ["b2"], root=tmp_path, cfg=_cfg())
    df = read_tob_csv(paths["b2"])
    assert list(df.columns) == ["t_ns", "leg", "bid", "ask", "bid_sz", "ask_sz", "clean"]
    leg0 = df[df.leg == "leg0"]
    assert list(leg0.bid.iloc[:2]) == [0.60, 0.61]
    ends = df[~df.clean]
    assert len(ends) == 4 and ends.bid.isna().all()  # one marker per leg per session end
    st = status("g", root=tmp_path)
    assert st["sessions"] == 2 and len(st["gaps_s"]) == 1


class FakeFeed:
    def __init__(self, assets):
        self.assets = assets
        self.q = DropQueue()
        self.stats = type("S", (), {"reconnects": 1})()
        self._stop = asyncio.Event()

    def subscribe_raw(self, **kw):
        return self.q

    def add_listener(self, cb):
        pass

    async def run(self):
        self.q.offer(RawRecord(T0 + 1, "ws", 1, _book(YES[0], "0.5", "0.52", 1)), 100)
        self.q.offer(MetaEvent("connect", {"conn": 1}, T0 + 2), 10)
        self.q.offer(RawRecord(T0 + 3, "ws", 1, _pc(YES[0], "0.51", "5", "BUY", 2)), 100)
        await self._stop.wait()

    async def stop(self):
        self._stop.set()


def test_record_session_with_fake_feed(tmp_path):
    clock = iter(range(T0, T0 + 10**12, 10**6))
    sess = asyncio.run(record("g", ["b2"], root=tmp_path, cfg=_cfg(), duration_s=0.2,
                              feed_factory=FakeFeed, clock_ns=lambda: next(clock)))
    assert sess["n_frames"] == 2 and sess["ended"] and sess["reconnects"] == 1
    (s,) = list_sessions("g", tmp_path)
    assert s["baskets"] == {"b2": YES} and s["files"][0]["sha256"] == sha256_file(tmp_path / "g" / s["files"][0]["path"])
    recs = list(iter_raw_records([tmp_path / "g" / f["path"] for f in s["files"]]))
    assert [r["src"] for r in recs] == ["meta", "ws", "meta", "ws", "meta"]
    assert recs[-1]["kind"] == "stop"


def test_record_refusals(tmp_path):
    cfg = _cfg()
    unresolved = replace(cfg.baskets["b2"], legs=(), status="unresolved")
    cfg2 = replace(cfg, baskets={"b2": unresolved})
    with pytest.raises(Exception):
        asyncio.run(record("g", ["b2"], root=tmp_path, cfg=cfg2, feed_factory=FakeFeed))
    synth = replace(cfg.baskets["b2"], synthetic=True)
    with pytest.raises(ValueError):
        asyncio.run(record("g", ["b2"], root=tmp_path, cfg=replace(cfg, baskets={"b2": synth}), feed_factory=FakeFeed))
    with pytest.raises(ValueError):
        asyncio.run(record("g", ["b2"], root=SYNTH_ROOT, cfg=cfg, feed_factory=FakeFeed))


def test_backfill_chunks_merges_and_retries(tmp_path):
    end = 1_790_900_000
    calls = []

    def history(method, url, params, body):
        calls.append(dict(params))
        a, b, fid = params["startTs"], params["endTs"], params["fidelity"]
        if params["market"] == YES[1] and fid == 1:
            return {"history": []}  # forces the fidelity-720 retry
        step = fid * 60
        return {"history": [{"t": t, "p": 0.5} for t in range(a - a % step, b, step * 500)]}

    s = FakeSession({("GET", "/prices-history"): history})
    client = ClobRestClient(session=s, min_interval_s=0)
    out = backfill("b2", end_ts=end, recent_days=30, start_ts=end - 40 * 86400, root=tmp_path, cfg=_cfg(), client=client)
    leg0 = [c for c in calls if c["market"] == YES[0]]
    recent = [c for c in leg0 if c["fidelity"] == 1]
    assert len(recent) == 3 and all(c["endTs"] - c["startTs"] <= 14 * 86400 for c in calls)
    assert any(c["fidelity"] == 60 for c in leg0)
    assert any(c["market"] == YES[1] and c["fidelity"] == 720 for c in calls)
    man = json.loads((out / "manifest.json").read_text())
    assert man["synthetic"] is False and "NOT executable" in man["semantics_note"]
    df = load_prices_history("b2", root=tmp_path)
    assert set(df.leg) == {"leg0", "leg1"} and df.groupby("leg").t.apply(lambda x: x.is_monotonic_increasing).all()
    assert not df.duplicated(["leg", "t"]).any()
