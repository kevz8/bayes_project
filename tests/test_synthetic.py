"""Tests for ``src.synthetic`` - the seeded SYNTHETIC negRisk basket generator.

The generator is only useful if (a) it is honest about being synthetic, (b) its wire output
is indistinguishable in *shape* from Polymarket's market channel so the real parser and
order book consume it unchanged, and (c) its statistics are the ones its docstring claims.
The default run uses 2-hour paths (about 1 s each); the 3-day statistical checks are ``slow``.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import inspect
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src
from src.clob_client import ParseStats, parse_frame, parse_message
from src.config import config_from_dict, get_basket
from src.events import (
    ASK,
    BID,
    PRICE_SCALE,
    TOB_COLUMNS,
    BookEvent,
    MarketResolvedEvent,
    PriceChangeEvent,
    TickSizeEvent,
)
from src.orderbook import BookManager
from src.synthetic import (
    BOP_LABELS,
    SyntheticParams,
    build_ladder,
    calibration_summary,
    flb,
    generate_pairing_demo,
    iter_raw_lines,
    iter_wire_messages,
    main,
    select_tick,
    simulate_latent,
    synthetic_basket,
    synthetic_tob,
    write_synthetic_recording,
)

# A leg priced near 0.04 makes the tick flip between 0.01 and 0.001 inside two hours, and a
# high gap rate injects recording gaps; seed 1 yields 10 tick changes and 3 gaps.
SHORT = dict(name="SYNTHETIC_short", duration_s=7200.0, seed=1, gap_prob_per_hour=1.5,
             pi0=(0.55, 0.30, 0.083, 0.037, 0.03))
FINE, COARSE = 1_000, 10_000  # ticks in micro-units
DEC = re.compile(r"^(0|[1-9]\d*)(\.\d*[1-9])?$")  # wire decimal: no exponent, no trailing zeros
MS = re.compile(r"^\d{13}$")


def _start_ns(p: SyntheticParams) -> int:
    return pd.Timestamp(p.start).value


def _nan_none(x: float) -> float | None:
    return None if x != x else x


# --------------------------------------------------------------------------- shared runs
@pytest.fixture(scope="module")
def short() -> dict[str, Any]:
    p = SyntheticParams(**SHORT)
    truth = simulate_latent(p)
    return {"p": p, "truth": truth, "basket": synthetic_basket(p), "raw": list(iter_raw_lines(p, truth)),
            "msgs": list(iter_wire_messages(p, truth)), "tob": synthetic_tob(p, truth)}


def _replay(p: SyntheticParams, raw: list[dict], truth: Any) -> dict[str, Any]:
    """Replay raw lines through the real parser and ``BookManager`` the way a consumer of a
    recording would (a ``gap`` meta line marks every book dirty), recording each leg's
    observable top-of-book state after every message plus ladder / tick violations.

    The grid is checked once all messages sharing a receive time are applied: on a switch
    back to the 0.01 tick the ``tick_size_change`` precedes the ``price_change`` (same
    timestamp) that cancels the levels quoted on the 0.001 grid.
    """
    basket = synthetic_basket(p)
    leg_of = {leg.yes_token_id: (i, leg.leg_id) for i, leg in enumerate(basket.legs)}
    resyncs: list[tuple[str, str]] = []
    bm = BookManager(basket.yes_ids, desync_tolerance=1, on_resync_needed=lambda a, r: resyncs.append((a, r)))
    start_ns = _start_ns(p)
    lag = next(r["t"] for r in raw if r["src"] != "meta") - start_ns  # first books are at t = 0
    states: dict[str, list[tuple[int, tuple]]] = {leg_id: [] for _, leg_id in leg_of.values()}
    off_grid, rule, ladder, tick_chain, ticks_seen = [], [], [], [], Counter()
    same_t: set[str] = set()
    for k, line in enumerate(raw):
        t = line["t"]
        if line["src"] == "meta":
            touched: set[str] = set(leg_of) if line["kind"] == "gap" else set()
            if touched:
                bm.mark_dirty(reason="gap")
        else:
            events = parse_message(line["msg"], t)
            tick_chain += [ev for ev in events
                           if isinstance(ev, TickSizeEvent) and bm.books[ev.asset_id].tick != ev.old_tick]
            bm.apply(events)
            touched = {getattr(ev, "asset_id", None) for ev in events}
            touched |= {ch.asset_id for ev in events if isinstance(ev, PriceChangeEvent) for ch in ev.changes}
        touched &= set(leg_of)
        for aid in touched:
            i, leg_id = leg_of[aid]
            book = bm.books[aid]
            st = tuple(_nan_none(x) for x in bm.top_floats(aid)) + (bm.is_clean(aid),)
            if not states[leg_id] or states[leg_id][-1][1] != st:
                states[leg_id].append((t, st))
            bp, _ = bm.depth(aid, BID)
            ap, _ = bm.depth(aid, ASK)
            if np.any(np.diff(bp) >= 0) or np.any(np.diff(ap) <= 0) or (bp.size and ap.size and bp[0] >= ap[0]):
                ladder.append((t, aid))
            if line["src"] != "meta" and line["msg"]["event_type"] == "price_change":
                q = float(truth.q_fair[(t - start_ns - lag) // int(p.dt_s * 1e9), i])
                must_fine = q < 0.04 or q > 0.96
                must_coarse = 0.04 + p.tick_hysteresis <= q <= 0.96 - p.tick_hysteresis
                if (must_fine and book.tick != FINE) or (must_coarse and book.tick != COARSE):
                    rule.append((t, aid, q, book.tick))
        same_t |= touched
        if k + 1 == len(raw) or raw[k + 1]["t"] != t:
            for aid in same_t:
                book = bm.books[aid]
                ticks_seen[(leg_of[aid][0], book.tick)] += 1
                if any(px % book.tick for px in (*book.bids, *book.asks)):
                    off_grid.append((t, aid, book.tick))
            same_t.clear()
    return {"states": states, "stats": bm.stats, "resyncs": resyncs, "off_grid": off_grid, "rule": rule,
            "ladder": ladder, "tick_chain": tick_chain, "ticks_seen": ticks_seen, "bm": bm, "lag": lag}


@pytest.fixture(scope="module")
def replay(short: dict[str, Any]) -> dict[str, Any]:
    return _replay(short["p"], short["raw"], short["truth"])


def _tob_states(tob: pd.DataFrame) -> dict[str, list[tuple[int, tuple]]]:
    out: dict[str, list[tuple[int, tuple]]] = {}
    for r in tob.itertuples(index=False):
        st = (_nan_none(r.bid), _nan_none(r.ask), _nan_none(r.bid_sz), _nan_none(r.ask_sz), bool(r.clean))
        out.setdefault(r.leg, []).append((int(r.t_ns), st))
    return out


# --------------------------------------------------------------------------- parameters
def test_params_prefix_defaults_and_validation() -> None:
    p = SyntheticParams(name="foo")
    assert p.name == "SYNTHETIC_foo"
    assert p.pi0 == pytest.approx((0.60, 0.30, 0.06, 0.03, 0.01), abs=1e-15)
    w = np.arange(1, 4) ** -1.5
    assert SyntheticParams(n_legs=3).pi0 == pytest.approx(tuple(w / w.sum()), abs=1e-15)
    assert SyntheticParams(n_legs=2, pi0=(3, 1)).pi0 == (3.0, 1.0)  # stored raw, normalised when used
    for bad in (dict(n_legs=1), dict(n_legs=2.5), dict(n_legs=3, pi0=(0.5, 0.5)), dict(n_legs=2, pi0=(1.0, 0.0)),
                dict(n_legs=2, leg_labels=("a",)), dict(tick=0.003), dict(fine_tick=0.02), dict(dt_s=0.0),
                dict(gap_len_s=(0.0, 1.0)), dict(gap_len_s=(5.0, 1.0)), dict(thin_top_prob=1.5),
                dict(shock_rate_per_hour=-1.0), dict(start="not a time")):
        with pytest.raises(ValueError):
            SyntheticParams(**bad)


def test_params_dict_round_trip_and_unknown_keys() -> None:
    p = SyntheticParams(name="SYNTHETIC_rt", n_legs=4, pi0=(0.4, 0.3, 0.2, 0.1), leg_labels=BOP_LABELS,
                        seed=3, gap_len_s=(10.0, 20.0))
    d = p.to_dict()
    assert isinstance(d["pi0"], list) and isinstance(d["gap_len_s"], list)
    assert SyntheticParams.from_dict(d, name=p.name) == p
    assert SyntheticParams.from_dict(json.loads(json.dumps(d)), name=p.name) == p  # survives JSON
    q = SyntheticParams.from_dict(d, name="other", seed=99)  # name/seed arguments win over the mapping
    assert (q.name, q.seed) == ("SYNTHETIC_other", 99)
    assert SyntheticParams.from_dict({}, name="SYNTHETIC_x").seed == SyntheticParams().seed
    with pytest.raises(ValueError, match="dislocation_half_life"):
        SyntheticParams.from_dict({"n_legs": 3, "dislocation_half_life": 60}, name="SYNTHETIC_typo")


# --------------------------------------------------------------------------- latent model
def test_latent_probabilities_sum_to_one_every_step(short: dict[str, Any]) -> None:
    p, tr = short["p"], short["truth"]
    T = int(p.duration_s / p.dt_s) + 1
    assert tr.pi.shape == tr.q_fair.shape == tr.disloc.shape == (T, p.n_legs)
    np.testing.assert_array_equal(tr.t_s, np.arange(T) * p.dt_s)
    assert np.max(np.abs(tr.pi.sum(axis=1) - 1.0)) < 1e-12
    assert np.all(tr.pi > 0)
    np.testing.assert_allclose(tr.pi[0], np.asarray(p.pi0) / sum(p.pi0), rtol=1e-12)
    np.testing.assert_array_equal(tr.basket_dislocation, tr.disloc.sum(axis=1))
    assert 0 <= tr.winner < p.n_legs
    assert len(tr.gaps) == 3 and all(0 < a < b <= p.duration_s for a, b in tr.gaps)
    assert all(b0 < a1 for (_, b0), (a1, _) in zip(tr.gaps, tr.gaps[1:]))  # merged, sorted, disjoint


def test_regime_breaks_are_permanent_steps() -> None:
    p = SyntheticParams(duration_s=86400.0, regime_break_prob_per_day=5.0, seed=2)
    tr = simulate_latent(p)
    assert tr.break_t.size > 0
    expected = np.zeros_like(tr.regime)
    for t, s in zip(tr.break_t, tr.break_size):
        expected[tr.t_s >= t] += s
    np.testing.assert_allclose(tr.regime, expected, atol=1e-12)
    assert set(np.abs(tr.break_size)) == {p.regime_break_size}


def test_dislocation_is_ar1_with_configured_sd_and_half_life() -> None:
    """Without sweeps each leg's dislocation is a stationary AR(1) with sd ``disloc_sigma``;
    with sweeps the basket dislocation still decays with ``disloc_half_life_s`` (OLS on 1 s)."""
    calm = simulate_latent(SyntheticParams(duration_s=86400.0, shock_rate_per_hour=0.0))
    np.testing.assert_allclose(calm.disloc.std(axis=0), 0.002, rtol=0.12)
    # sweep process on a cheap 60-day, 1-minute grid: ~1440 sweeps of mean size 0.08, either sign
    sweeps = simulate_latent(SyntheticParams(duration_s=60 * 86400.0, dt_s=60.0, seed=0))
    assert sweeps.shock_t.size == pytest.approx(60 * 24, rel=0.1) and np.all(np.diff(sweeps.shock_t) >= 0)
    assert np.mean(np.abs(sweeps.shock_size)) == pytest.approx(0.08, rel=0.05)
    assert np.mean(sweeps.shock_size > 0) == pytest.approx(0.5, abs=0.05)
    p = SyntheticParams(duration_s=86400.0)
    tr = simulate_latent(p)
    d = tr.basket_dislocation
    x, y = d[:-1] - d.mean(), d[1:] - d.mean()
    phi = float(x @ y / (x @ x))
    half_life = -math.log(2.0) / math.log(phi) * p.dt_s
    assert half_life == pytest.approx(p.disloc_half_life_s, rel=0.15)


def test_rng_streams_are_independent() -> None:
    """Changing gaps or shocks must not move the latent probabilities (separate SeedSequence streams)."""
    p = SyntheticParams(duration_s=3600.0, seed=4)
    base = simulate_latent(p)
    more_gaps = simulate_latent(dataclasses.replace(p, gap_prob_per_hour=10.0))
    np.testing.assert_array_equal(base.pi, more_gaps.pi)
    np.testing.assert_array_equal(base.q_fair, more_gaps.q_fair)
    assert base.gaps != more_gaps.gaps
    no_shocks = simulate_latent(dataclasses.replace(p, shock_rate_per_hour=0.0))
    np.testing.assert_array_equal(base.pi, no_shocks.pi)
    assert base.winner == no_shocks.winner


def test_flb_distortion() -> None:
    # hand-computed: sqrt(.25) / (sqrt(.25) + sqrt(.75)) = 0.5 / 1.3660254 = 0.3660254
    assert float(flb(0.25, 0.5)) == pytest.approx(0.36602540378, abs=1e-10)
    x = np.array([0.0, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0])
    np.testing.assert_allclose(flb(x, 1.0), x, atol=1e-15)
    np.testing.assert_allclose(flb(1.0 - x, 0.9), 1.0 - flb(x, 0.9), atol=1e-12)
    g = flb(x, 0.9)
    assert g[1] > 0.01 and g[5] < 0.99 and g[3] == pytest.approx(0.5)  # long shots over-, favourites under-priced
    assert np.all(np.diff(g) > 0)


# --------------------------------------------------------------------------- ticks and ladders
def test_select_tick_rule_with_hysteresis() -> None:
    p = SyntheticParams()
    assert [select_tick(q, None, p) for q in (0.039, 0.04, 0.5, 0.96, 0.961, 0.001)] == \
        [0.001, 0.01, 0.01, 0.01, 0.001, 0.001]
    # a fine-tick leg returns to 0.01 only once q is tick_hysteresis (0.005) inside the band
    assert select_tick(0.042, 0.001, p) == 0.001
    assert select_tick(0.042, 0.01, p) == 0.01
    assert select_tick(0.046, 0.001, p) == 0.01
    assert select_tick(0.958, 0.001, p) == 0.001
    assert select_tick(0.954, 0.001, p) == 0.01


def _check_ladder(bids: list, asks: list, tick: float, levels: int) -> None:
    tick_i = round(tick * PRICE_SCALE)
    for side in (bids, asks):
        assert len(side) <= levels
        for px, sz in side:
            assert round(px * PRICE_SCALE) % tick_i == 0 and 0.0 < px < 1.0
            assert sz >= 1.0 and round(sz, 2) == sz
    assert all(a[0] > b[0] for a, b in zip(bids, bids[1:]))  # bids descending, best first
    assert all(a[0] < b[0] for a, b in zip(asks, asks[1:]))  # asks ascending, best first
    if bids and asks:
        assert bids[0][0] < asks[0][0]


def test_build_ladder_hand_cases() -> None:
    p = SyntheticParams(depth_levels=4)
    rng = np.random.default_rng(0)
    bids, asks = build_ladder(0.503, 0.01, 1, rng, p)
    assert [b for b, _ in bids] == pytest.approx([0.50, 0.49, 0.48, 0.47])
    assert [a for a, _ in asks] == pytest.approx([0.51, 0.52, 0.53, 0.54])
    bids, asks = build_ladder(0.503, 0.01, 3, rng, p)  # bid = floor(50.3 - 1), ask = ceil(50.3 + 1)
    assert (bids[0][0], asks[0][0]) == pytest.approx((0.49, 0.52))
    bids, asks = build_ladder(0.0025, 0.001, 1, rng, p)  # only two fine ticks below fair value
    assert [b for b, _ in bids] == pytest.approx([0.002, 0.001]) and asks[0][0] == pytest.approx(0.003)
    bids, asks = build_ladder(0.0004, 0.001, 1, rng, p)  # dust: no bid side at all
    assert bids == [] and asks[0][0] == pytest.approx(0.001)
    bids, asks = build_ladder(0.9996, 0.001, 1, rng, p)
    assert asks == [] and bids[0][0] == pytest.approx(0.999)


@pytest.mark.parametrize("q,tick,k", [(0.5, 0.01, 1), (0.3333, 0.01, 4), (0.97, 0.001, 2), (0.012, 0.001, 6),
                                      (-0.05, 0.001, 1), (1.2, 0.01, 2), (0.995, 0.01, 1)])
def test_build_ladder_invariants(q: float, tick: float, k: int) -> None:
    p = SyntheticParams()
    bids, asks = build_ladder(q, tick, k, np.random.default_rng(1), p)
    assert bids or asks
    _check_ladder(bids, asks, tick, p.depth_levels)


# --------------------------------------------------------------------------- labelling / config
def test_synthetic_basket_is_labelled() -> None:
    p = SyntheticParams(name="SYNTHETIC_bop", n_legs=4, pi0=(0.4, 0.3, 0.2, 0.1), leg_labels=BOP_LABELS, seed=5)
    b = synthetic_basket(p)
    assert b.basket_id == "SYNTHETIC_bop" and b.synthetic and b.neg_risk and b.is_complete_partition
    assert "SYNTHETIC" in b.title and "SYNTHETIC" in b.notes
    assert [leg.leg_id for leg in b.legs] == ["d-senate-d-house", "r-senate-d-house", "d-senate-r-house",
                                              "r-senate-r-house"]
    assert all(t.startswith("SYNTHETIC-") for t in (*b.yes_ids, *b.no_ids))
    assert len(set(b.yes_ids) | set(b.no_ids)) == 8
    assert all(leg.fee.rate == p.fee_rate for leg in b.legs)
    assert b.end_date == "2026-01-04T00:00:00Z"
    dup = synthetic_basket(SyntheticParams(n_legs=2, leg_labels=("Same", "Same")))
    assert [leg.leg_id for leg in dup.legs] == ["same", "same-1"]


def test_get_basket_through_markets_json() -> None:
    b = get_basket("SYNTHETIC_demo5")  # config/markets.json synthetic_baskets (read-only)
    assert b.synthetic and b.basket_id == "SYNTHETIC_demo5" and b.n_legs == 5
    assert b.yes_ids[0].startswith("SYNTHETIC-") and b.yes_ids[0] == "SYNTHETIC-7-0-YES"
    cfg = config_from_dict({"synthetic_baskets": [
        {"basket_id": "SYNTHETIC_tiny", "synthetic": True, "seed": 3, "params": {"n_legs": 3}}]})
    assert get_basket("SYNTHETIC_tiny", cfg).yes_ids == ("SYNTHETIC-3-0-YES", "SYNTHETIC-3-1-YES", "SYNTHETIC-3-2-YES")


# --------------------------------------------------------------------------- wire format
def test_wire_messages_shape(short: dict[str, Any]) -> None:
    p, tr, b, msgs = short["p"], short["truth"], short["basket"], short["msgs"]
    n = p.n_legs
    yes, no = list(b.yes_ids), list(b.no_ids)
    markets = {leg.yes_token_id: leg.condition_id for leg in b.legs}
    kinds = Counter(m["event_type"] for _, m in msgs)
    assert set(kinds) == {"book", "price_change", "tick_size_change", "market_resolved"}
    assert kinds["tick_size_change"] >= 2 and kinds["market_resolved"] == n
    assert [m["event_type"] for _, m in msgs[:n]] == ["book"] * n and [m["asset_id"] for _, m in msgs[:n]] == yes

    def dec(s: Any, lo: float = 0.0, hi: float = 1.0) -> None:
        assert isinstance(s, str) and DEC.match(s), s
        assert lo <= float(s) <= hi

    prev_t, prev_ms = 0, 0
    for t, m in msgs:
        assert isinstance(t, int) and t >= prev_t
        assert MS.match(m["timestamp"]) and int(m["timestamp"]) >= prev_ms
        assert 0 <= t - int(m["timestamp"]) * 1_000_000 < 1_000_000_000  # received within 1 s of the stamp
        prev_t, prev_ms = t, int(m["timestamp"])
        et = m["event_type"]
        if et == "price_change":
            assert set(m) == {"event_type", "market", "price_changes", "timestamp"}
            assert m["price_changes"] and len({e["asset_id"] for e in m["price_changes"]}) == 1
            for e in m["price_changes"]:
                assert set(e) == {"asset_id", "price", "size", "side", "hash", "best_bid", "best_ask"}
                assert e["asset_id"] in markets and m["market"] == markets[e["asset_id"]]
                assert e["side"] in ("BUY", "SELL") and re.fullmatch(r"[0-9a-f]{40}", e["hash"])
                dec(e["price"], 0.0001, 0.9999)
                dec(e["size"], 0.0, 1e9)
                dec(e["best_bid"])
                dec(e["best_ask"])
                if e["best_bid"] != "0" and e["best_ask"] != "1":
                    assert float(e["best_bid"]) < float(e["best_ask"])
        elif et == "book":
            assert set(m) == {"event_type", "asset_id", "market", "bids", "asks", "timestamp", "hash", "tick_size"}
            assert m["market"] == markets[m["asset_id"]] and m["tick_size"] in ("0.01", "0.001")
            for lv in (*m["bids"], *m["asks"]):
                assert set(lv) == {"price", "size"}
                dec(lv["price"], 0.0001, 0.9999)
                dec(lv["size"], 1.0, 1e9)
            bp, ap = [float(x["price"]) for x in m["bids"]], [float(x["price"]) for x in m["asks"]]
            assert bp == sorted(bp) and ap == sorted(ap, reverse=True)  # wire order: best level last
            if bp and ap:
                assert bp[-1] < ap[-1]
        elif et == "tick_size_change":
            assert set(m) == {"event_type", "asset_id", "market", "old_tick_size", "new_tick_size", "timestamp"}
            assert {m["old_tick_size"], m["new_tick_size"]} == {"0.01", "0.001"}
        else:
            assert et == "market_resolved"
            assert {"id", "market", "assets_ids", "winning_asset_id", "winning_outcome", "event_message"} <= set(m)
            assert m["event_message"]["id"].startswith("SYNTHETIC-")
    resolved = [m for _, m in msgs[-n:]]
    assert all(m["event_type"] == "market_resolved" for m in resolved)
    assert [m["assets_ids"] for m in resolved] == [[y, x] for y, x in zip(yes, no)]
    assert [m["winning_outcome"] for m in resolved].count("Yes") == 1
    for i, m in enumerate(resolved):
        won = i == tr.winner
        assert m["winning_outcome"] == ("Yes" if won else "No")
        assert m["winning_asset_id"] == (yes[i] if won else no[i])
        assert int(m["timestamp"]) == (_start_ns(p) + int(p.duration_s * 1e9)) // 1_000_000


def test_snapshot_schedule_and_silent_gaps(short: dict[str, Any], replay: dict[str, Any]) -> None:
    """Books are re-sent at t = 0, every ``snapshot_interval_s`` outside gaps, and when a gap
    ends; nothing at all is emitted strictly inside a gap."""
    p, tr, msgs = short["p"], short["truth"], short["msgs"]
    t0 = _start_ns(p) + replay["lag"]
    rel = lambda t: (t - t0) / 1e9  # noqa: E731
    in_gap = lambda x: any(a < x < b for a, b in tr.gaps)  # noqa: E731
    assert not any(in_gap(rel(t)) for t, _ in msgs)
    expected = {0.0} | {b for _, b in tr.gaps if b < p.duration_s}
    expected |= {k * p.snapshot_interval_s for k in range(1, int(p.duration_s // p.snapshot_interval_s) + 1)
                 if k * p.snapshot_interval_s < p.duration_s
                 and not any(a <= k * p.snapshot_interval_s < b for a, b in tr.gaps)}
    books = Counter(t for t, m in msgs if m["event_type"] == "book")  # all legs share one receive time
    assert set(books.values()) == {p.n_legs}
    np.testing.assert_allclose(sorted(rel(t) for t in books), sorted(expected), rtol=0, atol=1e-6)
    gaps = [r for r in short["raw"] if r["src"] == "meta" and r["kind"] == "gap"]
    np.testing.assert_allclose([(rel(r["t"]), r["data"]["duration_s"]) for r in gaps],
                               [(a, b - a) for a, b in tr.gaps], rtol=0, atol=1e-6)


def test_every_wire_message_parses(short: dict[str, Any]) -> None:
    stats = ParseStats()
    expected = {"book": BookEvent, "price_change": PriceChangeEvent, "tick_size_change": TickSizeEvent,
                "market_resolved": MarketResolvedEvent}
    for t, m in short["msgs"]:
        events = parse_message(m, t, stats)
        assert len(events) == 1 and isinstance(events[0], expected[m["event_type"]])
        ev = events[0]
        assert ev.t_recv_ns == t and ev.ts_ms == int(m["timestamp"])
        if isinstance(ev, PriceChangeEvent):
            assert ev.schema == "v2" and len(ev.changes) == len(m["price_changes"])
            assert all(c.size is not None and c.price % FINE == 0 for c in ev.changes)
        elif isinstance(ev, BookEvent):
            assert ev.tick_size in (FINE, COARSE) and ev.bids and ev.asks
        elif isinstance(ev, TickSizeEvent):
            assert {ev.old_tick, ev.new_tick} == {FINE, COARSE}
        else:
            assert ev.winning_outcome in ("Yes", "No") and ev.winning_asset_id in ev.asset_ids
    assert stats.malformed == 0 and not stats.unknown and stats.events == len(short["msgs"])
    # The live socket batches messages in JSON arrays; the frame parser accepts them too.
    batch = [m for _, m in short["msgs"][:7]]
    assert len(parse_frame(json.dumps(batch), 0)) == 7


def test_replay_through_bookmanager_reproduces_tob(short: dict[str, Any], replay: dict[str, Any]) -> None:
    """The fast ``synthetic_tob`` table equals a replay of the wire stream through the real
    parser and order book, row for row and nanosecond for nanosecond, including the
    ``clean=False`` stretches of injected gaps. The server tops carried by every v2 entry
    agree with our book (no desync) and the book never crosses."""
    expected = _tob_states(short["tob"])
    assert set(expected) == set(replay["states"])
    for leg_id, rows in expected.items():
        assert replay["states"][leg_id] == rows, leg_id
    assert any(not st[-1] for rows in expected.values() for _, st in rows)  # the gaps were exercised
    st = replay["stats"]
    assert st["desync_mismatch"] == st["desync"] == st["crossed"] == st["stale_dropped"] == 0
    assert st["unknown_asset"] == 0 and replay["resyncs"] == []
    assert replay["bm"].all_clean()  # the last gap ended before resolution


def test_replay_with_gap_running_to_the_end() -> None:
    """Seed 3 has a gap that never ends: the books stay dirty through resolution."""
    p = SyntheticParams(**{**SHORT, "seed": 3})
    tr = simulate_latent(p)
    assert tr.gaps[-1][1] == p.duration_s
    rep = _replay(p, list(iter_raw_lines(p, tr)), tr)
    assert rep["states"] == _tob_states(synthetic_tob(p, tr))
    assert not rep["bm"].all_clean() and rep["stats"]["desync"] == 0 and not rep["off_grid"]


def test_prices_on_tick_grid_and_tick_rule(short: dict[str, Any], replay: dict[str, Any]) -> None:
    """Every resting level sits on the leg's current tick, and the tick is 0.001 whenever the
    fair value is outside [0.04, 0.96] and 0.01 when it is well inside (hysteresis band aside)."""
    assert replay["off_grid"] == []
    assert replay["rule"] == []
    assert replay["tick_chain"] == []  # old_tick_size always equals the tick the consumer holds
    seen = replay["ticks_seen"]
    assert {tick for (_, tick) in seen} == {FINE, COARSE}
    assert all(tick == COARSE for (i, tick) in seen if i == 0)  # the favourite never leaves 0.01
    tob = short["tob"]
    for col in ("bid", "ask"):
        micro = np.round(tob[col].dropna().to_numpy() * PRICE_SCALE).astype(np.int64)
        assert np.all(micro % FINE == 0) and np.all((micro > 0) & (micro < PRICE_SCALE))


def test_bid_below_ask_and_monotone_books(short: dict[str, Any], replay: dict[str, Any]) -> None:
    assert replay["ladder"] == []
    tob = short["tob"]
    both = tob.dropna(subset=["bid", "ask"])
    assert len(both) > 0.9 * len(tob) and np.all(both["bid"].to_numpy() < both["ask"].to_numpy())
    sizes = tob[["bid_sz", "ask_sz"]].to_numpy()
    assert np.all(np.isnan(sizes) | (sizes >= 1.0))
    assert np.array_equal(np.isnan(tob["bid"]), np.isnan(tob["bid_sz"]))


# --------------------------------------------------------------------------- tob / determinism
def test_synthetic_tob_schema(short: dict[str, Any]) -> None:
    tob, p, b = short["tob"], short["p"], short["basket"]
    assert tuple(tob.columns) == TOB_COLUMNS
    assert tob["t_ns"].dtype == np.int64 and tob["clean"].dtype == bool
    assert all(tob[c].dtype == np.float64 for c in ("bid", "ask", "bid_sz", "ask_sz"))
    assert tob.attrs["synthetic"] is True and tob.attrs["basket_id"] == p.name and tob.attrs["seed"] == p.seed
    assert np.all(np.diff(tob["t_ns"].to_numpy()) >= 0)
    assert tob["leg"].iloc[: p.n_legs].tolist() == [leg.leg_id for leg in b.legs]
    assert tob["t_ns"].iloc[: p.n_legs].nunique() == 1
    for _, g in tob.groupby("leg"):  # one row per change of a leg's observable state
        v = g[["bid", "ask", "bid_sz", "ask_sz", "clean"]].astype(float).fillna(-1.0).to_numpy()
        assert not np.any(np.all(v[1:] == v[:-1], axis=1))


def test_deterministic_given_seed() -> None:
    p = SyntheticParams(duration_s=1800.0, seed=21)
    a, b = list(iter_wire_messages(p)), list(iter_wire_messages(p))
    assert a == b and len(a) > 100
    pd.testing.assert_frame_equal(synthetic_tob(p), synthetic_tob(p))
    other = list(iter_wire_messages(dataclasses.replace(p, seed=22)))
    assert other != a
    assert not np.array_equal(simulate_latent(p).q_fair, simulate_latent(dataclasses.replace(p, seed=22)).q_fair)


# --------------------------------------------------------------------------- raw format / recording
def test_raw_lines_format(short: dict[str, Any]) -> None:
    raw, p = short["raw"], short["p"]
    first, last = raw[0], raw[-1]
    assert set(first) == {"t", "src", "kind", "data"} and first["src"] == "meta" and first["kind"] == "session_start"
    assert first["data"]["synthetic"] is True and first["data"]["basket_id"] == p.name
    assert first["data"]["assets"] == list(short["basket"].yes_ids) and first["t"] == _start_ns(p)
    assert last["kind"] == "stop" and last["data"] == {"synthetic": True, "n_messages": len(short["msgs"])}
    data = [r for r in raw if r["src"] != "meta"]
    assert all(set(r) == {"t", "src", "conn", "msg"} and r["src"] == "synthetic" and r["conn"] == 0 for r in data)
    assert [(r["t"], r["msg"]) for r in data] == short["msgs"]
    metas = Counter(r["kind"] for r in raw if r["src"] == "meta")
    assert metas == {"session_start": 1, "gap": len(short["truth"].gaps), "stop": 1}
    assert np.all(np.diff([r["t"] for r in raw]) >= 0)


def _read_gz_lines(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def test_write_synthetic_recording(tmp_path: Path) -> None:
    p = SyntheticParams(duration_s=900.0, seed=5, gap_prob_per_hour=4.0)
    out = write_synthetic_recording(p, tmp_path / "a")
    assert out == (tmp_path / "a" / f"{p.name}_seed5").resolve()
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["synthetic"] is True and meta["kind"] == "synthetic" and meta["basket_id"] == p.name
    assert SyntheticParams.from_dict(meta["params"], name=meta["basket_id"]) == p
    assert meta["basket"]["synthetic"] is True and meta["end_ns"] - meta["start_ns"] == 900 * 10**9
    (f,) = meta["files"]
    path = out / f["path"]
    assert f["path"].startswith("raw/") and f["path"].endswith(".jsonl.gz")
    blob = path.read_bytes()
    assert f["bytes"] == len(blob) and f["sha256"] == hashlib.sha256(blob).hexdigest()
    lines = _read_gz_lines(path)
    assert len(lines) == f["lines"] and lines == json.loads(json.dumps(list(iter_raw_lines(p))))
    assert len(meta["gaps"]) == sum(1 for r in lines if r["src"] == "meta" and r["kind"] == "gap")
    assert not list(out.rglob("*.tmp"))
    again = write_synthetic_recording(p, tmp_path / "b")  # byte-for-byte reproducible (gzip mtime 0)
    assert (again / f["path"]).read_bytes() == blob


def test_write_refuses_historical_books(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = SyntheticParams(duration_s=60.0)
    hist = tmp_path / "historical_books"
    with monkeypatch.context() as m:
        m.setattr(src, "HIST_ROOT", hist)
        for root in (hist, hist / "main", hist / "main" / "deeper"):
            with pytest.raises(ValueError, match="refusing"):
                write_synthetic_recording(p, root)
    assert not hist.exists()

    def no_mkdir(*a: Any, **k: Any) -> None:
        raise AssertionError("write_synthetic_recording tried to create a directory")

    # The real HIST_ROOT is refused before anything touches the disk (mkdir would fail the test).
    with monkeypatch.context() as m:
        m.setattr(Path, "mkdir", no_mkdir)
        with pytest.raises(ValueError, match="refusing"):
            write_synthetic_recording(p, src.HIST_ROOT)


def test_cli_writes_configured_basket(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "markets.json"
    cfg.write_text(json.dumps({"synthetic_baskets": [
        {"basket_id": "SYNTHETIC_cli", "synthetic": True, "seed": 4,
         "params": {"n_legs": 3, "pi0": [0.5, 0.3, 0.2], "duration_s": 86400}}]}), encoding="utf-8")
    assert main(["--name", "cli", "--days", "0.005", "--root", str(tmp_path / "out"), "--config", str(cfg)]) == 0
    out = Path(capsys.readouterr().out.strip())
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert out.name == "SYNTHETIC_cli_seed4" and meta["seed"] == 4
    assert meta["params"]["n_legs"] == 3 and meta["params"]["duration_s"] == pytest.approx(432.0)


# --------------------------------------------------------------------------- diagnostics / pairing
def test_calibration_summary_is_finite_and_consistent(short: dict[str, Any]) -> None:
    s = calibration_summary(short["p"], truth=short["truth"])
    assert all(isinstance(v, float) and math.isfinite(v) for v in s.values())
    assert s["s_bid_mean"] < s["s_mid_mean"] < s["s_ask_mean"]
    assert s["round_trip_cost"] == pytest.approx(s["spread_sum"] + 2.0 * s["fee_per_side"], abs=1e-15)
    assert s["dislocation_to_cost"] == pytest.approx(s["shock_size"] / s["round_trip_cost"])
    assert 0.0 <= s["frac_outside_band"] <= 1.0 and s["basket_disloc_sd"] > 0


def test_demo_defaults_hit_the_calibration_target() -> None:
    """Module docstring: the mean sweep is 0.5-1.5x the taker round-trip cost, so the
    strategy cannot win trivially; most of the time the sums sit inside the no-arb band."""
    s = calibration_summary(SyntheticParams(duration_s=6 * 3600.0))
    assert 0.5 <= s["dislocation_to_cost"] <= 1.5
    assert s["s_bid_mean"] < 1.0 < s["s_ask_mean"] and s["frac_outside_band"] < 0.25


def test_pairing_demo_relation() -> None:
    d = generate_pairing_demo(seed=11)
    assert d["synthetic"] is True and d["params"]["name"].startswith("SYNTHETIC_")
    joint, house, senate = d["joint"], d["house"], d["senate"]
    assert joint.columns.tolist() == list(BOP_LABELS)
    assert house.columns.tolist() == senate.columns.tolist() == ["Democratic", "Republican"]
    for df in (joint, house, senate):
        assert df.attrs["synthetic"] is True and df.attrs["event"].startswith("SYNTHETIC_")
        assert len(df) == 3 * 1440 and str(df.index.tz) == "UTC" and (df.index[1] - df.index[0]) == pd.Timedelta("1min")
        v = df.to_numpy()
        assert np.all((v > 0) & (v < 1))
    # house D = (D Senate, D House) + (R Senate, D House) up to stationary noise; pairing the
    # wrong joint legs (the D-Senate ones) is far off.
    right = house["Democratic"] - joint["D Senate, D House"] - joint["R Senate, D House"]
    wrong = house["Democratic"] - joint["D Senate, D House"] - joint["D Senate, R House"]
    assert right.abs().median() <= 0.01 and abs(right.mean()) < 0.005
    assert wrong.abs().median() > 5 * right.abs().median()
    assert np.corrcoef(house["Democratic"], joint["D Senate, D House"] + joint["R Senate, D House"])[0, 1] > 0.8
    sen = senate["Democratic"] - joint["D Senate, D House"] - joint["D Senate, R House"]
    assert sen.abs().median() <= 0.01
    again = generate_pairing_demo(seed=11)
    pd.testing.assert_frame_equal(again["house"], house)
    assert not generate_pairing_demo(seed=12)["house"].equals(house)


# --------------------------------------------------------------------------- slow: 3-day statistics
def _adf_pvalue(x: np.ndarray) -> float:
    from statsmodels.tsa.stattools import adfuller

    kw = {"result_object": False} if "result_object" in inspect.signature(adfuller).parameters else {}
    return float(adfuller(x, autolag="AIC", **kw)[1])


@pytest.fixture(scope="module")
def three_day_mids() -> tuple[SyntheticParams, pd.DataFrame]:
    """Observable mids of the configured demo (3 days, seed 7) on a 60 s LOCF grid; empty
    sides use the strict convention (bid 0, ask 1)."""
    p = SyntheticParams(duration_s=3 * 86400.0, seed=7)
    tob = synthetic_tob(p)
    grid = tob["t_ns"].iloc[0] + (np.arange(0.0, p.duration_s, 60.0) * 1e9).astype(np.int64)
    mids = {}
    for leg, g in tob.groupby("leg", sort=False):
        i = np.searchsorted(g["t_ns"].to_numpy(), grid, side="right") - 1
        bid = np.nan_to_num(g["bid"].to_numpy()[i], nan=0.0)
        ask = np.nan_to_num(g["ask"].to_numpy()[i], nan=1.0)
        mids[leg] = 0.5 * (bid + ask)
    return p, pd.DataFrame(mids)


@pytest.mark.slow
def test_adf_basket_sum_stationary_favourite_not(three_day_mids: tuple[SyntheticParams, pd.DataFrame]) -> None:
    """The basket mid sum mean-reverts (unit root rejected); a single leg follows the latent
    random walk (not rejected at 5 % for the configured demo seed)."""
    _, mids = three_day_mids
    assert _adf_pvalue(mids.sum(axis=1).to_numpy()) < 0.01
    assert _adf_pvalue(mids["outcome-0"].to_numpy()) > 0.05


@pytest.mark.slow
def test_half_life_recovered_from_observed_basket_sum(three_day_mids: tuple[SyntheticParams, pd.DataFrame]) -> None:
    """AR(1) half-life of the observed basket mid sum within 35 % of ``disloc_half_life_s``.

    Tick rounding and spread changes are measurement noise that attenuates the OLS lag-1
    coefficient (errors in variables). For AR(1) plus white noise the autocovariance ratio
    ``gamma(2) / gamma(1)`` is still ``phi`` (lag-1 instrument), so that is the estimator.
    """
    p, mids = three_day_mids
    s = mids.sum(axis=1).to_numpy()
    s = s - s.mean()
    phi = float(s[2:] @ s[:-2]) / float(s[1:] @ s[:-1])
    half_life = -math.log(2.0) / math.log(phi) * 60.0
    assert half_life == pytest.approx(p.disloc_half_life_s, rel=0.35)
