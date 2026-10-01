"""Tests for src.execution_sim. The numbers are hand-computed wherever possible. Fully offline."""
from __future__ import annotations

import math
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from src import REPO_ROOT
from src.config import Basket, Defaults, FeeSchedule, Leg
from src.events import DictBookView, Side
from src.execution_sim import (
    ATTRIBUTION_KEYS,
    ExecConfig,
    ExecutionSimulator,
    GasModel,
    LatencyModel,
    Portfolio,
    attribution_total,
    gross_up_factor,
    liquidation_value_of,
    mirror_to_no,
    order_size_for_delivery,
    short_leg_via_no,
    take_liquidity,
    walk_book,
    walk_book_notional,
)

LONG, SHORT = Side.LONG_BASKET, Side.SHORT_BASKET
MS = 1_000_000
S = 1_000_000_000


# --------------------------------------------------------------------------- helpers
def make_basket(n: int = 3, rate: float = 0.0, *, tick: float = 0.01, synthetic: bool = True,
                neg_risk: bool | None = None, end_date: str | None = None, convert_fee_bips: float = 0.0) -> Basket:
    legs = tuple(
        Leg(leg_id=f"L{i}", label=f"L{i}", yes_token_id=f"SYNTHETIC-Y{i}", no_token_id=f"SYNTHETIC-N{i}",
            tick_size=tick, fee=FeeSchedule(rate=rate))
        for i in range(n)
    )
    return Basket(basket_id="SYNTHETIC_test", title="test", legs=legs, status="active", synthetic=synthetic,
                  neg_risk=neg_risk, end_date=end_date, convert_fee_bips=convert_fee_bips)


def set_tops(view: DictBookView, basket: Basket, bids, asks, size: float = 1000.0) -> None:
    """One level per side per leg; ``None`` leaves that side empty."""
    for leg, b, a in zip(basket.legs, bids, asks):
        view.set_book(leg.yes_token_id, [] if b is None else [(b, size)], [] if a is None else [(a, size)])


def cfg0(**kw) -> ExecConfig:
    """Deterministic config: zero latency, small minimums."""
    base = dict(latency=LatencyModel(0.0, 0.0), min_order_size=1.0, min_notional_usd=0.0)
    base.update(kw)
    return ExecConfig(**base)


def make_sim(basket: Basket, view: DictBookView, cfg: ExecConfig | None = None, cash: float = 10_000.0,
             seed: int = 0, **kw) -> ExecutionSimulator:
    return ExecutionSimulator(basket, view, cfg or cfg0(), Portfolio(cash), np.random.default_rng(seed), **kw)


def assert_identity(rec) -> None:
    assert set(rec.attribution) == set(ATTRIBUTION_KEYS)
    assert rec.pnl_net == pytest.approx(attribution_total(rec.attribution), abs=1e-9)
    assert rec.pnl_net == pytest.approx(rec.exit_proceeds - rec.entry_cost - rec.gas_usd - rec.legging_cost, abs=1e-9)


ASKS = ([0.40, 0.41], [10.0, 100.0])


# --------------------------------------------------------------------------- walk_book
def test_vwap_and_fee_usd_mode():
    f = walk_book(*ASKS, 50, fee=FeeSchedule(rate=0.04), fee_mode="usd", asset_id="a")
    assert f.filled == pytest.approx(50.0)
    assert f.notional == pytest.approx(20.40)
    assert f.vwap == pytest.approx(0.408)
    # 10*0.04*0.40*0.60 + 40*0.04*0.41*0.59 = 0.096 + 0.38704
    assert f.fee == pytest.approx(0.48304, abs=1e-12)
    assert (f.top_price, f.worst_price, f.insufficient) == (0.40, 0.41, False)
    assert [(lv.price, lv.qty) for lv in f.levels] == [(0.40, 10.0), (0.41, 40.0)]
    assert f.shares_delivered == pytest.approx(50.0)
    assert f.cash_delta == pytest.approx(-(20.40 + 0.48304))
    assert f.fee_cost == pytest.approx(0.48304)


def test_shares_mode_withholds_fee_in_shares():
    f = walk_book(*ASKS, 50, fee=FeeSchedule(rate=0.04), fee_mode="shares")
    # delivered = sum(take_k - fee_k/p_k) = 50 - 0.096/0.40 - 0.38704/0.41 = 50 - 0.24 - 0.944
    assert f.shares_delivered == pytest.approx(48.816, abs=1e-12)
    assert f.fee_shares == pytest.approx(1.184, abs=1e-12)
    assert f.fee == pytest.approx(0.48304, abs=1e-12)          # reported USD fee (rounded per order)
    assert f.cash_delta == pytest.approx(-20.40)               # cash pays the notional only
    assert f.fee_cost == pytest.approx(1.184 * 0.408)          # withheld shares valued at the VWAP
    # sells always pay the fee in USD
    s = walk_book([0.41], [100.0], 10, side="sell", fee=FeeSchedule(rate=0.04), fee_mode="shares")
    assert s.cash_delta == pytest.approx(4.1 - round(10 * 0.04 * 0.41 * 0.59, 5))
    assert s.shares_delivered == pytest.approx(10.0)


def test_insufficient_depth():
    f = walk_book([0.40, 0.41], [10.0, 5.0], 50)
    assert f.filled == pytest.approx(15.0)
    assert f.notional == pytest.approx(6.05)
    assert f.insufficient
    assert f.worst_price == 0.41


@pytest.mark.parametrize("ticks,expected", [(0, 10.0), (1, 50.0), (3, 50.0)])
def test_slip_guard(ticks, expected):
    f = walk_book(*ASKS, 50, limit_price=0.40 + ticks * 0.01)
    assert f.filled == pytest.approx(expected)
    assert f.insufficient == (expected < 50)


def test_sell_walk_with_limit():
    bids = ([0.43, 0.42], [100.0, 50.0])
    assert walk_book(*bids, 120, side="sell").notional == pytest.approx(100 * 0.43 + 20 * 0.42)
    lim = walk_book(*bids, 120, side="sell", limit_price=0.425)
    assert lim.filled == pytest.approx(100.0) and lim.insufficient


def test_notional_walk():
    f = walk_book_notional(*ASKS, 10.0)
    assert f.notional == pytest.approx(10.0)
    assert f.filled == pytest.approx(10.0 + 6.0 / 0.41)
    assert not f.insufficient
    g = walk_book_notional([0.40, 0.41], [10.0, 5.0], 100.0)
    assert g.notional == pytest.approx(6.05) and g.insufficient
    with pytest.raises(ValueError):
        walk_book_notional(*ASKS, 10.0, side="sell")


def test_walk_validation():
    with pytest.raises(ValueError):
        walk_book([0.4], [1.0], -1)
    with pytest.raises(ValueError):
        walk_book([0.4], [1.0], 1, side="short")
    with pytest.raises(ValueError):
        walk_book([0.4], [1.0], 1, fee_mode="bps")
    with pytest.raises(ValueError):
        walk_book([0.4, 0.5], [1.0], 1)
    empty = walk_book([], [], 5)
    assert empty.filled == 0.0 and empty.insufficient and math.isnan(empty.vwap) and math.isnan(empty.top_price)


def test_mirror_and_short_leg_via_no():
    px, sz = mirror_to_no([0.43, 0.42], [100.0, 50.0])
    np.testing.assert_allclose(px, [0.57, 0.58])
    np.testing.assert_allclose(sz, [100.0, 50.0])
    view = DictBookView({"Y": ([(0.43, 100), (0.42, 50)], [(0.45, 30)])})
    f = short_leg_via_no(view, "Y", 120)
    assert (f.asset_id, f.token, f.side) == ("Y", "NO", "buy")
    assert f.notional == pytest.approx(57.0 + 11.6)            # 100*0.57 + 20*0.58 = 68.6
    assert f.top_price == pytest.approx(0.57)
    sell = take_liquidity(view, "Y", "NO", "sell", 10)        # NO bid = 1 - YES ask
    assert sell.top_price == pytest.approx(0.55) and sell.notional == pytest.approx(5.5)
    assert take_liquidity(view, "Y", "YES", "buy", 10).top_price == pytest.approx(0.45)
    assert take_liquidity(view, "Y", "YES", "sell", 10).top_price == pytest.approx(0.43)


def test_fee_examples_and_symmetry():
    assert FeeSchedule(rate=0.04).taker_fee(100, 0.5) == pytest.approx(1.00)
    assert FeeSchedule(rate=0.07).taker_fee(100, 0.30) == pytest.approx(1.47)
    fee = FeeSchedule(rate=0.07)
    yes = walk_book([0.30], [500.0], 100, fee=fee, fee_mode="usd")
    view = DictBookView({"Y": ([(0.30, 500)], [])})            # YES bid 0.30 -> NO ask 0.70
    no = short_leg_via_no(view, "Y", 100, fee=fee, fee_mode="usd")
    assert yes.fee == no.fee == pytest.approx(1.47)
    assert walk_book([0.30], [500.0], 100, fee=None).fee == 0.0


def test_fee_rounded_per_order_not_per_level():
    px, sz = [0.50, 0.51, 0.52], [0.0004] * 3
    f = walk_book(px, sz, 0.0012, fee=FeeSchedule(rate=0.04), fee_mode="usd")
    per_level = [round(lv.fee_unrounded, 5) for lv in f.levels]
    assert sum(per_level) == 0.0
    assert f.fee == pytest.approx(0.00001, abs=1e-15)            # 1.1992e-5 rounded once


def test_gross_up_factor():
    fee = FeeSchedule(rate=0.04)
    g = gross_up_factor(fee, 0.5)
    assert g == pytest.approx(1 / (1 - 0.04 * 0.5))
    f = walk_book([0.5], [1000.0], 100 * g, fee=fee, fee_mode="shares")
    assert f.shares_delivered == pytest.approx(100.0, abs=1e-12)
    assert gross_up_factor(fee, 0.5, "usd") == 1.0
    assert gross_up_factor(FeeSchedule(), 0.5) == 1.0
    e2 = FeeSchedule(rate=0.25, exponent=2.0)
    g2 = gross_up_factor(e2, 0.3)
    assert walk_book([0.3], [1e6], 50 * g2, fee=e2).shares_delivered == pytest.approx(50.0, abs=1e-10)


def test_order_size_for_delivery_inverts_multilevel_walk():
    fee = FeeSchedule(rate=0.04)
    assert order_size_for_delivery(*ASKS, 48.816, fee=fee) == pytest.approx(50.0)   # inverse of the 50-share walk
    for target in (5.0, 9.6, 30.0, 100.0):
        q = order_size_for_delivery(*ASKS, target, fee=fee, limit_price=0.43)
        assert walk_book(*ASKS, q, fee=fee, limit_price=0.43).shares_delivered == pytest.approx(target, abs=1e-12)
    assert order_size_for_delivery(*ASKS, 30.0, fee=fee, fee_mode="usd") == 30.0
    beyond = order_size_for_delivery([0.40], [10.0], 20.0, fee=fee)               # past the depth: top gross-up
    assert beyond == pytest.approx(10.0 + (20.0 - 10 * (1 - 0.04 * 0.6)) * gross_up_factor(fee, 0.40))


# --------------------------------------------------------------------------- cost models / config
def test_latency_model():
    rng = np.random.default_rng(1)
    x = LatencyModel(500, 250).sample_ns(1000, rng)
    assert x.dtype == np.int64 and x.min() >= 500 * MS and x.max() <= 750 * MS
    np.testing.assert_array_equal(LatencyModel(500, 250).sample_ns(4, np.random.default_rng(7)),
                                  LatencyModel(500, 250).sample_ns(4, np.random.default_rng(7)))
    assert set(LatencyModel(10, 0).sample_ns(3, rng).tolist()) == {10 * MS}
    with pytest.raises(ValueError):
        LatencyModel(-1, 0)


def test_gas_model():
    assert GasModel().usd("convert", 5) == 0.0
    g = GasModel(relayer_pays=False)
    assert g.usd("redeem") == pytest.approx(150_000 * 600e-9 * 0.11)
    assert g.usd("convert", 3) == pytest.approx((100_000 + 3 * 80_000) * 600e-9 * 0.11)
    assert g.usd("clob_fill") == 0.0
    with pytest.raises(ValueError):
        GasModel().usd("teleport")


def test_exec_config_validation_and_defaults():
    with pytest.raises(NotImplementedError, match="directional"):
        ExecConfig(short_mode="overvalued_legs")
    for bad in (dict(gate="expected"), dict(fee_mode="bps"), dict(exit_policy="never"), dict(leg_fail_prob=2.0)):
        with pytest.raises(ValueError):
            ExecConfig(**bad)
    d = Defaults()
    c = ExecConfig.from_defaults(d, gate="edge")
    assert c.latency == LatencyModel(d.latency_ms, d.latency_jitter_ms)
    assert (c.fee_mode, c.convert_enabled, c.rf_annual, c.gate) == ("shares", False, d.risk_free_rate, "edge")
    assert c.gas.relayer_pays and c.gas.units["convert_per_leg"] == 80_000
    assert ExecConfig().fee_mode == "shares" and ExecConfig().gate == "none"


# --------------------------------------------------------------------------- portfolio
def test_portfolio_cash_conservation_and_no_naked_selling():
    p = Portfolio(1000.0)
    fills = [
        walk_book(*ASKS, 50, fee=FeeSchedule(rate=0.04), fee_mode="usd", asset_id="A"),
        walk_book([0.57], [100.0], 30, fee=FeeSchedule(rate=0.04), fee_mode="shares", asset_id="B", token="NO"),
        walk_book([0.39], [100.0], 20, side="sell", fee=FeeSchedule(rate=0.04), asset_id="A"),
    ]
    for f in fills:
        p.apply_fill(f)
    assert p.cash == pytest.approx(1000.0 + sum(f.cash_delta for f in fills), abs=1e-12)
    assert p.qty("A", "YES") == pytest.approx(30.0)
    assert p.qty("B", "NO") == pytest.approx(fills[1].shares_delivered)
    with pytest.raises(ValueError, match="naked"):
        p.apply_fill(walk_book([0.39], [100.0], 31, side="sell", asset_id="A"))


def test_portfolio_convert_and_settlement():
    b = make_basket(3)
    p = Portfolio(0.0)
    for y in b.yes_ids:
        p.holdings[(y, "NO")] = 10.0
    assert p.apply_convert(b, 10.0, fee_bips=50, gas_usd=0.5) == pytest.approx(2 * 10 * (1 - 0.005) - 0.5)
    assert p.holdings == {}
    with pytest.raises(ValueError):
        p.apply_convert(b, 1.0, 0.0, 0.0)

    p = Portfolio(0.0)
    y0, y1, y2 = b.yes_ids
    for y in b.yes_ids:
        p.holdings[(y, "YES")] = 2.0
        p.holdings[(y, "NO")] = 3.0
    assert p.settle_leg(b, y2, "No") == pytest.approx(3.0)       # NO_2 pays 1, YES_2 pays 0
    assert (y2, "YES") not in p.holdings and (y2, "NO") not in p.holdings
    assert p.settle(b.without_leg(y2), y0) == pytest.approx(2.0 + 3.0)   # YES_0 + NO_1
    assert p.cash == pytest.approx(8.0) and p.holdings == {}
    assert p.settle_leg(b, y1, "yes", gas_usd=1.0) == 0.0          # nothing held -> no gas


def test_liquidation_and_mid_value_without_convert():
    b = make_basket(2)
    view = DictBookView()
    set_tops(view, b, [0.40, 0.55], [0.42, 0.58], size=100)
    p = Portfolio(0.0)
    y0, y1 = b.yes_ids
    p.holdings[(y0, "YES")] = 10.0
    p.holdings[(y1, "NO")] = 10.0
    assert p.liquidation_value(view, b) == pytest.approx(10 * 0.40 + 10 * (1 - 0.58))
    assert p.mid_value(view, b) == pytest.approx(10 * 0.41 + 10 * (1 - 0.565))
    view.set_book(y1, [(0.55, 100)], [])                           # no YES ask -> NO cannot be sold
    assert p.liquidation_value(view, b) == pytest.approx(4.0)


def test_convert_floor_marking_portfolio():
    b = make_basket(3, convert_fee_bips=10)
    view = DictBookView()
    p = Portfolio(0.0)
    for y in b.yes_ids:
        p.holdings[(y, "NO")] = 100.0
    p.holdings[(b.yes_ids[0], "NO")] = 130.0                        # 30 unhedged NO on leg 0
    floor = 2 * 100 * (1 - 0.001)
    for asks in ([0.56, 0.34, 0.16], [0.80, 0.60, 0.50], [None, None, None], [0.10, 0.10, 0.10]):
        set_tops(view, b, [0.55, 0.33, 0.15], asks)
        v = p.liquidation_value(view, b, convert_enabled=True)
        assert v >= floor + liquidation_value_of(view, b, {}, {b.yes_ids[0]: 30.0}) - 1e-9
    set_tops(view, b, [0.55, 0.33, 0.15], [None, None, None])
    assert p.liquidation_value(view, b, convert_enabled=False) == 0.0
    incomplete = make_basket(3, synthetic=False, neg_risk=False)
    assert liquidation_value_of(view, incomplete, {}, {y: 100.0 for y in incomplete.yes_ids},
                                convert_enabled=True) == 0.0


# --------------------------------------------------------------------------- sizing
def test_planned_size_caps():
    b = make_basket(2)
    view = DictBookView()
    y0, y1 = b.yes_ids
    view.set_book(y0, [(0.48, 500)], [(0.50, 100), (0.51, 100), (0.60, 1000)])
    view.set_book(y1, [(0.43, 500)], [(0.45, 400)])
    assert make_sim(b, view).planned_size(LONG) == pytest.approx(100.0)   # 0.5 * 200 within 3 ticks
    assert make_sim(b, view, cash=100.0).planned_size(LONG) == pytest.approx(21.05)  # 0.2*100/0.95 floored
    assert make_sim(b, view, cfg0(min_order_size=5.0), cash=20.0).planned_size(LONG) == 0.0
    assert make_sim(b, view, cfg0(min_notional_usd=50.0)).planned_size(LONG) == 0.0  # leg 1: 100*0.45
    assert make_sim(b, view).planned_size(SHORT) == pytest.approx(250.0)  # NO asks: 0.5 * 500
    view.set_dirty(y1)
    assert make_sim(b, view).planned_size(LONG) == 0.0
    with pytest.raises(ValueError):
        make_sim(b, view).planned_size(Side.FLAT)


def test_shares_mode_gross_up_keeps_set_complete():
    b = make_basket(3, rate=0.05)
    view = DictBookView()
    set_tops(view, b, [0.19, 0.29, 0.39], [0.20, 0.30, 0.40])
    sim = make_sim(b, view)
    g = [1 / (1 - 0.05 * (1 - p)) for p in (0.2, 0.3, 0.4)]
    assert sim.planned_size(LONG) == pytest.approx(480.0)                # 0.5 * 1000 / g_0
    assert sim.enter(0, LONG)
    sim.on_time(0)
    pos = sim.position
    for y in b.yes_ids:
        assert sim.portfolio.qty(y, "YES") == pytest.approx(480.0, abs=1e-9)
    assert pos.rec.qty_hedged == pytest.approx(480.0, abs=1e-9)
    assert sim.counters["legging_events"] == 0 and pos.attr["legging_cost"] == pytest.approx(0.0, abs=1e-9)
    assert sim.portfolio.cash == pytest.approx(10_000 - sum(480 * gi * p for gi, p in zip(g, (0.2, 0.3, 0.4))))
    assert pos.rec.entry_fees == pytest.approx(sum(480 * (gi - 1) * p for gi, p in zip(g, (0.2, 0.3, 0.4))))


# --------------------------------------------------------------------------- latency
def _two_leg_long_book():
    b = make_basket(2)
    view = DictBookView()
    y0, y1 = b.yes_ids
    view.set_book(y0, [(0.48, 100)], [(0.50, 100)])
    view.set_book(y1, [(0.43, 100)], [(0.45, 100)])
    return b, view, y0, y1


def test_latency_fill_uses_later_book():
    b, view, y0, y1 = _two_leg_long_book()
    sim = make_sim(b, view, cfg0(latency=LatencyModel(500, 0)))
    assert sim.enter(0, LONG)                       # decision on asks 0.50 / 0.45, q = 50
    assert sim.on_time(200 * MS) == []              # nothing due yet
    view.set_book(y0, [(0.50, 100)], [(0.52, 100)])  # frame at 200 ms moves leg 0
    sim.on_time(1 * S)
    rec = sim.position.rec
    f0 = next(f for f in rec.entry_fills if f.asset_id == y0)
    assert f0.vwap == pytest.approx(0.52) and f0.t_ns == 500 * MS
    a = sim.position.attr
    assert a["latency_drift"] == pytest.approx(-50 * (0.51 - 0.49))
    assert a["half_spread"] == pytest.approx(-50 * (0.01 + 0.01))


def test_require_newer_book_defers_fill():
    b, view, y0, y1 = _two_leg_long_book()
    sim = make_sim(b, view, cfg0(require_newer_book=True))
    assert sim.enter(0, LONG, book_version=7)
    assert sim.on_time(0, book_version=7) == [] and sim.position.state == "entering"
    assert sim.counters["deferred_fills"] == 2
    view.set_book(y0, [(0.50, 100)], [(0.52, 100)])
    sim.on_time(1 * S, book_version=8)
    assert sim.position.state == "open"
    f0 = next(f for f in sim.position.rec.entry_fills if f.asset_id == y0)
    assert f0.vwap == pytest.approx(0.52) and f0.t_ns == 1 * S
    assert sim.counters["deferred_fills"] == 2      # each order counted once


# --------------------------------------------------------------------------- legging
def test_legging_unwind_cost():
    b, view, y0, y1 = _two_leg_long_book()
    sim = make_sim(b, view)
    assert sim.enter(0, LONG) and sim.position.rec.qty_target == pytest.approx(50.0)
    view.set_book(y1, [(0.43, 100)], [(0.45, 20)])  # depth vanishes before the fill
    sim.on_time(0)
    pos = sim.position
    assert pos.rec.qty_hedged == pytest.approx(20.0)
    assert pos.attr["legging_cost"] == pytest.approx(30 * 0.50 - 30 * 0.48)   # 0.60
    assert sim.counters["legging_events"] == 1
    assert sim.portfolio.qty(y0, "YES") == pytest.approx(20.0) and sim.portfolio.qty(y1, "YES") == pytest.approx(20.0)
    assert sim.portfolio.cash == pytest.approx(10_000 - 25.0 - 9.0 + 14.4)
    assert len(pos.rec.unwind_fills) == 1


def test_failed_entry_unwinds_and_reports():
    b, view, y0, y1 = _two_leg_long_book()
    failed = []
    sim = make_sim(b, view, on_entry_failed=failed.append)
    assert sim.enter(0, LONG)
    view.set_book(y1, [(0.43, 100)], [])            # leg 1 cannot fill at all
    out = sim.on_time(0)
    assert len(out) == 1 and failed == out and sim.position is None
    rec = out[0]
    assert rec.entry_failed and rec.exit_method == "unwind" and rec.qty_hedged == 0.0
    assert rec.pnl_net == pytest.approx(-50 * (0.50 - 0.48)) and rec.legging_cost == pytest.approx(1.0)
    assert sim.failed_entries == [rec] and sim.trades == []
    assert sim.counters["entry_failed"] == 1 and sim.portfolio.holdings == {}
    assert_identity(rec)


def test_leg_fail_prob_one_fails_every_leg():
    b, view, *_ = _two_leg_long_book()
    sim = make_sim(b, view, cfg0(leg_fail_prob=1.0))
    assert sim.enter(0, LONG)
    out = sim.on_time(0)
    assert out[0].entry_failed and out[0].pnl_net == 0.0
    assert sim.counters["leg_failures"] == 2 and sim.portfolio.cash == 10_000.0


def test_enter_rejected_when_busy_or_unsized():
    b, view, *_ = _two_leg_long_book()
    sim = make_sim(b, view)
    assert sim.enter(0, LONG)
    assert not sim.enter(0, SHORT)
    assert make_sim(b, view, cash=1.0, cfg=cfg0(min_order_size=5.0)).enter(0, LONG) is False
    assert sim.counters["entries_rejected"] == 1 and sim.counters["entries"] == 1
    with pytest.raises(ValueError):
        sim.enter(0, Side.FLAT)


# --------------------------------------------------------------------------- core economics
def _short_scenario(bids, asks, rate=0.0, **cfg_kw):
    b = make_basket(3, rate=rate)
    view = DictBookView()
    set_tops(view, b, bids, asks)
    sim = make_sim(b, view, cfg0(convert_enabled=True, **cfg_kw))
    assert sim.enter(0, SHORT)
    sim.on_time(0)
    assert sim.exit(1 * S, "revert") == "convert"
    (rec,) = sim.on_time(1 * S)
    return sim, rec


def test_short_inside_band_with_convert_realises_sbid_minus_one():
    # S_bid = 0.90 < 1, S_ask = 1.05 > 1 -> convert beats selling the NOs (1.95 < 2.00)
    sim, rec = _short_scenario([0.30] * 3, [0.35] * 3)
    assert rec.qty_hedged == pytest.approx(500.0)
    assert rec.exit_method == "convert" and not rec.outside_band_at_entry
    assert rec.pnl_net == pytest.approx(500 * (0.90 - 1.0))
    assert sim.portfolio.cash == pytest.approx(10_000 + rec.pnl_net) and sim.portfolio.holdings == {}
    assert_identity(rec)


@pytest.mark.parametrize("fee_mode", ["shares", "usd"])
def test_short_with_fees_realises_sbid_minus_one_minus_fees(fee_mode):
    sim, rec = _short_scenario([0.30] * 3, [0.35] * 3, rate=0.04, fee_mode=fee_mode)
    q = rec.qty_hedged
    if fee_mode == "shares":
        g = 1 / (1 - 0.04 * 0.3)                    # buy NO at 0.70: fee in shares = r * (1 - 0.70)
        assert q == pytest.approx(494.0)            # 0.5 * 1000 / g
        expected_fees = 3 * q * 0.70 * (g - 1)
    else:
        assert q == pytest.approx(500.0)
        expected_fees = 3 * round(q * 0.04 * 0.70 * 0.30, 5)
    assert rec.entry_fees == pytest.approx(expected_fees)
    assert rec.pnl_net == pytest.approx(q * (0.90 - 1.0) - expected_fees)
    assert_identity(rec)


def test_short_outside_band_locks_edge():
    sim, rec = _short_scenario([0.55, 0.33, 0.15], [0.56, 0.34, 0.16])   # NO set 1.97, convert 2.00
    assert rec.outside_band_at_entry
    assert rec.s_bid_entry == pytest.approx(1.03) and rec.s_ask_entry == pytest.approx(1.06)
    assert rec.pnl_net == pytest.approx(500 * 0.03)
    assert rec.attribution["payoff_adjustment"] > 0
    assert_identity(rec)


def test_short_sells_when_sask_below_one():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.30] * 3, [0.32] * 3)
    sim = make_sim(b, view, cfg0(convert_enabled=True))
    sim.enter(0, SHORT, z=2.4, mu=0.93)
    sim.on_time(0)
    assert sim.exit(1 * S, "revert", z=0.1, mu=0.95) == "sell"   # NO bids 0.68*3 = 2.04 > convert 2.00
    (rec,) = sim.on_time(1 * S)
    assert rec.exit_method == "sell" and rec.pnl_net == pytest.approx(500 * (2.04 - 2.10))
    assert rec.baseline_drift == pytest.approx(0.02) and (rec.z_entry, rec.z_exit) == (2.4, 0.1)
    assert rec.holding_s == pytest.approx(1.0)
    assert_identity(rec)


def test_mark_interval_and_same_timestamp_replacement():
    b, view, *_ = _two_leg_long_book()
    sim = make_sim(b, view, cfg0(mark_interval_s=60.0))
    assert sim.mark(0).equity_liq == pytest.approx(10_000.0)
    assert sim.mark(30 * S) is None
    sim.enter(61 * S, LONG)
    sim.on_time(61 * S)                                             # fill -> forced mark at 61 s
    assert [p.t_ns for p in sim.equity] == [0, 61 * S]
    pt = sim.equity[-1]
    assert pt.side == int(LONG) and pt.cash == pytest.approx(10_000 - 50 * 0.95)
    assert pt.equity_liq == pytest.approx(pt.cash + 50 * (0.48 + 0.43))
    assert pt.equity_mid == pytest.approx(pt.cash + 50 * (0.49 + 0.44)) and pt.gross_exposure > 0
    view.set_book(b.yes_ids[0], [(0.40, 100)], [(0.42, 100)])
    sim.mark(61 * S, force=True)
    assert len(sim.equity) == 2 and sim.equity[-1].equity_liq < pt.equity_liq


def test_convert_fee_and_gas_are_attributed():
    gas = GasModel(relayer_pays=False)
    sim, rec = _short_scenario([0.55, 0.33, 0.15], [0.56, 0.34, 0.16], convert_fee_bips=20.0, gas=gas)
    q = rec.qty_hedged
    assert rec.exit_fees == pytest.approx(2 * q * 0.002)
    assert rec.gas_usd == pytest.approx(gas.usd("convert", 3))
    assert rec.pnl_net == pytest.approx(q * 0.03 - 2 * q * 0.002 - gas.usd("convert", 3))
    assert_identity(rec)


def test_no_convert_on_incomplete_partition_and_short_hold_policy():
    view = DictBookView()
    incomplete = make_basket(3, synthetic=False, neg_risk=False)
    set_tops(view, incomplete, [0.55, 0.33, 0.15], [0.56, 0.34, 0.16])
    sim = make_sim(incomplete, view, cfg0(convert_enabled=True))
    sim.enter(0, SHORT)
    sim.on_time(0)
    assert not sim.convert_available and sim.exit(S, "revert") == "sell"
    complete = make_basket(3)
    set_tops(view, complete, [0.55, 0.33, 0.15], [0.56, 0.34, 0.16])
    for convert, expected in ((True, "convert"), (False, "hold")):
        sim = make_sim(complete, view, cfg0(convert_enabled=convert, exit_policy="hold"))
        sim.enter(0, SHORT)
        sim.on_time(0)
        assert sim.exit(S, "revert") == expected


def test_convert_floor_marking_never_below_n_minus_one():
    b = make_basket(3)
    view = DictBookView()
    bids = [0.55, 0.33, 0.15]
    set_tops(view, b, bids, [0.56, 0.34, 0.16])
    sim = make_sim(b, view, cfg0(convert_enabled=True))
    sim.enter(0, SHORT)
    sim.on_time(0)
    q = sim.position.rec.qty_hedged
    for k, asks in enumerate(([0.70, 0.50, 0.40], [None, None, None], [0.56, 0.34, 0.16], [0.99, 0.99, 0.99])):
        set_tops(view, b, bids, asks)
        pt = sim.mark((k + 1) * 3600 * S)
        assert pt.equity_liq >= pt.cash + 2 * q - 1e-9
        assert pt.side == int(SHORT)
    set_tops(view, b, bids, [None, None, None])
    no_convert = make_sim(b, view, cfg0(convert_enabled=False))
    no_convert.portfolio = sim.portfolio
    assert no_convert.mark(0).equity_liq == pytest.approx(sim.portfolio.cash)   # the bug the floor fixes


# --------------------------------------------------------------------------- resolution
def test_early_no_resolution_shrinks_basket_then_yes_settles():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.55, 0.33, 0.15], [0.56, 0.34, 0.16])
    sim = make_sim(b, view)
    sim.enter(0, SHORT)
    sim.on_time(0)
    q = sim.position.rec.qty_hedged
    cash0 = sim.portfolio.cash
    y0, y1, y2 = b.yes_ids
    assert sim.on_leg_resolved(y2, "No", 10 * S) == []
    assert sim.basket.n_legs == 2 and y2 not in sim.position.legs
    assert sim.portfolio.cash == pytest.approx(cash0 + q)            # NO_2 paid 1
    (rec,) = sim.on_leg_resolved(y0, "Yes", 20 * S)
    assert rec.exit_method == "resolution" and rec.legs_resolved == [y2, y0]
    assert rec.pnl_net == pytest.approx(q * (1.03 - 1.0))            # NO set pays exactly n - 1
    assert sim.resolved and not sim.enter(30 * S, LONG)
    assert sim.portfolio.holdings == {}
    assert_identity(rec)


def test_early_no_resolution_without_position():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.55, 0.33, 0.15], [0.56, 0.34, 0.16])
    sim = make_sim(b, view)
    assert sim.on_leg_resolved(b.yes_ids[1], "No", 0) == []
    assert sim.basket.yes_ids == (b.yes_ids[0], b.yes_ids[2])
    assert sim.on_leg_resolved("unknown", "No", 0) == []


def test_long_yes_winner_settlement():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.28] * 3, [0.30] * 3)
    sim = make_sim(b, view, cfg0(exit_policy="hold"))
    sim.enter(0, LONG)
    sim.on_time(0)
    assert sim.exit(5 * S, "revert") == "hold" and sim.position is not None
    (rec,) = sim.on_leg_resolved(b.yes_ids[1], "Yes", 10 * S)
    assert rec.exit_method == "resolution" and rec.exit_reason == "revert"
    assert rec.pnl_net == pytest.approx(500 * (1 - 0.90))
    assert sim.counters["holds"] == 1
    assert_identity(rec)


# --------------------------------------------------------------------------- exits
def test_hybrid_long_sells_only_above_discounted_hold():
    end = "2027-01-01T00:00:00Z"
    t = int((datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp() - 365.25 * 86400) * S)
    df = 1 / 1.042
    for bid, expected in ((0.33, "sell"), (0.31, "hold")):            # 0.99 vs 0.93 against df = 0.9597
        b = make_basket(3, end_date=end)
        view = DictBookView()
        set_tops(view, b, [0.29] * 3, [0.30] * 3)
        sim = make_sim(b, view, cfg0(exit_policy="hybrid"))
        assert sim.discount_factor(t) == pytest.approx(df)
        sim.enter(t, LONG)
        sim.on_time(t)
        set_tops(view, b, [bid] * 3, [bid + 0.01] * 3)
        assert sim.exit(t + S, "revert") == expected


def test_exit_while_entering_runs_after_entry():
    b, view, *_ = _two_leg_long_book()
    sim = make_sim(b, view, cfg0(latency=LatencyModel(500, 0)))
    sim.enter(0, LONG)
    assert sim.exit(100 * MS, "flip") == "pending"
    (rec,) = sim.on_time(1 * S)
    assert rec.exit_method == "sell" and rec.exit_reason == "flip" and rec.t_exit_signal_ns == 100 * MS
    assert rec.pnl_net == pytest.approx(50 * (0.48 + 0.43 - 0.95))
    assert sim.exit(2 * S, "x") is None
    assert_identity(rec)


def test_partial_exit_leaves_remainder_then_retries():
    b, view, y0, y1 = _two_leg_long_book()
    sim = make_sim(b, view)
    sim.enter(0, LONG)
    sim.on_time(0)
    view.set_book(y1, [(0.43, 20)], [(0.45, 100)])
    assert sim.exit(1 * S, "revert") == "sell"
    assert sim.on_time(1 * S) == []
    assert sim.counters["partial_exits"] == 1
    assert sim.position.legs[y1].rem == pytest.approx(30.0) and sim.position.legs[y0].rem == 0.0
    view.set_book(y1, [(0.44, 100)], [(0.45, 100)])
    assert sim.exit(2 * S, "retry") == "sell"
    (rec,) = sim.on_time(2 * S)
    assert rec.exit_method == "sell" and rec.exit_reason == "revert"
    assert rec.pnl_net == pytest.approx(50 * 0.48 + 20 * 0.43 + 30 * 0.44 - 50 * 0.95)
    assert_identity(rec)


def test_finalize_marks_open_position():
    b, view, y0, y1 = _two_leg_long_book()
    sim = make_sim(b, view)
    sim.enter(0, LONG)
    sim.on_time(0)
    view.set_book(y0, [(0.60, 100)], [(0.62, 100)])
    (rec,) = sim.finalize(10 * S)
    assert rec.exit_method == "mark_end" and rec.exit_reason == "end_of_data"
    assert rec.pnl_net == pytest.approx(50 * (0.60 + 0.43) - 50 * 0.95)
    assert rec.hold_value == pytest.approx(50.0)
    assert sim.equity[-1].t_ns == 10 * S
    assert sim.equity[-1].equity_liq == pytest.approx(10_000 + rec.pnl_net)
    assert sim.position is None and sim.trades == [rec]
    assert rec.s_bid_exit == pytest.approx(1.03)
    assert_identity(rec)


def test_finalize_cancels_in_flight_entry():
    b, view, *_ = _two_leg_long_book()
    sim = make_sim(b, view, cfg0(latency=LatencyModel(500, 0)))
    sim.enter(0, LONG)
    (rec,) = sim.finalize(100 * MS)
    assert rec.entry_failed and rec.pnl_net == 0.0 and sim.portfolio.cash == 10_000.0


# --------------------------------------------------------------------------- gates
def test_gate_modes_and_outside_band():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.55, 0.33, 0.15], [0.56, 0.34, 0.16])        # S_bid = 1.03 > 1
    none = make_sim(b, view, cfg0(convert_enabled=True))
    ok, edge, outside = none.gate_ok(SHORT, 0, mu=1.045)
    assert ok and outside and edge == pytest.approx(0.03)            # convert (2.00) - NO set (1.97)
    for gate in ("edge", "arb_only"):
        ok, edge, outside = make_sim(b, view, cfg0(convert_enabled=True, gate=gate)).gate_ok(SHORT, 0, mu=1.045)
        assert ok and outside and edge == pytest.approx(0.03)
    assert not make_sim(b, view, cfg0(convert_enabled=True, gate="edge", min_edge_usd=1e6)).gate_ok(SHORT, 0, 1.0)[0]

    set_tops(view, b, [0.30] * 3, [0.35] * 3)                         # inside the band
    assert none.gate_ok(SHORT, 0, mu=0.95)[0]                         # blueprint rule ignores costs
    for gate in ("edge", "arb_only"):
        ok, edge, outside = make_sim(b, view, cfg0(convert_enabled=True, gate=gate)).gate_ok(SHORT, 0, mu=0.95)
        assert not ok and not outside and edge < 0
    # without convert the short exit is a sale at n - E[S_ask_exit], E[S_ask_exit] = mu + spread/2
    _, e, _ = make_sim(b, view, cfg0(gate="edge")).gate_ok(SHORT, 0, mu=0.95)
    assert e == pytest.approx((3 - (0.95 + 0.075)) - 2.10)


def test_gate_long_side():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.28] * 3, [0.30] * 3)                         # S_ask = 0.90 < 1
    ok, edge, outside = make_sim(b, view, cfg0(gate="edge", exit_policy="hold")).gate_ok(LONG, 0, mu=0.87)
    assert ok and outside and edge == pytest.approx(0.10)
    ok, edge, _ = make_sim(b, view, cfg0(gate="edge")).gate_ok(LONG, 0, mu=1.0)
    assert ok and edge == pytest.approx((1.0 - 0.03) - 0.90)           # E[S_bid_exit] = mu - spread/2
    ok, edge, _ = make_sim(b, view, cfg0(gate="edge")).gate_ok(LONG, 0, mu=0.90)
    assert not ok and edge == pytest.approx(0.87 - 0.90)
    ok, edge, _ = make_sim(b, view, cfg0(gate="arb_only")).gate_ok(LONG, 0, mu=math.nan)
    assert ok and edge == pytest.approx(0.10)
    set_tops(view, b, [0.33] * 3, [0.35] * 3)
    assert not make_sim(b, view, cfg0(gate="arb_only")).gate_ok(LONG, 0, mu=2.0)[0]


# --------------------------------------------------------------------------- attribution
class _BookWalk:
    """Smooth random books: leg probabilities follow a log random walk, and the basket sum
    carries an AR(1) premium/discount, so the sum keeps crossing the band."""

    def __init__(self, rng: np.random.Generator):
        self.rng, self.p, self.skew = rng, {}, 0.0

    def step(self, view: DictBookView, basket: Basket) -> None:
        rng, ids = self.rng, basket.yes_ids
        p = np.array([self.p.get(y, 1.0 / len(ids)) for y in ids]) * np.exp(rng.normal(0, 0.03, len(ids)))
        p /= p.sum()
        self.p.update(zip(ids, p))
        self.skew = 0.85 * self.skew + float(rng.normal(0, 0.025))
        for y, pi in zip(ids, p * (1.0 + self.skew)):
            mid = float(np.clip(pi, 0.05, 0.95))
            half = 0.005 * int(rng.integers(1, 4))
            bids = [(round(mid - half - 0.01 * k, 4), float(rng.uniform(5, 300))) for k in range(3)]
            asks = [(round(mid + half + 0.01 * k, 4), float(rng.uniform(5, 300))) for k in range(3)]
            if rng.random() < 0.02:
                bids = []
            if rng.random() < 0.02:
                asks = []
            view.set_book(y, [x for x in bids if x[0] > 0], [x for x in asks if x[0] < 1])


def _run_random(seed: int, *, steps: int = 400, n: int = 4, rate: float = 0.04, resolve: bool = True, **cfg_kw):
    rng = np.random.default_rng(seed)
    b = make_basket(n, rate=rate)
    view = DictBookView()
    walk = _BookWalk(rng)
    walk.step(view, b)
    cfg = ExecConfig(latency=LatencyModel(400, 300), min_order_size=1.0, min_notional_usd=0.0,
                     participation_cap=0.8, **cfg_kw)
    sim = ExecutionSimulator(b, view, cfg, Portfolio(5_000.0), np.random.default_rng(seed + 1))
    closed, version, t = [], 0, 0
    for step in range(steps):
        t = step * 300 * MS
        closed += sim.on_time(t, book_version=version)
        if rng.random() < 0.7:
            walk.step(view, sim.basket)
            version += 1
        u = rng.random()
        s_mid = sum(sum(view.top(y)) / 2 for y in sim.basket.yes_ids)
        mu = s_mid + float(rng.normal(0, 0.02)) if math.isfinite(s_mid) else math.nan
        if sim.position is None and u < 0.15:
            sim.enter(t, LONG if rng.random() < 0.5 else SHORT, z=2.5, mu=mu, s_mid=s_mid, book_version=version)
        elif sim.position is not None and u < 0.12:
            sim.exit(t, "revert", z=0.1, mu=mu, s_mid=s_mid, book_version=version)
        elif resolve and u > 0.997 and sim.basket.n_legs > 2:
            closed += sim.on_leg_resolved(sim.basket.yes_ids[int(rng.integers(sim.basket.n_legs))], "No", t)
        sim.mark(t)
    closed += sim.finalize(t + S)
    return sim, closed


@pytest.mark.parametrize("cfg_kw", [
    dict(fee_mode="shares"),
    dict(fee_mode="usd"),
    dict(fee_mode="shares", convert_enabled=True, exit_policy="hybrid", leg_fail_prob=0.1),
    dict(fee_mode="usd", convert_enabled=True, require_newer_book=True, gas=GasModel(relayer_pays=False)),
    dict(fee_mode="shares", exit_policy="hold", convert_enabled=True),
], ids=["shares", "usd", "shares-convert-hybrid-legfail", "usd-convert-newer-gas", "hold"])
def test_attribution_identity_and_equity_consistency(cfg_kw):
    sim, closed = _run_random(11, **cfg_kw)
    records = sim.trades + sim.failed_entries
    assert len(records) == len(closed) and len(records) >= 1
    for rec in records:
        assert_identity(rec)
        assert rec.exit_method in ("sell", "convert", "resolution", "mark_end", "unwind")
        assert rec.baseline_drift == pytest.approx(rec.mu_exit - rec.mu_entry, nan_ok=True)
    assert sim.position is None
    total = sum(r.pnl_net for r in records)
    assert sim.equity[-1].equity_liq - sim.portfolio.initial_cash == pytest.approx(total, abs=1e-7)
    if cfg_kw.get("exit_policy", "z_exit") == "z_exit":
        assert len(sim.trades) >= 5


@pytest.mark.slow
@pytest.mark.parametrize("fee_mode", ["shares", "usd"])
def test_attribution_identity_many_seeds(fee_mode):
    for seed in range(40):
        for kw in (dict(), dict(convert_enabled=True, exit_policy="hybrid", leg_fail_prob=0.2),
                   dict(convert_enabled=True, require_newer_book=True)):
            sim, _ = _run_random(seed, steps=600, fee_mode=fee_mode, **kw)
            records = sim.trades + sim.failed_entries
            for rec in records:
                assert_identity(rec)
            total = sum(r.pnl_net for r in records)
            assert sim.equity[-1].equity_liq - sim.portfolio.initial_cash == pytest.approx(total, abs=1e-6)


def test_attribution_components_have_cost_signs():
    sim, _ = _run_random(3, fee_mode="usd", resolve=False)
    for rec in sim.trades:
        if rec.qty_hedged > 0 and rec.exit_method == "sell":
            assert rec.attribution["fees"] >= 0.0
            assert rec.attribution["depth_slippage"] <= 1e-12
            assert rec.attribution["half_spread"] <= 1e-12


def test_trade_record_to_row():
    sim, rec = _short_scenario([0.55, 0.33, 0.15], [0.56, 0.34, 0.16])
    row = rec.to_row()
    assert row["side"] == -1 and row["exit_method"] == "convert" and row["attr_fees"] == 0.0
    assert "entry_fills" not in row and row["outside_band_at_entry"] is True


# --------------------------------------------------------------------------- sanity bounds
def test_no_dislocation_with_costs_never_profits():
    rng = np.random.default_rng(5)
    b = make_basket(3, rate=0.02)
    view = DictBookView()
    cfg = ExecConfig(latency=LatencyModel(500, 0), min_order_size=1.0, min_notional_usd=0.0)
    sim = ExecutionSimulator(b, view, cfg, Portfolio(), np.random.default_rng(6))

    def books():
        p = rng.dirichlet(np.full(3, 3.0))                            # S_mid == 1 exactly
        for leg, pi in zip(b.legs, p):
            view.set_book(leg.yes_token_id, [(pi - 0.01, 1e4)], [(pi + 0.01, 1e4)])

    books()
    for step in range(600):
        t = step * S
        sim.on_time(t)
        books()
        if sim.position is None and rng.random() < 0.2:
            sim.enter(t, LONG if rng.random() < 0.5 else SHORT)
        elif sim.position is not None and rng.random() < 0.2:
            sim.exit(t, "revert")
    sim.finalize(600 * S)
    assert len(sim.trades) > 20
    assert all(r.pnl_net <= 1e-9 for r in sim.trades)
    assert all(abs(r.attribution["gross_mid"]) < 1e-9 for r in sim.trades)


def test_zero_costs_and_shock_profit():
    b = make_basket(3)
    view = DictBookView()
    set_tops(view, b, [0.40, 0.40, 0.30], [0.40, 0.40, 0.30])        # shock: S = 1.10, zero spread
    sim = make_sim(b, view)
    sim.enter(0, SHORT)
    sim.on_time(0)
    set_tops(view, b, [0.35, 0.35, 0.30], [0.35, 0.35, 0.30])        # reverts to 1.00
    sim.exit(S, "revert")
    (rec,) = sim.on_time(S)
    assert rec.pnl_net == pytest.approx(rec.qty_hedged * 0.10) and rec.pnl_net > 0
    assert rec.attribution["gross_mid"] == pytest.approx(rec.pnl_net)


# --------------------------------------------------------------------------- hygiene
def test_import_does_not_pull_pandas():
    code = "import sys, src.execution_sim; sys.exit(1 if 'pandas' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, timeout=60).returncode == 0
