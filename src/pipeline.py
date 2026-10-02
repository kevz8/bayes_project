"""One code path from market data to P&L: replay backtests, parameter grids, live paper trading.

``StrategyRunner`` glues the pieces in a fixed order per frame stamped ``t``::

    sim.on_time(t)                      # orders due by t fill against the book BEFORE frame t
    books.apply(frame)                  # L2 books (src.orderbook)
    basket_state.update_from_view(...)  # S_mid / S_bid / S_ask (src.arb_engine)
    engine.update(t, S_mid)             # rolling z, excluding S_t from its own window
    state_machine.step(t, z, gate)      # FLAT / LONG / SHORT with the executable-edge gate
    sim.enter / sim.exit                # latency, depth walking, fees, legging (src.execution_sim)
    sim.mark(t)                         # liquidation- and mid-marked equity

Strategies:

* ``zscore`` - the blueprint rule (enter at |z| > z_entry, exit at |z| < z_exit), gated by
  ``ExecConfig.gate`` (``none`` = pure blueprint, ``edge`` = exact exit-formula edge > 0);
* ``arb`` - no z at all: enter whenever the walked book offers a riskless edge (NO set cheaper
  than n-1 or YES set cheaper than 1 after fees) and hold to resolution / convert.

Live paper trading (``run_live_paper``) feeds the same runner from ``MarketDataFeed``; there is
no order-signing code anywhere in this repository.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import sys
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .arb_engine import (
    Action,
    BasketState,
    SignalConfig,
    SignalStateMachine,
    ZConfig,
    ZScoreEngine,
    positions_from_z,
    zscore_batch,
)
from .config import Basket, basket_fee_per_unit, load_markets
from .data_io import Dataset, DatasetInfo, load_dataset
from .events import MarketResolvedEvent, MetaEvent, NewMarketEvent, Side, TickSizeEvent
from .execution_sim import ExecConfig, ExecutionSimulator, Portfolio, TradeRecord
from .metrics import summarize
from .orderbook import BookManager

logger = logging.getLogger(__name__)
_NS = 1_000_000_000


@dataclass(frozen=True)
class RunConfig:
    strategy: Literal["zscore", "arb"] = "zscore"
    z: ZConfig = field(default_factory=ZConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)
    initial_cash: float = 10_000.0
    gap_reset_s: float = 60.0          # no frame for this long -> z engine restarts (recording gap)
    trade_start_ns: int | None = None  # engine warms on earlier data but cannot trade before this
    trade_end_ns: int | None = None    # data after this is ignored
    arb_min_edge: float = 0.0          # per-unit top-of-book edge that triggers an arb check
    series_every_s: float = 0.0        # 0 = record every pushed S; >0 = at most one row per interval
    seed: int = 0

    def describe(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "window": self.z.window, "sigma_floor": self.z.sigma_floor,
                "z_entry": self.signal.z_entry, "z_exit": self.signal.z_exit, "exit_ref": self.signal.exit_ref,
                "gate": self.exec.gate, "exit_policy": self.exec.exit_policy,
                "convert_enabled": self.exec.convert_enabled, "fee_mode": self.exec.fee_mode,
                "latency_ms": self.exec.latency.base_ms, "jitter_ms": self.exec.latency.jitter_ms,
                "require_newer_book": self.exec.require_newer_book, "initial_cash": self.initial_cash}


class SeriesRecorder:
    """Append-only columnar buffer (amortised O(1)); no DataFrame until ``to_frame``."""

    COLS = ("t_ns", "s_mid", "s_bid", "s_ask", "mu", "sigma", "z", "valid", "side")

    def __init__(self, cap: int = 4096) -> None:
        self._a = {c: np.empty(cap) for c in self.COLS}
        self._n = 0

    def append(self, *vals: float) -> None:
        if self._n == len(self._a["t_ns"]):
            for c in self.COLS:
                self._a[c] = np.resize(self._a[c], 2 * self._n)
        for c, v in zip(self.COLS, vals):
            self._a[c][self._n] = v
        self._n += 1

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({c: self._a[c][: self._n] for c in self.COLS})
        df["t_ns"] = df["t_ns"].astype("int64")
        df["valid"] = df["valid"].astype(bool)
        df["side"] = df["side"].astype(int)
        df.index = pd.to_datetime(df["t_ns"], unit="ns", utc=True)
        df.index.name = "time"
        return df


@dataclass
class BacktestResult:
    info: DatasetInfo
    config: RunConfig
    series: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    summary: dict
    counters: dict
    attribution: dict


class StrategyRunner:
    def __init__(self, basket: Basket, cfg: RunConfig, *, books: BookManager | None = None) -> None:
        self.basket = basket
        self.cfg = cfg
        self.books_owned = books is None
        self.books = books if books is not None else BookManager(list(basket.yes_ids))
        self.state = BasketState.from_basket(basket)
        self.engine = ZScoreEngine(cfg.z)
        self.sm = SignalStateMachine(cfg.signal, sigma_floor=cfg.z.sigma_floor)
        self.portfolio = Portfolio(cfg.initial_cash)
        self.sim = ExecutionSimulator(basket, self.books, cfg.exec, self.portfolio, np.random.default_rng(cfg.seed),
                                      on_entry_failed=lambda rec: self.sm.on_entry_failed())
        self.series = SeriesRecorder()
        self.last_t: int | None = None
        self.halted = False
        self.n_frames = 0
        self.n_resets = 0
        self._last_series_t = -1
        self._zs = None

    # ------------------------------------------------------------------ per frame
    def on_frame(self, t: int, events: Sequence[Any]) -> None:
        """Replay entry point: control events, then book updates, then the strategy."""
        cfg = self.cfg
        if cfg.trade_end_ns is not None and t > cfg.trade_end_ns:
            return
        self.n_frames += 1
        if self.last_t is not None and (t - self.last_t) > cfg.gap_reset_s * _NS:
            self._reset("gap")
        self.last_t = t
        self.sim.on_time(t, self.books.book_version)
        market = []
        for ev in events:
            if isinstance(ev, MetaEvent):
                self.on_control_event(ev)
            else:
                market.append(ev)
        changed = self.books.apply(market) if (self.books_owned and market) else set()
        for ev in market:
            if isinstance(ev, (MarketResolvedEvent, NewMarketEvent)):
                self.on_control_event(ev)
        self.on_books_changed(t, changed)

    def on_control_event(self, ev: Any) -> None:
        if isinstance(ev, MetaEvent):
            if ev.kind in ("replay_start", "disconnect", "gap", "connect"):
                if self.books_owned and ev.kind != "connect":
                    self.books.mark_dirty(None, ev.kind)
                self._reset(ev.kind)
            return
        if isinstance(ev, MarketResolvedEvent):
            self._on_resolved(ev)
        elif isinstance(ev, NewMarketEvent):
            logger.warning("new market %s appeared in a monitored event; halting entries (re-run discovery)", ev.market)
            self.halted = True

    def _on_resolved(self, ev: MarketResolvedEvent) -> None:
        leg_ids = set(self.basket.yes_ids)
        yes = [a for a in ev.asset_ids if a in leg_ids]
        if not yes:
            return
        yes_id = yes[0]
        if ev.winning_asset_id is not None:
            outcome = "Yes" if ev.winning_asset_id == yes_id else "No"
        else:
            outcome = "Yes" if (ev.winning_outcome or "").lower() == "yes" else "No"
        t = ev.t_recv_ns
        self.sim.on_leg_resolved(yes_id, outcome, t)
        if outcome == "Yes":
            self.sm.force_flat(t, "resolution")
            self.halted = True
        else:
            self.state.remove_leg(yes_id)
            self.basket = self.sim.basket
            if self.sim.position is None:
                self.sm.force_flat(t, "resolution")
            self._reset("leg_resolved")

    def _reset(self, reason: str) -> None:
        self.engine.reset(reason)
        self.n_resets += 1

    def on_books_changed(self, t: int, changed: Iterable[str]) -> None:
        cfg = self.cfg
        self.state.update_from_view(self.books, changed, t)
        snap = self.state.snapshot(t)
        zs = self.engine.update(t, snap.s_mid, snap.valid and snap.clean)
        self._zs = zs
        if zs.pushed and (cfg.series_every_s <= 0 or t - self._last_series_t >= cfg.series_every_s * _NS):
            self.series.append(t, snap.s_mid, snap.s_bid, snap.s_ask, zs.mu, zs.sigma, zs.z, snap.valid, int(self.sim.side))
            self._last_series_t = t
        can_trade = (not self.halted and snap.tradable and snap.clean
                     and (cfg.trade_start_ns is None or t >= cfg.trade_start_ns))
        if cfg.strategy == "arb":
            if can_trade and self.sim.position is None and not self.sim.resolved:
                self._try_arb(t, snap)
        else:
            self._step_zscore(t, zs, snap, can_trade)
        self.sim.mark(t)

    def _step_zscore(self, t: int, zs: Any, snap: Any, can_trade: bool) -> None:
        sm, sim = self.sm, self.sim
        if sm.state is Side.FLAT and not can_trade:
            return

        def gate(side: Side) -> bool:
            ok, _, _ = sim.gate_ok(side, t, zs.mu, zs.sigma)
            return ok

        sig = sm.step(t, zs, gate)
        if sig is None:
            return
        if sig.action is Action.EXIT:
            sim.exit(t, sig.reason, z=sig.z, mu=sig.mu, s_mid=sig.s, book_version=self.books.book_version)
            return
        side = sig.side_after
        ok = sim.enter(t, side, z=sig.z, mu=sig.mu, sigma=sig.sigma, s_mid=sig.s, strategy="zscore",
                       book_version=self.books.book_version)
        if not ok:
            sm.on_entry_failed()

    def _try_arb(self, t: int, snap: Any) -> None:
        fee = basket_fee_per_unit(self.basket.fees, self.state.mid)
        for side, edge in ((Side.SHORT_BASKET, snap.s_bid - 1.0), (Side.LONG_BASKET, 1.0 - snap.s_ask)):
            if edge - fee > self.cfg.arb_min_edge:
                ok, e, _ = self.sim.gate_ok(side, t, math.nan)
                if ok and self.sim.enter(t, side, mu=math.nan, s_mid=snap.s_mid, strategy="arb",
                                         book_version=self.books.book_version):
                    if self.cfg.exec.exit_policy != "z_exit":
                        self.sim.exit(t, "hold_to_resolution", book_version=self.books.book_version)
                    return

    # ------------------------------------------------------------------ results
    def finish(self, info: DatasetInfo) -> BacktestResult:
        t_end = self.last_t or 0
        self.sim.finalize(t_end)
        trades = pd.DataFrame([r.to_row() for r in self.sim.trades])
        if len(trades):
            trades["t_entry"] = pd.to_datetime(trades["t_signal_ns"], unit="ns", utc=True)
            trades["t_exit"] = pd.to_datetime(trades["t_close_ns"], unit="ns", utc=True)
        else:
            trades = pd.DataFrame(columns=["t_entry", "t_exit", "side", "pnl_net", "holding_s"])
        eq = pd.DataFrame([dataclasses.asdict(p) for p in self.sim.equity])
        if len(eq):
            eq.index = pd.to_datetime(eq["t_ns"], unit="ns", utc=True)
            eq.index.name = "time"
        attribution = {k: float(sum(r.attribution.get(k, 0.0) for r in self.sim.trades))
                       for k in ("gross_mid", "latency_drift", "half_spread", "depth_slippage",
                                 "payoff_adjustment", "fees", "gas", "legging_cost")}
        attribution["net"] = float(sum(r.pnl_net for r in self.sim.trades))
        summary = summarize(eq["equity_liq"] if len(eq) else pd.Series(dtype=float),
                            trades if len(trades) else None, rf_annual=self.cfg.exec.rf_annual) if len(eq) > 1 else {}
        summary.update({"n_frames": self.n_frames, "engine_resets": self.n_resets, "data_kind": info.kind.value,
                        "config": self.cfg.describe(), "final_equity": float(eq["equity_liq"].iloc[-1]) if len(eq) else self.cfg.initial_cash,
                        "n_failed_entries": len(self.sim.failed_entries)})
        return BacktestResult(info, self.cfg, self.series.to_frame(), trades, eq, summary,
                              dict(self.sim.counters), attribution)


def run_backtest(dataset: Dataset, cfg: RunConfig) -> BacktestResult:
    runner = StrategyRunner(dataset.basket, cfg)
    for t, events in dataset.frames():
        runner.on_frame(t, events)
    return runner.finish(dataset.info)


# --------------------------------------------------------------------------- fast grid
def fast_grid(sums: pd.DataFrame, fee_per_unit: float, n_legs: int, *, windows: Sequence[int],
              z_entries: Sequence[float], z_exits: Sequence[float], sigma_floor: float = 0.005,
              cost_mode: Literal["taker", "mid"] = "taker") -> pd.DataFrame:
    """Approximate per-unit P&L of the z rule for every (N, z_entry, z_exit) in O(T) per N.

    ``sums`` has ``t_ns, s_mid, s_bid, s_ask, valid``. z is computed once per window with the
    vectorised engine (identical to the streaming one); each threshold pair replays the state
    machine and books taker round trips at the top-of-book sums: long = ``S_bid_exit - S_ask_entry``,
    short (NO set) = ``S_bid_entry - S_ask_exit``, minus ``2 * fee_per_unit``. ``cost_mode="mid"``
    instead scores mid-to-mid (the frictionless upper bound). Used only to *select* parameters;
    the selected configuration is re-run through the full simulator.
    """
    t = sums["t_ns"].to_numpy(np.int64)
    s = sums["s_mid"].to_numpy(float)
    sb = sums["s_bid"].to_numpy(float)
    sa = sums["s_ask"].to_numpy(float)
    valid = sums["valid"].to_numpy(bool)
    rows = []
    for w in windows:
        zc = ZConfig(window=int(w), sigma_floor=sigma_floor)
        b = zscore_batch(s, valid, zc)
        for ze in z_entries:
            for zx in z_exits:
                if not zx < ze:
                    continue
                _, sigs = positions_from_z(t, s, b["mu"], b["sigma"], b["z"], SignalConfig(z_entry=ze, z_exit=zx))
                pos = np.searchsorted(t, [sg.t_ns for sg in sigs])
                pnl = []
                for k in range(0, len(sigs) - 1, 2):
                    e, x = sigs[k], sigs[k + 1]
                    i, j = pos[k], pos[k + 1]
                    if cost_mode == "mid":
                        g = (s[j] - s[i]) * (1 if e.action is Action.ENTER_LONG else -1)
                    elif e.action is Action.ENTER_LONG:
                        g = sb[j] - sa[i] - 2 * fee_per_unit
                    else:
                        g = sb[i] - sa[j] - 2 * fee_per_unit
                    pnl.append(g)
                pnl = np.array(pnl)
                rows.append({"window": int(w), "z_entry": ze, "z_exit": zx, "n_trades": len(pnl),
                             "pnl_per_unit": float(pnl.sum()) if len(pnl) else 0.0,
                             "mean_per_trade": float(pnl.mean()) if len(pnl) else math.nan,
                             "hit_rate": float((pnl > 0).mean()) if len(pnl) else math.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- live paper trading
async def run_live_paper(basket_id: str, cfg: RunConfig, *, duration_s: float | None = None,
                         status_every_s: float = 10.0) -> BacktestResult:
    """Paper-trade a basket on the live WebSocket with the same runner as the backtests."""
    from .clob_client import ClobRestClient, MarketDataFeed
    from .data_io import DataKind
    import time

    basket = load_markets().baskets[basket_id]
    books = BookManager(list(basket.yes_ids))
    feed = MarketDataFeed(list(basket.yes_ids), rest=ClobRestClient(), books=books)
    runner = StrategyRunner(basket, cfg, books=books)
    updates = feed.subscribe_updates()
    feed.add_listener(runner.on_control_event)
    task = asyncio.create_task(feed.run())
    t0 = time.monotonic()
    last_status = t0
    try:
        while duration_s is None or time.monotonic() - t0 < duration_s:
            try:
                batch = await asyncio.wait_for(updates.get_batch(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            t = max(batch) if batch else time.time_ns()
            runner.sim.on_time(t, books.book_version)
            runner.on_books_changed(t, set(basket.yes_ids))
            runner.last_t = t
            if time.monotonic() - last_status > status_every_s:
                zs = runner._zs
                logger.info("S_mid=%.4f z=%.2f side=%s equity=%.2f", zs.s if zs else math.nan,
                            zs.z if zs else math.nan, runner.sim.side.name,
                            runner.sim.equity[-1].equity_liq if runner.sim.equity else cfg.initial_cash)
                last_status = time.monotonic()
    finally:
        await feed.stop()
        task.cancel()
    info = DatasetInfo(DataKind.REAL_BOOKS, basket_id, basket.title, notes=["live paper trading session"])
    return runner.finish(info)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--basket", required=True)
    b.add_argument("--data", default="auto", help="auto | real | real_books | real_prices | synthetic")
    b.add_argument("--strategy", default="zscore", choices=("zscore", "arb"))
    b.add_argument("--window", type=int, default=500)
    b.add_argument("--z-entry", type=float, default=2.0)
    b.add_argument("--z-exit", type=float, default=0.2)
    b.add_argument("--gate", default="none", choices=("none", "edge", "arb_only"))
    b.add_argument("--exit-policy", default="z_exit", choices=("z_exit", "hold", "hybrid"))
    b.add_argument("--convert", action="store_true")
    lv = sub.add_parser("live")
    lv.add_argument("--basket", required=True)
    lv.add_argument("--duration", type=float, default=600)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfgm = load_markets()
    if args.cmd == "backtest":
        ds = load_dataset(args.basket, mode=args.data, cfg=cfgm)
        ex = ExecConfig.from_defaults(cfgm.defaults, gate=args.gate, exit_policy=args.exit_policy,
                                      convert_enabled=args.convert,
                                      require_newer_book=ds.info.kind.value != "real_books")
        rc = RunConfig(strategy=args.strategy, z=ZConfig(window=args.window),
                       signal=SignalConfig(z_entry=args.z_entry, z_exit=args.z_exit), exec=ex)
        res = run_backtest(ds, rc)
        print(ds.info.banner_markdown().splitlines()[0])
        print(json.dumps({k: v for k, v in res.summary.items() if not isinstance(v, (dict, list))}, indent=1, default=str))
        return 0
    rc = RunConfig(exec=ExecConfig.from_defaults(cfgm.defaults))
    res = asyncio.run(run_live_paper(args.basket, rc, duration_s=args.duration))
    print(json.dumps({k: v for k, v in res.summary.items() if not isinstance(v, (dict, list))}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
