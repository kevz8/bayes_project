"""Data loading honesty rules + the end-to-end strategy pipeline (offline, synthetic data)."""
from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src import REPO_ROOT
from src.arb_engine import SignalConfig, ZConfig
from src.config import load_markets
from src.data_io import (
    DataKind,
    DataUnavailable,
    load_dataset,
    load_real_books,
    load_real_prices,
    load_synthetic,
    modelled_book,
)
from src.events import BookEvent, MetaEvent, Side
from src.execution_sim import ExecConfig, attribution_total
from src.orderbook import BookManager
from src.pipeline import RunConfig, StrategyRunner, fast_grid, run_backtest


@pytest.fixture(scope="module")
def synth():
    return load_synthetic("SYNTHETIC_demo5", duration_s=8 * 3600)


def _cfg(**exec_kw) -> RunConfig:
    ex = ExecConfig(**{"latency": ExecConfig().latency, **exec_kw})
    return RunConfig(z=ZConfig(window=60), signal=SignalConfig(z_entry=2.0, z_exit=0.2), exec=ex)


def test_synthetic_dataset_is_labelled(synth):
    assert synth.info.kind == DataKind.SYNTHETIC and synth.info.synthetic
    assert "SYNTHETIC" in synth.info.banner_markdown() and "implied by the generator" in synth.info.banner_markdown()
    assert synth.info.to_json()["synthetic"] is True
    tob = synth.top_of_book()
    assert list(tob.columns) == ["t_ns", "leg", "bid", "ask", "bid_sz", "ask_sz", "clean"]


def test_real_loaders_raise_without_data(tmp_path):
    with pytest.raises(DataUnavailable):
        load_real_books("fed-oct-2026", root=tmp_path)
    with pytest.raises(DataUnavailable):
        load_real_prices("fed-oct-2026", root=tmp_path)


def test_real_mode_never_falls_back(monkeypatch):
    import src.data_io as dio

    def nope(*a, **k):
        raise DataUnavailable("none")

    monkeypatch.setattr(dio, "load_real_books", nope)
    monkeypatch.setattr(dio, "load_real_prices", nope)
    with pytest.raises(DataUnavailable):
        load_dataset("fed-oct-2026", mode="real")
    ds = load_dataset("fed-oct-2026", mode="auto")
    assert ds.info.synthetic and any("FALLBACK" in n for n in ds.info.notes)


def test_modelled_book():
    bids, asks = modelled_book(0.55, 0.01, 2, 100.0, levels=3)
    assert bids[0][0] == 540000 and asks[0][0] == 560000 and len(bids) == len(asks) == 3
    assert asks[1][1] == 200.0  # depth grows with distance
    bids, asks = modelled_book(0.0125, 0.01, 1, 50.0)  # long shot -> 0.001 tick
    assert asks[0][0] - bids[0][0] == 1000


def test_end_to_end_backtest_and_attribution(synth):
    res = run_backtest(synth, _cfg())
    assert len(res.series) > 100 and len(res.equity) > 10
    assert (res.equity["equity_liq"] > 0).all()
    assert res.summary["data_kind"] == "synthetic"
    assert res.summary["n_frames"] > 1000
    for _, row in res.trades.iterrows():
        attr = {k[5:]: row[k] for k in row.index if k.startswith("attr_")}
        assert attribution_total(attr) == pytest.approx(row["pnl_net"], abs=1e-6)
        assert isinstance(bool(row["outside_band_at_entry"]), bool)
    assert res.attribution["net"] == pytest.approx(res.trades["pnl_net"].sum() if len(res.trades) else 0.0)


def test_synthetic_resolution_closes_positions(synth):
    res = run_backtest(synth, _cfg(exit_policy="hold"))
    if len(res.trades):
        assert set(res.trades["exit_method"]) <= {"resolution", "sell", "convert", "mark_end"}
    assert res.counters["resolutions"] >= 1


def test_live_and_replay_paths_agree(synth):
    cfg = _cfg()
    a = StrategyRunner(synth.basket, cfg)
    for t, evs in synth.frames():
        a.on_frame(t, evs)
    books = BookManager(list(synth.basket.yes_ids))
    b = StrategyRunner(synth.basket, cfg, books=books)
    for t, evs in synth.frames():  # what run_live_paper does: the feed owns the books
        if b.last_t is not None and t - b.last_t > cfg.gap_reset_s * 1e9:
            b._reset("gap")
        b.last_t = t
        b.sim.on_time(t, books.book_version)
        for ev in evs:
            if isinstance(ev, MetaEvent):
                books.mark_dirty(None, ev.kind)
                b.on_control_event(ev)
        changed = books.apply([e for e in evs if not isinstance(e, MetaEvent)])
        for ev in evs:
            if not isinstance(ev, (MetaEvent, BookEvent)):
                b.on_control_event(ev)
        b.on_books_changed(t, changed)
    ra, rb = a.finish(synth.info), b.finish(synth.info)
    assert list(ra.trades.get("t_signal_ns", [])) == list(rb.trades.get("t_signal_ns", []))
    assert ra.summary["final_equity"] == pytest.approx(rb.summary["final_equity"])


def test_fast_grid_mid_bounds_taker(synth):
    res = run_backtest(synth, _cfg())
    sums = res.series.rename(columns={})[["t_ns", "s_mid", "s_bid", "s_ask", "valid"]]
    kw = dict(windows=[30, 60], z_entries=[1.5, 2.0], z_exits=[0.0, 0.2])
    taker = fast_grid(sums, 0.01, 5, **kw)
    mid = fast_grid(sums, 0.01, 5, cost_mode="mid", **kw)
    assert len(taker) == 8
    assert (mid["pnl_per_unit"].to_numpy() >= taker["pnl_per_unit"].to_numpy() - 1e-12).all()


def test_hot_path_modules_do_not_import_pandas():
    code = ("import sys; import src.orderbook, src.clob_client, src.arb_engine, src.execution_sim; "
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
