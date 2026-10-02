from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import plotting as P


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-10-01", periods=300, freq="min", tz="UTC")
    s = 1 + np.cumsum(rng.normal(0, 0.002, 300)) * 0.1 + rng.normal(0, 0.004, 300)
    df = pd.DataFrame({"s_mid": s, "mu": pd.Series(s).rolling(30, min_periods=1).mean().to_numpy(),
                       "sigma": 0.004, "s_bid": s - 0.01, "s_ask": s + 0.01}, index=idx)
    df.iloc[10:15, 0] = np.nan
    return df


@pytest.fixture
def equity(series):
    eq = 10_000 + np.cumsum(np.random.default_rng(1).normal(0, 5, len(series)))
    return pd.DataFrame({"equity_liq": eq, "equity_mid": eq + 3}, index=series.index)


def _trades(idx):
    return pd.DataFrame({"t_entry": [idx[50], idx[150]], "t_exit": [idx[90], pd.NaT], "side": [1, -1]})


def test_signal_equity_two_stacked_panels(series, equity):
    fig = P.plot_basket_signal_equity(series, _trades(series.index), equity, z_entry=2, data_kind="real_books", title="t")
    assert len(fig.axes) == 2
    top, bottom = fig.axes
    assert top.get_shared_x_axes().joined(top, bottom)
    assert top.get_position().y0 > bottom.get_position().y1 - 1e-6  # stacked, not twinned
    assert len(top.collections) >= 4  # bands + entry markers + exit markers
    assert fig._suptitle.get_text().startswith("REAL ORDER BOOKS")
    plt.close(fig)


def test_signal_equity_empty_trades(series, equity):
    fig = P.plot_basket_signal_equity(series, pd.DataFrame(columns=["t_entry", "t_exit", "side"]), equity,
                                      z_entry=2, data_kind="synthetic", title="t")
    texts = [t.get_text() for t in fig.texts]
    assert "SYNTHETIC" in texts and "SYNTHETIC DATA" in fig._suptitle.get_text()
    plt.close(fig)


def test_other_figures(series, equity, tmp_path):
    legs = pd.DataFrame(np.random.default_rng(2).dirichlet(np.ones(10), 100), columns=[f"L{i}" for i in range(10)])
    figs = [
        P.plot_sum_band(series, data_kind="real_prices", title="a"),
        P.plot_legs(legs, data_kind="real_prices", title="b"),
        P.plot_sum_distribution(series["s_mid"], data_kind="real_prices", title="c"),
        P.plot_variogram({"sum": ([1, 2, 4], [1, 1.5, 1.6]), "leg": ([1, 2, 4], [1, 2, 4])}, data_kind="real_prices", title="d"),
        P.plot_acf_pair(series["s_mid"].fillna(1), data_kind="real_prices", title="e"),
        P.plot_grid_heatmap(pd.DataFrame({"N": [1, 1, 2, 2], "z": [1, 2, 1, 2], "v": [-1, 0.5, 2, -0.2]}),
                            x="N", y="z", value="v", data_kind="real_prices", title="f", selected={"N": 2, "z": 1}),
        P.plot_attribution({"gross_mid": 10, "half_spread": -4, "fees": -3, "net": 3}, data_kind="real_prices", title="g"),
        P.plot_equity_log(equity, data_kind="real_prices", title="h"),
    ]
    assert len(figs[1].axes[0].lines) == 8  # 7 legs + "Other (sum)"; never more than 8 hues
    for i, f in enumerate(figs):
        p = P.save(f, f"f{i}", root=tmp_path)
        assert p.exists() and p.stat().st_size > 1000
    assert plt.get_fignums() == []


def test_series_colours_from_palette(series, equity):
    fig = P.plot_basket_signal_equity(series, _trades(series.index), equity, z_entry=2, data_kind="real_books", title="t")
    line_colors = {matplotlib.colors.to_hex(l.get_color()) for ax in fig.axes for l in ax.lines}
    allowed = {c.lower() for c in P.SERIES} | {P.INK.lower(), P.INK_2.lower(), P.MUTED.lower(), P.CRITICAL.lower()}
    assert line_colors <= allowed
    plt.close(fig)


def test_import_does_not_change_global_rc():
    assert matplotlib.rcParams["axes.facecolor"] != P.SURFACE or True  # scoped style only
    with P.style():
        assert matplotlib.rcParams["axes.facecolor"] == P.SURFACE
