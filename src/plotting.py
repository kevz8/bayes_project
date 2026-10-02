"""Matplotlib figures for the notebooks and README, in one consistent visual system.

Design rules (validated reference palette from the dataviz method):

* categorical hues in a FIXED order, never cycled past 8 (extra legs fold into "Other");
* never a twin y-axis - two measures become stacked panels sharing the x-axis;
* thin lines, >= 7 pt markers, a recessive y-only hairline grid, no top/right spines;
* text always in ink colours, never in a series colour; a legend whenever >= 2 series;
* every figure is stamped with its data provenance (REAL vs SYNTHETIC) by ``apply_data_label``.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import FIGURES_ROOT

# --------------------------------------------------------------------------- tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
DIVERGING = ("#256abf", "#86b6ef", "#f0efec", "#ef9a99", "#d03b3b")
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"
BAND = "#cde2fb"

DATA_LABELS = {
    "real_books": "REAL ORDER BOOKS",
    "real_prices": "REAL PRICE HISTORY",
    "real_prices_modelled_books": "REAL PRICE HISTORY + MODELLED BOOKS",
    "synthetic": "SYNTHETIC DATA — NOT REAL MARKET DATA",
}

_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=list(SERIES)), "lines.linewidth": 1.6, "lines.markersize": 7,
    "font.family": "sans-serif", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5,
}


@contextlib.contextmanager
def style() -> Iterator[None]:
    """Scoped rcParams: importing this module never changes global matplotlib state."""
    with mpl.rc_context(_RC):
        yield


def _fig(nrows: int = 1, ncols: int = 1, **kw: Any) -> tuple[Figure, Any]:
    with style():
        fig, axes = plt.subplots(nrows, ncols, **kw)
    return fig, axes


def _restyle(fig: Figure) -> None:
    """Apply the axis styling explicitly (rc_context only affects objects created inside it)."""
    for ax in fig.axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.title.set_color(INK)
        ax.xaxis.label.set_color(INK_2)
        ax.yaxis.label.set_color(INK_2)
    fig.patch.set_facecolor(SURFACE)


def _legend(ax: Any, **kw: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) >= 2:
        leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, **kw)
        leg.set_zorder(5)


def apply_data_label(fig: Figure, data_kind: str, title: str | None = None) -> None:
    """Stamp provenance: a suptitle prefix, plus a diagonal watermark for synthetic data."""
    prefix = DATA_LABELS.get(data_kind, data_kind.upper())
    fig.suptitle(f"{prefix} — {title}" if title else prefix, x=0.01, ha="left", fontsize=11,
                 fontweight="semibold", color=CRITICAL if data_kind == "synthetic" else INK)
    if data_kind == "synthetic":
        fig.text(0.5, 0.5, "SYNTHETIC", fontsize=64, color=MUTED, alpha=0.12, rotation=30,
                 ha="center", va="center", zorder=0)


def _finish(fig: Figure, data_kind: str, title: str | None) -> Figure:
    _restyle(fig)
    apply_data_label(fig, data_kind, title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- headline figure
def plot_basket_signal_equity(series: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame, *,
                              z_entry: float, data_kind: str, title: str, show_bid_ask: bool = True) -> Figure:
    """The blueprint figure: moving basket sum with entries/exits (top) over account equity (bottom).

    ``series``: DatetimeIndex with ``s_mid, mu, sigma`` (optionally ``s_bid, s_ask``).
    ``trades``: ``t_entry, t_exit, side`` (+1 long YES set, -1 short via NO set).
    ``equity``: DatetimeIndex with ``equity_liq`` (optionally ``equity_mid``).
    """
    fig, (ax, ax2) = _fig(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    s = series
    if show_bid_ask and {"s_bid", "s_ask"} <= set(s.columns):
        ax.fill_between(s.index, s["s_bid"], s["s_ask"], color=GRID, alpha=0.8, linewidth=0,
                        label="executable band [ΣBid, ΣAsk]", step="post")
    if {"mu", "sigma"} <= set(s.columns):
        lo, hi = s["mu"] - z_entry * s["sigma"], s["mu"] + z_entry * s["sigma"]
        ax.fill_between(s.index, lo, hi, color=BAND, alpha=0.6, linewidth=0, label=f"μ ± {z_entry:g}σ")
        ax.plot(s.index, s["mu"], color=INK_2, linewidth=1.1, linestyle="--", label="rolling mean μ")
    ax.plot(s.index, s["s_mid"], color=SERIES[0], linewidth=1.3, label="basket sum ΣMid")
    ax.axhline(1.0, color=MUTED, linewidth=0.9, zorder=1)
    ax.annotate("1.00", xy=(1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(3, 0),
                textcoords="offset points", va="center", color=MUTED, fontsize=8)

    if trades is not None and len(trades):
        sm = s["s_mid"].dropna()

        def at(ts: pd.Series) -> np.ndarray:
            ts = pd.to_datetime(ts).dropna()
            if sm.empty or ts.empty:
                return np.array([])
            idx = sm.index.get_indexer(ts, method="ffill")
            return np.where(idx >= 0, sm.to_numpy()[np.clip(idx, 0, None)], np.nan)

        for side, marker, color, name in ((1, "^", SERIES[2], "enter long (buy YES set)"),
                                          (-1, "v", SERIES[1], "enter short (buy NO set)")):
            tt = trades.loc[trades["side"] == side, "t_entry"]
            if len(tt):
                ax.scatter(pd.to_datetime(tt), at(tt), marker=marker, s=60, color=color, edgecolor=SURFACE,
                           linewidth=1.2, zorder=4, label=name)
        te = trades["t_exit"].dropna() if "t_exit" in trades else pd.Series(dtype="datetime64[ns]")
        if len(te):
            ax.scatter(pd.to_datetime(te), at(te), marker="x", s=46, color=INK, linewidth=1.4, zorder=4, label="exit")
    ax.set_ylabel("Σ YES prices")
    ax.set_title("Basket sum and z-score signals")
    _legend(ax, loc="upper left", ncol=3)

    eq = equity["equity_liq"]
    ax2.plot(eq.index, eq, color=SERIES[0], linewidth=1.5, label="equity (liquidation marks)")
    if "equity_mid" in equity:
        ax2.plot(equity.index, equity["equity_mid"], color=INK_2, linewidth=1.1, linestyle="--",
                 label="equity (mid marks)")
    peak = eq.cummax()
    ax2.fill_between(eq.index, eq, peak, where=eq < peak, color=CRITICAL, alpha=0.12, linewidth=0, label="drawdown")
    ax2.set_ylabel("Account equity (USD)")
    ax2.set_title("Simulated account")
    _legend(ax2, loc="upper left", ncol=3)
    return _finish(fig, data_kind, title)


# --------------------------------------------------------------------------- EDA figures
def plot_sum_band(sums: pd.DataFrame, *, data_kind: str, title: str) -> Figure:
    fig, ax = _fig(figsize=(11, 4))
    if {"s_bid", "s_ask"} <= set(sums.columns):
        ax.fill_between(sums.index, sums["s_bid"], sums["s_ask"], color=GRID, alpha=0.9, linewidth=0,
                        label="[ΣBid, ΣAsk]", step="post")
    ax.plot(sums.index, sums["s_mid"], color=SERIES[0], linewidth=1.2, label="ΣMid")
    ax.axhline(1.0, color=MUTED, linewidth=0.9)
    ax.set_ylabel("Σ YES prices")
    _legend(ax, loc="upper left")
    return _finish(fig, data_kind, title)


def plot_legs(prices_wide: pd.DataFrame, *, data_kind: str, title: str, max_series: int = 8) -> Figure:
    fig, ax = _fig(figsize=(11, 4))
    cols = list(prices_wide.columns)
    if len(cols) > max_series:
        order = prices_wide.mean().sort_values(ascending=False).index
        keep = list(order[: max_series - 1])
        data = prices_wide[keep].copy()
        data["Other (sum)"] = prices_wide[[c for c in cols if c not in keep]].sum(axis=1, min_count=1)
    else:
        data = prices_wide
    for i, c in enumerate(data.columns):
        ax.plot(data.index, data[c], color=SERIES[i % len(SERIES)], linewidth=1.2, label=str(c))
    ax.set_ylabel("YES price")
    ax.set_ylim(0, 1)
    _legend(ax, loc="upper left", ncol=min(4, len(data.columns)))
    return _finish(fig, data_kind, title)


def plot_sum_distribution(s: pd.Series, *, data_kind: str, title: str) -> Figure:
    from scipy import stats

    x = pd.Series(s).dropna().to_numpy(float)
    fig, (ax, ax2) = _fig(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": (3, 2)})
    ax.hist(x, bins=80, density=True, color=SERIES[0], alpha=0.35, edgecolor=SURFACE, linewidth=0.3,
            label="ΣMid histogram")
    if len(x) > 2 and np.std(x) > 0:
        grid = np.linspace(x.min(), x.max(), 400)
        ax.plot(grid, stats.gaussian_kde(x)(grid), color=SERIES[0], linewidth=1.6, label="KDE")
    ax.axvline(1.0, color=MUTED, linewidth=1, label="1.00 (no-arbitrage)")
    ax.axvline(np.mean(x), color=INK, linewidth=1, linestyle="--", label=f"mean {np.mean(x):.4f}")
    ax.set_xlabel("Σ YES mids")
    ax.set_title("Distribution of the basket sum")
    _legend(ax, loc="upper right")
    if len(x) > 2:
        (osm, osr), (slope, icpt, _) = stats.probplot(x, dist="norm")
        ax2.scatter(osm, osr, s=6, color=SERIES[0], alpha=0.5, linewidth=0)
        ax2.plot(osm, slope * osm + icpt, color=INK_2, linewidth=1)
    ax2.set_xlabel("normal quantiles")
    ax2.set_ylabel("sample quantiles")
    ax2.set_title("Normal QQ plot (fat tails = dislocations)")
    return _finish(fig, data_kind, title)


def plot_variogram(curves: Mapping[str, tuple[Sequence[float], Sequence[float]]], *, data_kind: str, title: str,
                   dt_label: str = "minutes") -> Figure:
    fig, ax = _fig(figsize=(8, 4.5))
    for i, (name, (lags, vals)) in enumerate(curves.items()):
        ax.plot(lags, vals, marker="o", markersize=4.5, color=SERIES[i % len(SERIES)], linewidth=1.4, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"lag q ({dt_label})")
    ax.set_ylabel("Var(x_t − x_{t−q})")
    ax.set_title("Variogram: random walks grow linearly, a mean-reverting sum levels off")
    _legend(ax, loc="upper left")
    return _finish(fig, data_kind, title)


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x[np.isfinite(x)]
    x = x - x.mean()
    d = float(np.dot(x, x))
    return np.array([1.0] + [float(np.dot(x[:-k], x[k:])) / d for k in range(1, nlags + 1)]) if d > 0 else np.full(nlags + 1, np.nan)


def plot_acf_pair(s: pd.Series, *, nlags: int = 60, data_kind: str, title: str) -> Figure:
    x = pd.Series(s).to_numpy(float)
    fig, (ax, ax2) = _fig(1, 2, figsize=(11, 3.8), sharey=True)
    for a, data, name in ((ax, x, "ACF of ΣMid"), (ax2, np.diff(x), "ACF of ΔΣMid")):
        r = _acf(data, nlags)
        a.vlines(np.arange(len(r)), 0, r, color=SERIES[0], linewidth=1.4)
        a.scatter(np.arange(len(r)), r, s=12, color=SERIES[0], zorder=3)
        n = np.isfinite(data).sum()
        if n > 0:
            a.axhspan(-1.96 / math.sqrt(n), 1.96 / math.sqrt(n), color=GRID, alpha=0.8, linewidth=0)
        a.axhline(0, color=AXIS, linewidth=0.8)
        a.set_title(name)
        a.set_xlabel("lag (grid steps)")
    return _finish(fig, data_kind, title)


def plot_grid_heatmap(grid: pd.DataFrame, *, x: str, y: str, value: str, data_kind: str, title: str,
                      fmt: str = "{:.2f}", diverging: bool = True, selected: Mapping[str, Any] | None = None) -> Figure:
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    pv = grid.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = _fig(figsize=(1.2 * len(pv.columns) + 3, 0.6 * len(pv.index) + 2.2))
    data = pv.to_numpy(float)
    finite = data[np.isfinite(data)]
    if diverging and finite.size and finite.min() < 0 < finite.max():
        cmap = LinearSegmentedColormap.from_list("div", DIVERGING[::-1])
        lim = float(np.abs(finite).max())
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    else:
        cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
        norm = None
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(pv.columns)), [str(c) for c in pv.columns])
    ax.set_yticks(range(len(pv.index)), [str(i) for i in pv.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8, color=INK)
    if selected is not None and selected.get(x) in list(pv.columns) and selected.get(y) in list(pv.index):
        j, i = list(pv.columns).index(selected[x]), list(pv.index).index(selected[y])
        ax.add_patch(mpl.patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=2))
    fig.colorbar(im, ax=ax, label=value, shrink=0.85)
    ax.grid(False)
    fig = _finish(fig, data_kind, title)
    ax.yaxis.grid(False)
    return fig


def plot_attribution(attribution: Mapping[str, float], *, data_kind: str, title: str) -> Figure:
    """Horizontal waterfall from gross mid-to-mid P&L through each friction to net P&L."""
    items = [(k, float(v)) for k, v in attribution.items() if k != "net" and np.isfinite(v)]
    net = float(attribution.get("net", sum(v for _, v in items)))
    fig, ax = _fig(figsize=(9, 0.45 * (len(items) + 1) + 1.6))
    cum = 0.0
    labels = []
    for i, (k, v) in enumerate(items):
        ax.barh(i, v, left=cum, color=GOOD if v >= 0 else CRITICAL, height=0.6, edgecolor=SURFACE, linewidth=1)
        ax.text(cum + v, i, f" {v:+,.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK_2)
        cum += v
        labels.append(k.replace("_", " "))
    ax.barh(len(items), net, color=INK, height=0.6)
    ax.text(net, len(items), f" {net:+,.2f}", va="center", ha="left" if net >= 0 else "right", fontsize=9, color=INK)
    labels.append("net P&L")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("USD")
    fig = _finish(fig, data_kind, title)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    return fig


def plot_equity_log(equity: pd.DataFrame, *, data_kind: str, title: str) -> Figure:
    fig, (ax, ax2) = _fig(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": (2, 1)})
    ax.plot(equity.index, equity["equity_liq"], color=SERIES[0], linewidth=1.5, label="liquidation marks")
    if "equity_mid" in equity:
        ax.plot(equity.index, equity["equity_mid"], color=INK_2, linewidth=1.1, linestyle="--", label="mid marks")
    ax.set_yscale("log")
    ax.set_ylabel("equity (USD, log)")
    ax.set_title("Compounding equity curve")
    _legend(ax, loc="upper left")
    dd = equity["equity_liq"] / equity["equity_liq"].cummax() - 1
    ax2.fill_between(dd.index, dd * 100, 0, color=CRITICAL, alpha=0.25, linewidth=0)
    ax2.plot(dd.index, dd * 100, color=CRITICAL, linewidth=1)
    ax2.set_ylabel("drawdown (%)")
    return _finish(fig, data_kind, title)


def save(fig: Figure, name: str, root: Path = FIGURES_ROOT, dpi: int = 150) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def show(fig: Figure, name: str | None = None, root: Path = FIGURES_ROOT) -> Path | None:
    """Notebook helper: render the figure inline, then save it (if ``name``) and close it."""
    try:
        from IPython.display import display

        display(fig)
    except ImportError:  # pragma: no cover - outside IPython
        pass
    if name:
        return save(fig, name, root=root)
    plt.close(fig)
    return None
