"""Notebook-level helpers: assemble panels and summary tables from the tested building blocks.

Everything here composes ``data_io``, ``stats_tools``, ``arb_engine`` and ``metrics`` - it adds
no new statistics, only the table layouts the notebooks print, so the notebooks stay thin.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from . import RESULTS_ROOT, code_version
from . import stats_tools as st
from .arb_engine import basket_cost_hurdle, edge_to_cost_ratio, executable_edges
from .config import Basket, MarketsConfig, basket_fee_per_unit, get_basket, load_markets
from .data_io import DataUnavailable, DatasetInfo, load_real_books, load_real_prices, prices_grid

warnings.filterwarnings("ignore", category=FutureWarning)


def basket_overview(cfg: MarketsConfig, basket_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for bid in basket_ids:
        b = get_basket(bid, cfg)
        rows.append({"basket": bid, "title": b.title, "legs": b.n_legs, "negRisk": b.neg_risk,
                     "augmented": b.neg_risk_augmented, "excluded markets": len(b.excluded),
                     "taker fee rate": max(l.fee.rate for l in b.legs), "complete partition": b.is_complete_partition,
                     "structural discount expected": b.structural_discount_expected})
    return pd.DataFrame(rows).set_index("basket")


def leg_table(b: Basket) -> pd.DataFrame:
    return pd.DataFrame([{"leg": l.label, "leg_id": l.leg_id, "fee rate": l.fee.rate, "fee source": l.fee.source,
                          "tick": l.tick_size, "accepting orders": l.accepting_orders} for l in b.legs]).set_index("leg")


def price_panel(basket_id: str, cfg: MarketsConfig | None = None, fidelity_s: int = 60) -> tuple[pd.DataFrame, DatasetInfo]:
    """Wide 1-minute grid of real leg prices (complete rows only) + provenance."""
    ds = load_real_prices(basket_id, cfg=cfg, fidelity_s=fidelity_s, calibrate_from_books=False)
    return prices_grid(basket_id, fidelity_s=fidelity_s, cfg=cfg), ds.info


def book_sums(basket_id: str, cfg: MarketsConfig | None = None, grid: str = "10s",
              max_gap: str = "60s") -> tuple[pd.DataFrame, pd.Series, DatasetInfo] | None:
    """S_mid / S_bid / S_ask on a grid from the recorded live books (None if nothing recorded)."""
    try:
        ds = load_real_books(basket_id, cfg=cfg)
    except DataUnavailable:
        return None
    b = ds.basket
    wide, seg = st.resample_locf(ds.top_of_book(), [l.leg_id for l in b.legs], grid=grid, max_gap=max_gap)
    sums = st.basket_sums(wide)
    return sums, seg, ds.info


def unit_root_table(series: dict[str, pd.Series], *, alpha: float = 0.05) -> pd.DataFrame:
    """ADF (H0 unit root) + KPSS (H0 stationary) per series, Holm-adjusted ADF p-values."""
    rows = []
    for name, x in series.items():
        x = pd.Series(x).dropna()
        a = st.adf_test(x)
        k = st.kpss_test(x)
        rows.append({"series": name, "mean": float(x.mean()), "std": float(x.std()), "ADF stat": a["stat"],
                     "ADF p": a["pvalue"], "KPSS p": k["pvalue_str"], "_kpss_p": k["pvalue"]})
    df = pd.DataFrame(rows).set_index("series")
    df["ADF p (Holm)"] = st.holm(df["ADF p"].to_numpy())
    df["verdict"] = [st.adf_kpss_verdict(a, k, alpha) for a, k in zip(df["ADF p (Holm)"], df["_kpss_p"])]
    df["mean-reverting (ADF)"] = df["ADF p (Holm)"] < alpha
    return df.drop(columns="_kpss_p")


def half_life_table(s: pd.Series, grids: Sequence[str] = ("1min", "5min", "15min"), n_boot: int = 200) -> pd.DataFrame:
    rows = []
    for g in grids:
        x = s.resample(g).last().dropna() if g != "1min" else s.dropna()
        dt = pd.Timedelta(g).total_seconds()
        f = st.ar1_fit(x, dt, n_boot=n_boot, seed=0)
        lo, hi = (f["ci_boot"] or (math.nan, math.nan)) if isinstance(f.get("ci_boot"), (tuple, list)) else (math.nan, math.nan)
        rows.append({"grid": g, "phi": f["phi"], "half-life (min)": f["half_life_s"] / 60,
                     "delta-method SE (min)": f["half_life_se_s"] / 60, "bootstrap 95% lo (min)": lo / 60 if lo == lo else lo,
                     "bootstrap 95% hi (min)": hi / 60 if hi == hi else hi, "sigma_inf": f["sigma_inf"], "n": f["nobs"]})
    return pd.DataFrame(rows).set_index("grid")


def basket_stats_table(cfg: MarketsConfig, basket_ids: Sequence[str]) -> pd.DataFrame:
    """One row per basket: level, dispersion, unit-root verdict, half-life, overround test."""
    rows = []
    for bid in basket_ids:
        try:
            w, _ = price_panel(bid, cfg)
        except DataUnavailable:
            continue
        s = w.sum(axis=1)
        a, k = st.adf_test(s), st.kpss_test(s)
        f = st.ar1_fit(s, 60.0)
        h = st.hac_mean_test(s, 1.0)
        b = get_basket(bid, cfg)
        rows.append({"basket": bid, "legs": b.n_legs, "minutes": len(s), "mean ΣP": s.mean(), "sd ΣP": s.std(),
                     "q01": s.quantile(0.01), "q99": s.quantile(0.99), "ADF p": a["pvalue"], "KPSS p": k["pvalue_str"],
                     "_kpss": k["pvalue"], "half-life (min)": f["half_life_s"] / 60, "HAC t (mean=1)": h["t"],
                     "augmented": bool(b.neg_risk_augmented)})
    df = pd.DataFrame(rows).set_index("basket")
    df["ADF p (Holm)"] = st.holm(df["ADF p"].to_numpy())
    df["verdict"] = [st.adf_kpss_verdict(a, k) for a, k in zip(df["ADF p (Holm)"], df["_kpss"])]
    return df.drop(columns="_kpss")


def variance_ratio_table(series: dict[str, pd.Series], qs: Sequence[int] = (2, 5, 15, 60, 240)) -> pd.DataFrame:
    rows = []
    for name, x in series.items():
        x = pd.Series(x).dropna().to_numpy(float)
        row = {"series": name}
        for q in qs:
            vr, z, p = st.variance_ratio(x, q)
            row[f"VR({q})"] = vr
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def band_stats(sums: pd.DataFrame, b: Basket) -> dict[str, float]:
    """Share of recorded time the executable sums sit outside the no-arbitrage band."""
    v = sums.dropna(subset=["s_bid", "s_ask"])
    fee = basket_fee_per_unit(b.fees, np.full(b.n_legs, 1.0 / b.n_legs))  # rough unit fee for the 'after fees' row
    return {"rows": len(v), "share S_ask<1": float((v["s_ask"] < 1).mean()), "share S_bid>1": float((v["s_bid"] > 1).mean()),
            "share S_ask<1-fees": float((v["s_ask"] < 1 - fee).mean()), "share S_bid>1+fees": float((v["s_bid"] > 1 + fee).mean()),
            "median spread_sum": float((v["s_ask"] - v["s_bid"]).median()),
            "min S_ask": float(v["s_ask"].min()), "max S_bid": float(v["s_bid"].max())}


def band_sentence(hours: float, min_ask: float, max_bid: float, share_ask_below: float, share_bid_above: float,
                  arb_trades: int | None = None, share_beyond_fees: float | None = None) -> str:
    """Plain-language summary of how often the live books left the no-arbitrage band (used by notebooks and README)."""
    head = f"In {hours:.1f} hours of recorded live order books"
    if min_ask >= 1 and max_bid <= 1:
        return (f"{head} a full YES set never cost less than ${min_ask:.3f} and never sold for more than ${max_bid:.3f}: "
                "no risk-free trade was available.")
    parts = []
    if max_bid > 1:
        parts.append(f"a full YES set sold for more than $1 only {share_bid_above:.1%} of the time (at most ${max_bid:.3f})")
    if min_ask < 1:
        parts.append(f"a full YES set cost less than $1 only {share_ask_below:.1%} of the time (at least ${min_ask:.3f})")
    if arb_trades is not None:
        tail = ("after taker fees and a half-second order delay the simulator found no profitable risk-free trade."
                if arb_trades == 0 else f"after taker fees and a half-second order delay the simulator found {arb_trades} "
                                        "risk-free trades.")
    elif share_beyond_fees is not None and share_beyond_fees == 0:
        tail = "and never by more than the taker fees, so there was no free money after costs."
    else:
        tail = f"and by more than the taker fees {share_beyond_fees or 0:.2%} of the time."
    return f"{head} {' and '.join(parts)}; {tail}" if arb_trades is not None else f"{head} {' and '.join(parts)}, {tail}"


def hurdle_table(cfg: MarketsConfig, basket_ids: Sequence[str]) -> pd.DataFrame:
    """Round-trip cost hurdle vs typical S excursion (ECR, pro-tip 2), per basket.

    Spreads come from the recorded live books when available (else one tick per leg);
    sigma(S) from the 1-minute real price history."""
    rows = []
    for bid in basket_ids:
        b = get_basket(bid, cfg)
        try:
            w, _ = price_panel(bid, cfg)
        except DataUnavailable:
            continue
        mids = w.mean().reindex([l.leg_id for l in b.legs]).to_numpy(float)
        bs = book_sums(bid, cfg)
        if bs is not None and len(bs[0].dropna(subset=["s_bid", "s_ask"])):
            spread = float((bs[0]["s_ask"] - bs[0]["s_bid"]).median())
            src = "recorded books"
        else:
            spread = float(sum(0.001 if (m < 0.04 or m > 0.96) else l.tick_size for m, l in zip(mids, b.legs)))
            src = "1 tick/leg"
        bid_ = mids - spread / (2 * b.n_legs)
        ask_ = mids + spread / (2 * b.n_legs)
        h = basket_cost_hurdle(bid_, ask_, b.fees)
        s = w.sum(axis=1)
        rows.append({"basket": bid, "spread_sum": spread, "spread source": src, "fee/unit F": float(h["fee_entry_long"]),
                     "round-trip hurdle": float(h["hurdle_long"]), "sd ΣP (1-min)": float(s.std()),
                     "ECR = sd/hurdle": edge_to_cost_ratio(s.to_numpy(), float(h["hurdle_long"]))})
    return pd.DataFrame(rows).set_index("basket")


def save_dataset_info(key: str, infos: Iterable[DatasetInfo], extra: dict | None = None) -> Path:
    infos = list(infos)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    out = {"notebook": key, "data_kinds": sorted({i.kind.value for i in infos}),
           "baskets": sorted({i.basket_id for i in infos}), "datasets": [i.to_json() for i in infos],
           "synthetic": any(i.synthetic for i in infos), "code_version": code_version(), **(extra or {})}
    p = RESULTS_ROOT / f"dataset_info_{key}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    return p


def fmt_p(p: float) -> str:
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"
