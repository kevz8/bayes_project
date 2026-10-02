"""Tests for src/stats_tools.py: LOCF grids, basket sums, unit-root/VR/AR(1) statistics,
cointegration, regressions and model selection. Deterministic (fixed seeds), offline."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src import stats_tools as st
from src.arb_engine import BasketState
from src.events import TOB_COLUMNS
from src.synthetic import SyntheticParams, synthetic_tob

T0 = pd.Timestamp("2026-01-01T00:00:00Z")
NAN = math.nan


def _ns(spec: str) -> int:
    """'mm:ss' offset from T0 -> epoch ns."""
    m, s = spec.split(":")
    return (T0 + pd.Timedelta(minutes=int(m), seconds=int(s))).value


def _tob(rows: list[tuple]) -> pd.DataFrame:
    """rows: (mm:ss, leg, bid, ask[, clean]) -> TOB_COLUMNS frame (sizes 100/200)."""
    recs = []
    for r in rows:
        t, leg, bid, ask = r[:4]
        clean = r[4] if len(r) > 4 else True
        recs.append((_ns(t), leg, bid, ask, 100.0, 200.0, clean))
    return pd.DataFrame(recs, columns=list(TOB_COLUMNS))


def _at(wide: pd.DataFrame, field: str, leg: str, spec: str) -> float:
    return wide[(field, leg)].loc[T0 + pd.Timedelta(minutes=int(spec.split(":")[0]))]


def _ar1(n: int, phi: float, seed: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(1 - phi * phi) if abs(phi) < 1 else e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x + mu


def _rw(n: int, seed: int, sigma: float = 1.0) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(0.0, sigma, n))


# =========================================================================== resample_locf

def test_locf_is_as_of_never_interpolates_nor_looks_ahead():
    rows = [("00:05", "B", 0.30, 0.33), ("00:30", "A", 0.40, 0.42),
            ("02:10", "A", 0.50, 0.52), ("04:00", "A", 0.60, 0.62)]
    wide, seg = st.resample_locf(_tob(rows), max_gap="1h")
    assert list(wide.index) == [T0 + pd.Timedelta(minutes=m) for m in (1, 2, 3, 4)]
    assert str(wide.index.tz) == "UTC"
    assert list(wide.columns.names) == ["field", "leg"]
    assert set(wide.columns.get_level_values(0)) == set(st.TOB_FIELDS)
    # 00:02 sits between the 00:30 and 02:10 updates: the old quote, not a blend of the two
    assert [_at(wide, "bid", "A", f"{m:02d}:00") for m in (1, 2, 3, 4)] == [0.40, 0.40, 0.50, 0.60]
    assert _at(wide, "mid", "A", "02:00") == pytest.approx(0.41)
    # an update exactly on a grid point (t <= g) is used at that point
    assert _at(wide, "age_s", "A", "04:00") == 0.0
    assert _at(wide, "age_s", "A", "02:00") == pytest.approx(90.0)
    assert _at(wide, "age_s", "B", "04:00") == pytest.approx(235.0)
    assert _at(wide, "bid_sz", "A", "03:00") == 100.0 and _at(wide, "ask_sz", "A", "03:00") == 200.0
    assert wide[("clean", "A")].dtype == bool and wide[("clean", "A")].all()
    assert (seg == 0).all() and seg.dtype == np.int64

    # prefix invariance: a later update never changes earlier grid values
    later, _ = st.resample_locf(_tob(rows + [("03:30", "A", 0.90, 0.92)]), max_gap="1h")
    pd.testing.assert_frame_equal(later.iloc[:3], wide.iloc[:3])
    assert _at(later, "bid", "A", "04:00") == 0.60  # 04:00 row still wins (later in time)


def test_locf_gap_starts_new_segment_and_nothing_is_carried():
    rows = [("00:10", "A", 0.40, 0.42), ("00:10", "B", 0.50, 0.52), ("00:50", "A", 0.41, 0.43),
            # > 60 s without any update: 00:50 -> 05:30
            ("05:30", "A", 0.45, 0.47), ("06:20", "B", 0.55, 0.57), ("07:05", "A", 0.46, 0.48)]
    wide, seg = st.resample_locf(_tob(rows), grid="1min", max_gap="60s")
    assert seg.tolist() == [0, -1, -1, -1, -1, 1, 1]  # 00:01 .. 00:07
    # 00:01 is within max_gap of the last update: still live, B carried (age 50 s)
    assert _at(wide, "bid", "B", "01:00") == 0.50 and _at(wide, "age_s", "B", "01:00") == pytest.approx(50)
    gap = wide[seg.to_numpy() == -1]
    assert gap.drop(columns="clean", level=0).isna().all().all()
    assert not gap["clean"].to_numpy().any()
    # after the gap B has no update in the new segment -> NaN, not the pre-gap 0.50
    assert _at(wide, "bid", "A", "06:00") == 0.45
    assert math.isnan(_at(wide, "bid", "B", "06:00")) and not _at(wide, "clean", "B", "06:00")
    assert _at(wide, "bid", "B", "07:00") == 0.55
    assert _at(wide, "age_s", "A", "07:00") == pytest.approx(90.0)  # stale but live: kept


def test_locf_unclean_rows_propagate_nan_until_next_clean_update():
    rows = [("00:10", "A", 0.40, 0.42), ("00:20", "B", 0.50, 0.52), ("00:50", "B", 0.50, 0.52),
            ("01:30", "A", NAN, NAN, False),  # explicit session-end style marker
            ("01:40", "B", 0.51, 0.53), ("02:30", "A", 0.44, 0.46), ("02:40", "B", 0.52, 0.54, False),
            ("03:10", "A", 0.45, 0.47)]       # unclean row that still carries old quotes ^
    wide, seg = st.resample_locf(_tob(rows))
    assert (seg == 0).all()
    assert _at(wide, "bid", "A", "01:00") == 0.40 and _at(wide, "clean", "A", "01:00")
    for f in ("bid", "ask", "bid_sz", "ask_sz", "mid", "age_s"):
        assert math.isnan(_at(wide, f, "A", "02:00")), f
    assert not _at(wide, "clean", "A", "02:00")
    assert _at(wide, "bid", "A", "03:00") == 0.44
    # an unclean row is unusable even if it repeats the old prices
    assert math.isnan(_at(wide, "bid", "B", "03:00")) and not _at(wide, "clean", "B", "03:00")


def _naive_locf(tob: pd.DataFrame, legs, grid_ns, gap_ns):
    """Brute-force reference: per grid point, scan all rows."""
    t = tob["t_ns"].to_numpy()
    out = {leg: [] for leg in legs}
    seg_of = np.concatenate([[0], np.cumsum(np.diff(np.sort(t)) > gap_ns)])
    ts = np.sort(t)
    for g in grid_ns:
        last_any = ts[ts <= g].max()
        if g - last_any > gap_ns:
            for leg in legs:
                out[leg].append(NAN)
            continue
        cur = seg_of[np.flatnonzero(ts == last_any)[-1]]
        start = ts[np.flatnonzero(seg_of == cur)[0]]
        for leg in legs:
            sub = tob[(tob["leg"] == leg) & (tob["t_ns"] <= g) & (tob["t_ns"] >= start)]
            if sub.empty or not sub["clean"].iloc[-1]:
                out[leg].append(NAN)
            else:
                out[leg].append(sub["bid"].iloc[-1])
    return out


def test_locf_matches_brute_force_reference_on_random_stream():
    rng = np.random.default_rng(11)
    n = 300
    dt = rng.exponential(12.0, n)
    dt[rng.random(n) < 0.03] += 150.0  # occasional recording gaps
    t = (T0.value + np.cumsum(dt * 1e9)).astype(np.int64)
    legs = rng.choice(["A", "B", "C"], n)
    bid = np.round(rng.uniform(0.1, 0.5, n), 2)
    bid[rng.random(n) < 0.1] = NAN
    tob = pd.DataFrame({"t_ns": t, "leg": legs, "bid": bid, "ask": bid + 0.01, "bid_sz": 1.0,
                        "ask_sz": 1.0, "clean": rng.random(n) > 0.1})
    tob = tob.sample(frac=1.0, random_state=3)  # input order must not matter (distinct times)
    wide, seg = st.resample_locf(tob, legs=["A", "B", "C"], grid="30s", max_gap="60s")
    ref = _naive_locf(tob.sort_values("t_ns"), ["A", "B", "C"], wide.index.asi8, 60 * 10**9)
    for leg in "ABC":
        np.testing.assert_array_equal(wide[("bid", leg)].to_numpy(), np.asarray(ref[leg]))
    assert (seg == -1).any() and seg.max() >= 1
    assert seg[seg >= 0].is_monotonic_increasing


def test_locf_leg_selection_order_and_validation(caplog):
    rows = [("00:10", "A", 0.40, 0.42), ("00:20", "B", 0.50, 0.52), ("00:40", "A", 0.41, 0.43),
            ("01:20", "A", 0.42, 0.44), ("02:10", "A", 0.43, 0.45)]
    wide, _ = st.resample_locf(_tob(rows), legs=["B", "C"])
    assert list(wide["bid"].columns) == ["B", "C"] and len(wide) == 2
    assert wide[("bid", "B")].tolist() == [0.50, 0.50]
    assert wide[("bid", "C")].isna().all() and not wide[("clean", "C")].any()
    assert "C" in caplog.text
    with pytest.raises(ValueError, match="lacks columns"):
        st.resample_locf(_tob(rows).drop(columns="clean"))
    with pytest.raises(ValueError, match="empty"):
        st.resample_locf(_tob(rows).iloc[:0])


def test_locf_on_synthetic_tob_keeps_schema_and_label():
    tob = synthetic_tob(SyntheticParams(duration_s=3 * 3600, seed=5, gap_prob_per_hour=2.0))
    wide, seg = st.resample_locf(tob)
    assert wide.attrs["synthetic"] is True  # data-honesty label survives
    assert wide.index.freq == pd.Timedelta("1min")
    assert len(wide["bid"].columns) == 5
    bs = st.basket_sums(wide)
    assert bs.attrs["synthetic"] is True
    assert not bs.loc[seg.to_numpy() == -1, "valid"].any()
    ok = bs["valid"].to_numpy()
    assert ok.mean() > 0.8
    assert ((bs["s_ask"] - bs["s_bid"])[ok] > 0).all()
    assert bs["s_mid"][ok].between(0.8, 1.2).all()


# =========================================================================== prices_wide

def test_prices_wide_locf_and_gap_segments():
    h = 3600
    base = int(T0.value // 10**9)
    long = pd.DataFrame({
        "t": [base + 0, base + 0, base + h, base + 2 * h, base + 7 * h, base + 8 * h],
        "leg": ["x", "y", "x", "y", "x", "y"],
        "p": [0.40, 0.60, 0.42, 0.58, 0.45, 0.55],
    })
    wide, seg = st.prices_wide(long, grid="1h", max_gap="2h")
    assert list(wide.columns) == ["x", "y"]
    assert len(wide) == 9 and wide.index[0] == T0
    # 4h is exactly max_gap after the 2h update (still live); 5h and 6h are inside the gap
    assert seg.tolist() == [0, 0, 0, 0, 0, -1, -1, 1, 1]
    np.testing.assert_array_equal(wide["x"].to_numpy(), [0.40, 0.42, 0.42, 0.42, 0.42, NAN, NAN, 0.45, 0.45])
    np.testing.assert_array_equal(wide["y"].to_numpy(), [0.60, 0.60, 0.58, 0.58, 0.58, NAN, NAN, NAN, 0.55])
    # t_ns input gives the same frame
    wide2, seg2 = st.prices_wide(long.assign(t_ns=long["t"] * 10**9).drop(columns="t"), grid="1h", max_gap="2h")
    pd.testing.assert_frame_equal(wide, wide2)
    pd.testing.assert_series_equal(seg, seg2)
    with pytest.raises(ValueError, match="'t'"):
        st.prices_wide(long.drop(columns="t"))


# =========================================================================== basket_sums

def _wide(bids, asks, clean=None, ages=None, legs=("a", "b", "c")) -> pd.DataFrame:
    bids, asks = np.asarray(bids, float), np.asarray(asks, float)
    clean = np.ones(bids.shape, bool) if clean is None else np.asarray(clean, bool)
    ages = np.zeros(bids.shape) if ages is None else np.asarray(ages, float)
    data = {}
    for f, v in (("bid", bids), ("ask", asks), ("clean", clean), ("age_s", ages)):
        for k, leg in enumerate(legs):
            data[(f, leg)] = v[:, k]
    return pd.DataFrame(data)


def test_basket_sums_mid_policy_and_strict_executable_sums():
    bids = [[0.30, 0.20, 0.40],   # 0 two-sided
            [0.50, 0.45, NAN],    # 1 dust leg (ask 0.01) -> mid imputed 0.005
            [0.50, 0.45, NAN],    # 2 one-sided, not dust (ask 0.10) -> invalid
            [0.50, 0.45, 0.02],   # 3 missing ask -> invalid, counts 1 in s_ask
            [0.50, 0.45, 0.02],   # 4 leg b unknown (unclean) -> everything NaN
            [0.50, 0.45, NAN]]    # 5 dust exactly at the 0.02 boundary
    asks = [[0.32, 0.22, 0.44], [0.52, 0.47, 0.01], [0.52, 0.47, 0.10],
            [NAN, 0.47, 0.03], [0.52, 0.47, 0.03], [0.52, 0.47, 0.02]]
    clean = np.ones((6, 3), bool)
    clean[4, 1] = False
    ages = [[1, 5, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0], [7, NAN, 2], [0, 0, 0]]
    bs = st.basket_sums(_wide(bids, asks, clean, ages))
    assert list(bs.columns) == ["s_mid", "s_bid", "s_ask", "valid", "spread_sum", "n_imputed", "max_age_s"]
    np.testing.assert_allclose(bs["s_mid"], [0.94, 0.975, NAN, NAN, NAN, 0.98], equal_nan=True)
    np.testing.assert_allclose(bs["s_bid"], [0.90, 0.95, 0.95, 0.97, NAN, 0.95], equal_nan=True)
    np.testing.assert_allclose(bs["s_ask"], [0.98, 1.00, 1.09, 1.50, NAN, 1.01], equal_nan=True)
    np.testing.assert_allclose(bs["spread_sum"], bs["s_ask"] - bs["s_bid"], equal_nan=True)
    assert bs["valid"].tolist() == [True, True, False, False, False, True]
    assert bs["n_imputed"].tolist() == [0, 1, 0, 0, 0, 1]
    assert bs["max_age_s"].tolist()[:2] == [5.0, 0.0] and bs["max_age_s"].iloc[4] == 7.0
    # a looser dust threshold admits the 0.10 long shot
    assert st.basket_sums(_wide(bids, asks, clean), dust_max_ask=0.10)["valid"].iloc[2]


def test_basket_sums_agree_with_streaming_basket_state():
    rng = np.random.default_rng(4)
    n_rows, legs = 200, ("a", "b", "c", "d")
    bids = np.round(rng.uniform(0.0, 0.5, (n_rows, 4)), 3)
    asks = np.round(bids + rng.uniform(0.005, 0.05, (n_rows, 4)), 3)
    bids[rng.random((n_rows, 4)) < 0.15] = NAN
    asks[rng.random((n_rows, 4)) < 0.05] = NAN
    asks[rng.random((n_rows, 4)) < 0.1] = 0.015
    bs = st.basket_sums(_wide(bids, asks, legs=legs))
    state = BasketState(list(legs))
    for r in range(n_rows):
        for i in range(4):
            state.set_leg(i, bids[r, i], asks[r, i], 1.0, 1.0, r, True)
        snap = state.snapshot(r)
        assert bs["valid"].iloc[r] == snap.valid
        assert bs["n_imputed"].iloc[r] == snap.n_imputed
        assert bs["s_bid"].iloc[r] == pytest.approx(snap.s_bid, abs=1e-12)
        assert bs["s_ask"].iloc[r] == pytest.approx(snap.s_ask, abs=1e-12)
        if snap.valid:
            assert bs["s_mid"].iloc[r] == pytest.approx(snap.s_mid, abs=1e-12)
    assert 0 < bs["valid"].mean() < 1


def test_basket_sums_requires_bid_ask_fields():
    with pytest.raises(ValueError, match="bid"):
        st.basket_sums(pd.DataFrame({"a": [0.5]}))


# =========================================================================== unit roots

def test_adf_rejects_for_ar1_not_for_random_walk():
    ar = st.adf_test(_ar1(1000, 0.9, seed=1, mu=1.0))
    rw = st.adf_test(_rw(1000, seed=2))
    assert ar["pvalue"] < 0.01 and ar["stat"] < ar["crit"]["1%"]
    assert rw["pvalue"] > 0.10
    assert set(ar) >= {"stat", "pvalue", "lags", "nobs", "crit"}
    assert ar["nobs"] + ar["lags"] + 1 == 1000
    ct = st.adf_test(_rw(1000, seed=2), regression="ct", autolag=None, maxlag=2)
    assert ct["lags"] == 2 and ct["regression"] == "ct"


def test_kpss_reports_bounds_as_strings():
    stat = st.kpss_test(_ar1(1000, 0.3, seed=3))
    unit = st.kpss_test(_rw(1000, seed=4))
    assert stat["bounded"] and stat["pvalue_str"] == ">0.10" and stat["pvalue"] == pytest.approx(0.10)
    assert unit["bounded"] and unit["pvalue_str"] == "<0.01" and unit["pvalue"] == pytest.approx(0.01)
    assert set(unit["crit"]) == {"10%", "5%", "2.5%", "1%"}
    assert unit["lags"] > 0


def test_kpss_interior_pvalue_is_a_point_value():
    # find (deterministically) a moderately persistent series whose statistic is inside the table
    for seed in range(50):
        r = st.kpss_test(_ar1(400, 0.97, seed=seed))
        if not r["bounded"]:
            break
    else:  # pragma: no cover - would mean the table bounds changed
        pytest.fail("no interior KPSS p-value found")
    assert 0.01 < r["pvalue"] < 0.10 and r["pvalue_str"] == f"{r['pvalue']:.3f}"


@pytest.mark.parametrize("adf_p, kpss_p, verdict", [
    (0.01, 0.10, "stationary"),
    (0.40, 0.01, "unit_root"),
    (0.01, 0.01, "inconclusive"),
    (0.40, 0.10, "inconclusive"),
    (0.05, 0.10, "inconclusive"),  # p == alpha is not a rejection
    (NAN, 0.10, "inconclusive"),
])
def test_adf_kpss_verdict_table(adf_p, kpss_p, verdict):
    assert st.adf_kpss_verdict(adf_p, kpss_p) == verdict


def test_unit_root_inputs_are_validated():
    x = _ar1(300, 0.5, seed=5)
    padded = np.concatenate([[NAN, NAN], x, [NAN]])
    assert st.adf_test(padded) == st.adf_test(x)  # leading/trailing NaNs are stripped
    holed = x.copy()
    holed[150] = NAN
    with pytest.raises(ValueError, match="interior NaN"):
        st.adf_test(holed)
    with pytest.raises(ValueError, match="interior NaN"):
        st.kpss_test(holed)
    with pytest.raises(ValueError, match="too short"):
        st.adf_test(x[:10])
    with pytest.raises(ValueError, match="constant"):
        st.kpss_test(np.ones(100))


# =========================================================================== variance ratio / variogram

def test_variance_ratio_hand_computed():
    x = [0.0, 1.0, 3.0, 2.0, 4.0]
    # increments 1,2,-1,2: mu=1, var_a=2; 2-differences 3,1,1 -> (1,-1,-1), m=2*3*(1/2)=3 -> var_c=1
    vr, z, p = st.variance_ratio(x, 2, robust=False)
    assert vr == pytest.approx(0.5) and z == pytest.approx(-1.0)
    assert p == pytest.approx(2 * 0.15865525393145707)
    # robust: delta(1) = 4 * (1*0 + 4*1 + 1*4) / 6^2 = 8/9, theta = 8/9, z = -0.5 / sqrt(8/9/4)
    vr_r, z_r, _ = st.variance_ratio(x, 2, robust=True)
    assert vr_r == pytest.approx(0.5) and z_r == pytest.approx(-0.5 / math.sqrt(2 / 9))


def test_variance_ratio_random_walk_is_one():
    x = _rw(20000, seed=6)
    for q in (2, 5, 15):
        vr, z, p = st.variance_ratio(x, q)
        assert abs(vr - 1) < 0.05 and abs(z) < 3 and p > 0.003
        _, z_h, _ = st.variance_ratio(x, q, robust=False)
        assert z == pytest.approx(z_h, rel=0.1)  # iid increments: robust ~ homoskedastic


@pytest.mark.parametrize("q", [2, 5, 10])
def test_variance_ratio_ar1_matches_theory(q):
    phi = 0.8
    vr, z, p = st.variance_ratio(_ar1(50000, phi, seed=7), q)
    assert vr == pytest.approx((1 - phi**q) / (q * (1 - phi)), abs=0.03)
    assert z < -5 and p < 1e-6


def test_variance_ratio_validation():
    with pytest.raises(ValueError, match="q must be"):
        st.variance_ratio(_rw(100, 1), 1)
    with pytest.raises(ValueError, match="interior NaN"):
        st.variance_ratio([0, 1, NAN, 2, 3, 4, 5, 6], 2)


def test_variogram_hand_computed_and_nan_pairs():
    np.testing.assert_allclose(st.variogram([0.0, 1.0, 3.0, 2.0, 4.0], [1, 2, 10]), [2.0, 4 / 3, NAN])
    # complete pairs only: lag1 (1, 2) -> 0.5; lag2 one pair -> NaN; lag3 (2, 3) -> 0.5
    np.testing.assert_allclose(st.variogram([0.0, 1.0, NAN, 2.0, 4.0], [1, 2, 3]), [0.5, NAN, 0.5])


def test_variogram_linear_for_random_walk_flat_for_ar1():
    lags = [1, 5, 20, 50]
    v_rw = st.variogram(_rw(50000, seed=8, sigma=2.0), lags)
    np.testing.assert_allclose(v_rw / np.asarray(lags), 4.0, rtol=0.1)
    phi = 0.5
    v_ar = st.variogram(_ar1(50000, phi, seed=9), [10, 20, 50])
    np.testing.assert_allclose(v_ar, 2 / (1 - phi**2), rtol=0.05)
    assert v_ar[2] / v_ar[0] == pytest.approx(1.0, abs=0.05)


# =========================================================================== AR(1)

def test_ar1_fit_exact_geometric_decay():
    x = 1.0 + 100.0 * 0.5 ** np.arange(30)  # x_t = 0.5 + 0.5 x_{t-1} exactly
    r = st.ar1_fit(x, dt_s=60.0)
    assert r["phi"] == pytest.approx(0.5) and r["c"] == pytest.approx(0.5)
    assert r["mu"] == pytest.approx(1.0)
    assert r["half_life_s"] == pytest.approx(60.0)
    assert r["theta"] == pytest.approx(math.log(2) / 60.0)
    assert r["sigma_eps"] == pytest.approx(0.0, abs=1e-9) and r["nobs"] == 29
    assert r["ci_boot"] is None


def test_ar1_fit_recovers_phi_and_half_life():
    phi, n, dt = 0.95, 20000, 60.0
    r = st.ar1_fit(_ar1(n, phi, seed=10, mu=1.0, sigma=0.01), dt_s=dt)
    true_hl = -math.log(2) / math.log(phi) * dt  # 810.9 s
    assert abs(r["phi"] - phi) < 4 * r["phi_se"]
    assert r["phi_se"] == pytest.approx(math.sqrt((1 - phi**2) / n), rel=0.25)
    assert abs(r["half_life_s"] - true_hl) < 4 * r["half_life_se_s"]
    # delta method: ln2 / (phi ln(phi)^2) * se(phi) * dt
    assert r["half_life_se_s"] == pytest.approx(
        math.log(2) / (r["phi"] * math.log(r["phi"]) ** 2) * r["phi_se"] * dt)
    assert r["mu"] == pytest.approx(1.0, abs=0.005)
    assert r["sigma_inf"] == pytest.approx(0.01 / math.sqrt(1 - phi**2), rel=0.1)
    assert r["theta"] == pytest.approx(-math.log(phi) / dt, rel=0.1)
    assert r["phi_bias"] == pytest.approx(-(1 + 3 * r["phi"]) / (n - 1))
    assert "Kendall" in r["bias_note"]


def test_ar1_bootstrap_ci_covers_truth_and_is_reproducible():
    phi, dt = 0.9, 15.0
    x = _ar1(4000, phi, seed=12)
    r = st.ar1_fit(x, dt_s=dt, n_boot=300, seed=1)
    lo, hi = r["ci_phi_boot"]
    assert lo < phi < hi
    hl_true = -math.log(2) / math.log(phi) * dt
    assert r["ci_boot"][0] < hl_true < r["ci_boot"][1]
    assert r["ci_boot"][0] < r["half_life_s"] < r["ci_boot"][1]
    assert r["ci_boot"][0] == pytest.approx(-math.log(2) / math.log(lo) * dt)
    assert st.ar1_fit(x, dt_s=dt, n_boot=300, seed=1)["ci_boot"] == r["ci_boot"]


def test_ar1_fit_never_pairs_across_gaps():
    a = _ar1(500, 0.7, seed=13)
    b = _ar1(500, 0.7, seed=14) + 50.0  # a level jump that splicing would read as persistence
    r = st.ar1_fit(np.concatenate([a, [NAN], b]), dt_s=1.0)
    assert r["nobs"] == 998
    xp = np.concatenate([a[:-1], b[:-1]])
    xc = np.concatenate([a[1:], b[1:]])
    expect = np.linalg.lstsq(np.column_stack([np.ones(998), xp]), xc, rcond=None)[0]
    assert r["phi"] == pytest.approx(expect[1]) and r["c"] == pytest.approx(expect[0])


def test_ar1_fit_non_reverting_cases():
    rng = np.random.default_rng(15)
    x = np.empty(300)
    x[0] = 1.0
    for t in range(1, 300):
        x[t] = 1.02 * x[t - 1] + rng.normal(0, 0.1)
    explosive = st.ar1_fit(x, dt_s=1.0)
    assert explosive["phi"] > 1 and explosive["half_life_s"] == math.inf
    assert math.isnan(explosive["mu"]) and math.isnan(explosive["theta"])
    assert math.isnan(explosive["half_life_se_s"])
    flip = st.ar1_fit(_ar1(2000, -0.5, seed=16), dt_s=1.0)
    assert flip["phi"] < 0 and math.isnan(flip["half_life_s"])
    with pytest.raises(ValueError, match="pairs"):
        st.ar1_fit([1.0, NAN, 2.0, NAN, 3.0], dt_s=1.0)
    with pytest.raises(ValueError, match="dt_s"):
        st.ar1_fit(x, dt_s=0)


# =========================================================================== HAC mean / Holm

def test_hac_mean_test_matches_manual_newey_west():
    rng = np.random.default_rng(17)
    e = rng.normal(size=400)
    x = 1.0 + np.convolve(e, [1.0, 0.6, 0.3])[:400]
    r = st.hac_mean_test(x, mu0=1.0, lags=5)
    u = x - x.mean()
    n = x.size
    gam = [u[j:] @ u[: n - j] / n for j in range(6)]
    lrv = gam[0] + 2 * sum((1 - j / 6) * gam[j] for j in range(1, 6))
    assert r["se"] == pytest.approx(math.sqrt(lrv / n), rel=1e-10)
    assert r["t"] == pytest.approx((x.mean() - 1.0) / r["se"])
    assert r["pvalue"] == pytest.approx(2 * (1 - 0.5 * (1 + math.erf(abs(r["t"]) / math.sqrt(2)))))
    assert r["lags"] == 5 and r["nobs"] == 400


def test_hac_mean_test_accounts_for_persistence():
    phi = 0.95
    x = _ar1(5000, phi, seed=18, mu=1.0, sigma=0.002)
    r = st.hac_mean_test(np.concatenate([[NAN], x, [NAN]]))  # NaNs dropped
    assert r["nobs"] == 5000
    naive_se = x.std(ddof=1) / math.sqrt(x.size)
    assert 3.5 < r["se"] / naive_se < 9  # theory sqrt((1+phi)/(1-phi)) ~ 6.2
    assert r["pvalue"] > 0.05
    assert st.hac_mean_test(x + 0.01)["pvalue"] < 1e-6


def test_holm_known_example_and_properties():
    np.testing.assert_allclose(st.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(st.holm([0.5, 0.6]), [1.0, 1.0])  # capped
    np.testing.assert_allclose(st.holm([0.02, NAN, 0.01]), [0.02, NAN, 0.02])  # NaN excluded from m
    p = np.random.default_rng(19).uniform(0, 0.2, 25)
    np.testing.assert_allclose(st.holm(p), multipletests(p, method="holm")[1])
    assert st.holm([]).size == 0
    with pytest.raises(ValueError):
        st.holm([0.5, 1.5])


# =========================================================================== cointegration

def test_engle_granger_detects_cointegrated_pair_both_directions():
    x = _rw(1500, seed=20)
    y = 0.5 + 2.0 * x + _ar1(1500, 0.5, seed=21)
    r = st.engle_granger(pd.Series(y), pd.Series(x))
    assert r["nobs"] == 1500
    assert r["y_on_x"]["pvalue"] < 0.01 and r["x_on_y"]["pvalue"] < 0.01
    assert r["y_on_x"]["beta"] == pytest.approx(2.0, abs=0.02)
    assert r["y_on_x"]["stat"] < r["y_on_x"]["crit"]["1%"]
    indep = st.engle_granger(_rw(1500, seed=22), _rw(1500, seed=23))
    assert indep["y_on_x"]["pvalue"] > 0.05 and indep["x_on_y"]["pvalue"] > 0.05


def _softmax_basket(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logits = np.cumsum(rng.normal(0, 0.01, (n, 3)), axis=0) + np.log([0.5, 0.3, 0.2])
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return pd.DataFrame(p + rng.normal(0, 0.001, (n, 3)), columns=["a", "b", "c"])


def test_johansen_finds_rank_one_close_to_iota():
    r = st.johansen(_softmax_basket())
    assert r["rank_trace"] == 1 and r["rank_maxeig"] == 1
    assert 0 <= r["angle_to_iota_deg"] < 2.0
    assert r["evec0"].index.tolist() == ["a", "b", "c"]
    assert np.linalg.norm(r["evec0"]) == pytest.approx(1.0) and r["evec0"].sum() > 0
    assert r["k_ar_diff"] >= 1 and r["lr1"].shape == (3,) and r["cvt"].shape == (3, 3)
    assert st.johansen(_softmax_basket(), k_ar_diff=2)["k_ar_diff"] == 2


def test_johansen_rejects_exactly_collinear_legs():
    df = _softmax_basket(500)
    df["c"] = 1.0 - df["a"] - df["b"]
    with pytest.raises(ValueError, match="collinear"):
        st.johansen(df)
    with pytest.raises(ValueError, match="two series"):
        st.johansen(df[["a"]])


def test_aggregate_tail_legs():
    means = [0.30, 0.01, 0.25, 0.02, 0.15, 0.005, 0.2, 0.003]
    cols = [f"l{i}" for i in range(8)]
    df = pd.DataFrame(np.tile(means, (4, 1)), columns=cols)
    df.iloc[2, 1] = NAN
    out = st.aggregate_tail_legs(df, max_legs=4)
    assert list(out.columns) == ["l0", "l2", "l6", "rest"]
    np.testing.assert_allclose(out["rest"], [0.01 + 0.02 + 0.15 + 0.005 + 0.003] * 2 + [NAN, 0.188], equal_nan=True)
    small = st.aggregate_tail_legs(df[cols[:3]], max_legs=4)
    pd.testing.assert_frame_equal(small, df[cols[:3]])
    with pytest.raises(ValueError):
        st.aggregate_tail_legs(df, max_legs=1)


# =========================================================================== regressions

def test_identity_regression_beta_minus_one():
    rng = np.random.default_rng(24)
    n = 3000
    others = pd.Series(0.6 + 0.1 * np.sin(np.arange(n) / 300) + np.cumsum(rng.normal(0, 0.001, n)))
    m1 = 1.02 - others + _ar1(n, 0.8, seed=25, sigma=0.002)
    m1.iloc[100] = NAN
    r = st.identity_regression(m1, others)
    assert r["beta"] == pytest.approx(-1.0, abs=0.02)
    assert r["alpha"] == pytest.approx(1.02, abs=0.02)
    assert abs(r["t_beta_eq_minus1"]) < 3 and r["p_beta_eq_minus1"] > 0.003
    assert r["nobs"] == n - 1 and r["hac_lags"] >= 1
    assert len(r["resid"]) == n and math.isnan(r["resid"].iloc[100])
    two = st.identity_regression(m1, pd.DataFrame({"u": others * 0.5, "v": others * 0.5 + rng.normal(0, 0.01, n)}))
    assert isinstance(two["beta"], pd.Series) and list(two["beta"].index) == ["u", "v"]


def test_rolling_ols_spread_uses_only_past_parameters():
    rng = np.random.default_rng(26)
    n, w = 120, 20
    x = pd.Series(np.cumsum(rng.normal(size=n)))
    y = 1.0 - 0.8 * x + rng.normal(0, 0.1, n)
    base = st.rolling_ols_spread(y, x, w)
    assert base[["alpha", "beta"]].iloc[:w].isna().all().all()
    # parameters used at t come from rows [t - w, t - 1]
    t = 50
    coef = np.linalg.lstsq(np.column_stack([np.ones(w), x[t - w:t]]), y[t - w:t], rcond=None)[0]
    assert base["alpha"].iloc[t] == pytest.approx(coef[0]) and base["beta"].iloc[t] == pytest.approx(coef[1])
    assert base["spread"].iloc[t] == pytest.approx(y[t] - coef[0] - coef[1] * x[t])
    # perturbing y_t leaves every parameter used up to and including t unchanged
    y2 = y.copy()
    y2.iloc[t] += 5.0
    pert = st.rolling_ols_spread(y2, x, w)
    pd.testing.assert_frame_equal(pert[["alpha", "beta"]].iloc[: t + 1], base[["alpha", "beta"]].iloc[: t + 1])
    pd.testing.assert_frame_equal(pert.iloc[:t], base.iloc[:t])
    assert pert["spread"].iloc[t] - base["spread"].iloc[t] == pytest.approx(5.0)
    assert pert["beta"].iloc[t + 1] != pytest.approx(base["beta"].iloc[t + 1])


def test_rolling_ols_spread_gaps_and_validation():
    x = pd.Series(np.arange(40.0) + np.sin(np.arange(40)))
    y = 2.0 + 0.5 * x
    x.iloc[10] = NAN
    r = st.rolling_ols_spread(y, x, 5)
    assert r["beta"].iloc[11:16].isna().all()  # windows containing the gap (used one step later)
    assert r["beta"].iloc[17] == pytest.approx(0.5) and r["alpha"].iloc[17] == pytest.approx(2.0)
    # a window where x is constant has no hedge ratio (NaN, not a near-singular blow-up)
    xc = pd.Series(np.r_[np.linspace(0.3, 0.5, 30), np.full(10, 0.47)])
    flat = st.rolling_ols_spread(1.0 - xc, xc, 5)
    assert flat["beta"].iloc[35:].isna().all() and flat["beta"].iloc[30] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="window"):
        st.rolling_ols_spread(y, x, 2)
    with pytest.raises(ValueError, match="exceeds"):
        st.rolling_ols_spread(y, x, 41)


def test_identity_residual_stats():
    phi, dt = 0.8, 60.0
    r = st.identity_residual_stats(_ar1(5000, phi, seed=27, sigma=0.001), dt)
    assert abs(r["mean"]) < 3 * r["se"] and r["pvalue"] > 0.003
    assert r["hac_t"] == pytest.approx(r["mean"] / r["se"])
    assert r["half_life_s"] == pytest.approx(-math.log(2) / math.log(phi) * dt, rel=0.15)
    assert r["phi"] == pytest.approx(phi, abs=0.03)


# =========================================================================== model selection

def test_walk_forward_splits_sizes_and_embargo():
    assert st.walk_forward_splits(100) == (slice(0, 60), slice(60, 80), slice(80, 100))
    tr, va, te = st.walk_forward_splits(100, embargo=5)
    assert (tr, va, te) == (slice(0, 60), slice(65, 80), slice(85, 100))
    idx = pd.date_range("2026-01-01", periods=100, freq="1min", tz="UTC")
    assert st.walk_forward_splits(idx, embargo="5min") == (tr, va, te)
    assert st.walk_forward_splits(idx, embargo=pd.Timedelta(0)) == st.walk_forward_splits(100)
    assert st.walk_forward_splits(10, train=0.5, val=0.3) == (slice(0, 5), slice(5, 8), slice(8, 10))
    # time embargo on an irregular index drops exactly the rows inside the span
    irregular = pd.DatetimeIndex(T0 + pd.to_timedelta([0, 1, 2, 3, 4, 5, 30, 31, 32, 33], unit="min"))
    assert st.walk_forward_splits(irregular, train=0.5, val=0.3, embargo="1min") == (
        slice(0, 5), slice(6, 8), slice(9, 10))
    with pytest.raises(ValueError, match="empty split"):
        st.walk_forward_splits(100, embargo=20)
    with pytest.raises(ValueError):
        st.walk_forward_splits(100, train=0.8, val=0.2)
    with pytest.raises(TypeError):
        st.walk_forward_splits(100, embargo="5min")


def test_plateau_select_prefers_plateau_over_isolated_spike():
    rows = []
    for n_win in (10, 25, 50, 100, 250):
        for z in (1.5, 2.0, 2.5, 3.0, 3.5):
            rows.append({"N": n_win, "z_entry": z, "sharpe": 0.0})
    grid = pd.DataFrame(rows)
    plateau = grid["N"].isin([50, 100, 250]) & grid["z_entry"].isin([1.5, 2.0, 2.5])
    grid.loc[plateau, "sharpe"] = 1.0
    grid.loc[(grid["N"] == 100) & (grid["z_entry"] == 2.0), "sharpe"] = 1.2
    grid.loc[(grid["N"] == 10) & (grid["z_entry"] == 3.5), "sharpe"] = 5.0  # lucky spike
    res = st.plateau_select(grid.sample(frac=1.0, random_state=0), "sharpe", ["N", "z_entry"])
    assert res["params"] == {"N": 100, "z_entry": 2.0}
    assert res["plateau_score"] == pytest.approx(1.0) and res["metric"] == pytest.approx(1.2)
    assert res["n_neighbours"] == 9 and res["n_configs"] == 25
    assert res["best_single"] == {"params": {"N": 10, "z_entry": 3.5}, "metric": 5.0}
    assert isinstance(res["params"]["N"], int)


def test_plateau_select_one_dimensional_and_validation():
    grid = pd.DataFrame({"w": [1, 2, 3, 4, 5, 6, 7, 8], "m": [0.0, 10.0, 0.0, 6.0, 7.0, 6.0, 0.0, NAN]})
    res = st.plateau_select(grid, "m", ["w"])
    assert res["params"] == {"w": 5} and res["plateau_score"] == 6.0
    assert res["best_single"]["params"] == {"w": 2}
    with pytest.raises(ValueError, match="duplicate"):
        st.plateau_select(pd.concat([grid, grid.iloc[:1]]), "m", ["w"])
    with pytest.raises(ValueError, match="lacks"):
        st.plateau_select(grid, "sharpe", ["w"])
    with pytest.raises(ValueError, match="finite"):
        st.plateau_select(grid.assign(m=NAN), "m", ["w"])
