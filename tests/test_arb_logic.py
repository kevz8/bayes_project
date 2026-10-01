"""Phase 3 engine: rolling statistics, basket sums, z-scores, signals and basket economics.

Hand-computed values are derived in the comments; payoff identities use exact rationals.
"""
from __future__ import annotations

import logging
import math
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from src import REPO_ROOT
from src.arb_engine import (
    Action,
    BasketSnapshot,
    BasketState,
    RollingStats,
    Signal,
    SignalConfig,
    SignalStateMachine,
    ZConfig,
    ZScoreEngine,
    ZState,
    basket_cost_hurdle,
    edge_to_cost_ratio,
    executable_edges,
    positions_from_z,
    push_mask,
    rolling_mean_std,
    zscore_batch,
)
from src.config import Basket, FeeSchedule, Leg, basket_fee_per_unit
from src.events import DictBookView, Side

log = logging.getLogger(__name__)
NAN = math.nan
S = 1_000_000_000  # ns per second

# The worked n=3 example used throughout (bids/asks of the YES legs).
BIDS3 = (0.55, 0.33, 0.15)
ASKS3 = (0.56, 0.34, 0.16)


# --------------------------------------------------------------------------- helpers
def stream_rolling(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    rs = RollingStats(window)
    mean, std = np.empty(x.size), np.empty(x.size)
    for i, v in enumerate(x.tolist()):
        rs.push(v)
        mean[i], std[i] = rs.mean(), rs.std()
    return mean, std


def run_engine(s: np.ndarray, valid: np.ndarray, cfg: ZConfig, reset: np.ndarray | None = None,
               t_ns: np.ndarray | None = None) -> dict[str, np.ndarray]:
    eng = ZScoreEngine(cfg)
    t = np.arange(s.size) * S if t_ns is None else t_ns
    out = {k: np.empty(s.size) for k in ("z", "mu", "sigma")}
    pushed = np.zeros(s.size, dtype=bool)
    for i in range(s.size):
        if reset is not None and reset[i]:
            eng.reset("test")
        st = eng.update(int(t[i]), float(s[i]), bool(valid[i]))
        out["z"][i], out["mu"][i], out["sigma"][i] = st.z, st.mu, st.sigma
        pushed[i] = st.pushed
    out["pushed"] = pushed
    return out


def seeded_series(seed: int = 7, n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S ≈ 1 + AR(1) dislocation, quantised to half a tick (so unchanged S repeats), with
    invalid stretches, a NaN run and a valid S equal to the last valid one after a gap."""
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    for i in range(1, n):
        d[i] = 0.97 * d[i - 1] + 0.004 * rng.standard_normal()
    s = np.round((1.0 + d) / 0.0005) * 0.0005
    valid = np.ones(n, dtype=bool)
    valid[700:760] = False
    valid[2000:2001] = False
    valid[3500:3900] = False
    s[4100:4120] = NAN
    s[3900] = s[3499]  # back from a gap at exactly the last valid value -> not pushed
    valid[rng.random(n) < 0.02] = False
    t_ns = np.cumsum(rng.integers(1, 3 * S, size=n))
    return t_ns, s, valid


def zs(z: float, s: float = 1.0, mu: float = 1.0, sigma: float = 0.01) -> ZState:
    return ZState(0, s, mu, sigma, z, True, 100)


def sig_key(sig: Signal) -> tuple:
    def f(v: float) -> float | None:
        return None if v != v else v
    return (sig.t_ns, sig.action, sig.reason, sig.side_before, f(sig.z), f(sig.s), f(sig.mu),
            f(sig.sigma), f(sig.mu_entry))


def assert_same(a: np.ndarray, b: np.ndarray, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


# --------------------------------------------------------------------------- RollingStats
@pytest.mark.parametrize("window", [2, 3, 50, 500])
def test_rolling_stats_matches_pandas(window: int) -> None:
    x = np.random.default_rng(window).standard_normal(3000)
    mean, std = stream_rolling(x, window)
    pm = pd.Series(x).rolling(window, min_periods=1).mean().to_numpy()
    ps = pd.Series(x).rolling(window, min_periods=1).std().to_numpy()  # ddof=1, NaN at count 1
    np.testing.assert_allclose(mean, pm, rtol=1e-10, atol=1e-12)
    assert_same(std, ps, rtol=1e-10, atol=1e-12)


def test_rolling_stats_hand_example_and_warmup() -> None:
    rs = RollingStats(3)
    assert rs.count == 0 and math.isnan(rs.mean()) and math.isnan(rs.var())
    rs.push(1.0)
    assert rs.mean() == 1.0 and math.isnan(rs.var())  # ddof=1 needs two values
    for v in (2.0, 3.0, 4.0):
        rs.push(v)
    # window 3 over [1, 2, 3, 4] holds [2, 3, 4]: mean 3, var ((-1)² + 0 + 1²) / 2 = 1
    assert rs.full and rs.count == 3
    assert rs.mean() == pytest.approx(3.0, abs=1e-15)
    assert rs.var() == pytest.approx(1.0, abs=1e-15)
    assert rs.var(ddof=0) == pytest.approx(2.0 / 3.0, abs=1e-15)
    np.testing.assert_array_equal(rs.values(), [2.0, 3.0, 4.0])
    rs.reset()
    assert rs.count == 0 and rs.values().size == 0
    with pytest.raises(ValueError):
        RollingStats(1)


def test_rolling_stats_recompute_every_bounds_state() -> None:
    x = 1.0 + 1e-3 * np.random.default_rng(1).standard_normal(1234)
    for every in (1, 7, None):
        rs = RollingStats(50, recompute_every=every)
        for v in x:
            rs.push(v)
        w = x[-50:]
        assert rs.mean() == pytest.approx(math.fsum(w) / 50, abs=1e-15)
        assert rs.var() == pytest.approx(np.var(w, ddof=1), rel=1e-12)
        np.testing.assert_array_equal(rs.values(), w)


def _adversarial(n_push: int, window: int = 500, n_checks: int = 40) -> None:
    rng = np.random.default_rng(11)
    x = 1.0 + 1e-9 * rng.standard_normal(n_push)
    checks = set(np.linspace(window, n_push - 1, n_checks).astype(int).tolist())
    rs = RollingStats(window)
    for i, v in enumerate(x.tolist()):
        rs.push(v)
        if i in checks:
            ref = np.var(x[i - window + 1:i + 1], ddof=1)
            assert rs.var() >= 0.0
            assert rs.var() == pytest.approx(ref, rel=1e-9)


def test_rolling_stats_adversarial_tiny_variance() -> None:
    """S ≈ 1 with σ ≈ 1e-9: an uncentred running Σx² would return garbage or a negative var."""
    _adversarial(1_000_000)


@pytest.mark.slow
def test_rolling_stats_adversarial_tiny_variance_1e7() -> None:
    _adversarial(10_000_000)


def test_rolling_stats_large_offset() -> None:
    x = 1e6 + np.random.default_rng(5).standard_normal(20_000)
    rs = RollingStats(500)
    for i, v in enumerate(x.tolist()):
        rs.push(v)
        if i >= 499 and i % 1000 == 0:
            w = x[i - 499:i + 1]
            assert rs.mean() == pytest.approx(math.fsum(w) / 500, rel=1e-14)
            assert rs.var() == pytest.approx(np.var(w, ddof=1), rel=1e-9)


# --------------------------------------------------------------------------- batch rolling
@pytest.mark.parametrize("window", [2, 3, 20, 33, 50, 500])
def test_rolling_mean_std_matches_streaming_and_pandas(window: int) -> None:
    x = 1.0 + 0.01 * np.random.default_rng(window).standard_normal(4000)
    bm, bs = rolling_mean_std(x, window, min_periods=1)
    sm, ss = stream_rolling(x, window)
    assert_same(bm, sm, rtol=1e-12)
    assert_same(bs, ss, rtol=1e-9)
    pm = pd.Series(x).rolling(window, min_periods=window // 2 + 1).mean().to_numpy()
    ps = pd.Series(x).rolling(window, min_periods=window // 2 + 1).std().to_numpy()
    bm2, bs2 = rolling_mean_std(x, window, min_periods=window // 2 + 1)
    assert_same(bm2, pm, rtol=1e-12)
    assert_same(bs2, ps, rtol=1e-8)


def test_rolling_mean_std_level_shifts() -> None:
    """Level shifts far larger than the noise force the O(T) path to re-centre; it must
    still match an exact per-window two-pass and the streaming RollingStats."""
    rng = np.random.default_rng(17)
    x = 1.0 + 0.0005 * np.round(rng.standard_normal(20_000))
    x[7000:] += 0.5
    x[12000:] -= 1.2
    for w in (40, 500):
        m, s = rolling_mean_std(x, w)
        win = sliding_window_view(x, w)
        np.testing.assert_allclose(m[w - 1:], win.mean(axis=1), rtol=1e-13)
        np.testing.assert_allclose(s[w - 1:], win.std(axis=1, ddof=1), rtol=1e-10)
        np.testing.assert_allclose(stream_rolling(x, w)[1][w - 1:], s[w - 1:], rtol=1e-10)


def test_rolling_mean_std_hand_example_and_errors() -> None:
    m, s = rolling_mean_std(np.array([1.0, 2.0, 3.0, 4.0]), 3)
    np.testing.assert_allclose(m, [NAN, NAN, 2.0, 3.0], equal_nan=True)
    np.testing.assert_allclose(s, [NAN, NAN, 1.0, 1.0], equal_nan=True)
    m, s = rolling_mean_std(np.array([1.0, 2.0, 3.0, 4.0]), 3, min_periods=1, ddof=0)
    np.testing.assert_allclose(m, [1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(s, [0.0, 0.5, math.sqrt(2 / 3), math.sqrt(2 / 3)])
    assert rolling_mean_std(np.array([]), 5)[0].size == 0
    with pytest.raises(ValueError):
        rolling_mean_std(np.array([1.0, NAN]), 2)


# --------------------------------------------------------------------------- z-score engine
def test_z_exclude_current_hand_value() -> None:
    eng = ZScoreEngine(ZConfig(window=3))
    for t, v in enumerate((1.0, 2.0, 3.0)):
        assert math.isnan(eng.update(t, v, True).z)  # warm-up: fewer than min_periods = 3
    st = eng.update(3, 5.0, True)
    # reference window [1, 2, 3]: μ = 2, σ = 1 -> z = (5 - 2) / 1 = 3
    assert (st.mu, st.sigma, st.z) == pytest.approx((2.0, 1.0, 3.0))
    assert st.pushed and st.n_obs == 3
    np.testing.assert_array_equal(eng.stats.values(), [2.0, 3.0, 5.0])


def test_z_include_current_differs() -> None:
    eng = ZScoreEngine(ZConfig(window=3, include_current=True))
    for t, v in enumerate((1.0, 2.0, 3.0)):
        eng.update(t, v, True)
    st = eng.update(3, 5.0, True)
    # window [2, 3, 5]: μ = 10/3, σ² = (16/9 + 1/9 + 25/9) / 2 = 7/3 -> z = (5/3) / √(7/3)
    assert st.z == pytest.approx((5 / 3) / math.sqrt(7 / 3), rel=1e-12)
    assert st.z < 3.0  # the jump inflates its own σ (self-dampening)


def test_z_sigma_floor_hand_value_event_mode() -> None:
    # change_eps < 0 pushes every valid value, so the window can hold [1, 1, 1] (σ = 0)
    eng = ZScoreEngine(ZConfig(window=3, sigma_floor=0.005, change_eps=-1.0))
    for t in range(3):
        eng.update(t, 1.0, True)
    st = eng.update(3, 1.004, True)
    assert st.sigma == 0.0
    assert st.z == pytest.approx(0.004 / 0.005)  # = 0.8


def test_z_zero_sigma_without_floor_is_nan() -> None:
    eng = ZScoreEngine(ZConfig(window=3, sigma_floor=0.0, change_eps=-1.0))
    for t in range(3):
        eng.update(t, 1.0, True)
    st = eng.update(3, 1.004, True)
    assert st.sigma == 0.0 and math.isnan(st.z)


def test_z_warmup_min_periods() -> None:
    eng = ZScoreEngine(ZConfig(window=10, min_periods=3))
    zs_ = [eng.update(t, v, True) for t, v in enumerate((1.0, 1.01, 1.02))]
    assert all(math.isnan(x.z) and math.isnan(x.mu) for x in zs_) and eng.warm
    zs_.append(eng.update(3, 1.05, True))
    # reference [1.00, 1.01, 1.02]: μ = 1.01, σ = 0.01 -> z = 0.04 / 0.01 = 4
    assert zs_[3].z == pytest.approx(4.0, rel=1e-9)


def test_z_invalid_pushes_nothing() -> None:
    eng = ZScoreEngine(ZConfig(window=3))
    for t, v in enumerate((1.0, 2.0, 3.0)):
        eng.update(t, v, True)
    for st in (eng.update(3, 9.0, False), eng.update(4, NAN, True)):
        assert math.isnan(st.z) and math.isnan(st.mu) and not st.pushed and st.n_obs == 3
    assert eng.update(5, 5.0, True).z == pytest.approx(3.0)


def test_z_unchanged_returns_cached_state() -> None:
    eng = ZScoreEngine(ZConfig(window=3))
    for t, v in enumerate((1.0, 2.0, 3.0)):
        eng.update(t, v, True)
    first = eng.update(3, 5.0, True)
    again = eng.update(4, 5.0, True)
    assert not again.pushed and again.n_obs == 3
    assert (again.z, again.mu, again.sigma) == (first.z, first.mu, first.sigma)
    eng.update(5, 1.0, False)  # an invalid stretch does not clear the cache...
    back = eng.update(6, 5.0 + 1e-13, True)  # ...and a within-eps S is still "unchanged"
    assert not back.pushed and back.z == first.z
    np.testing.assert_array_equal(eng.stats.values(), [2.0, 3.0, 5.0])


def test_z_reset_clears_window() -> None:
    eng = ZScoreEngine(ZConfig(window=3))
    for t, v in enumerate((1.0, 2.0, 3.0)):
        eng.update(t, v, True)
    eng.reset("gap")
    assert eng.n_resets == 1 and eng.stats.count == 0 and not eng.warm
    st = eng.update(10, 3.0, True)  # equal to the pre-reset S but pushed: history is gone
    assert st.pushed and math.isnan(st.z) and st.n_obs == 1


def test_z_clock_mode_locf_fill() -> None:
    eng = ZScoreEngine(ZConfig(window=3, sample_mode="clock", clock_interval_s=1.0))
    st = eng.update(S // 2, 1.0, True)  # first grid boundary at 1 s
    assert not st.pushed and st.n_obs == 0
    st = eng.update(3 * S + S // 2, 1.004, True)
    # boundaries 1 s, 2 s, 3 s < 3.5 s each receive the held 1.0 -> window [1, 1, 1]
    assert st.pushed and st.n_obs == 3
    assert st.sigma == 0.0 and st.z == pytest.approx(0.8)
    # an update exactly on a boundary does not see that boundary's sample yet
    st = eng.update(4 * S, 1.010, True)
    assert st.n_obs == 3 and st.z == pytest.approx((1.010 - 1.0) / 0.005)
    st = eng.update(4 * S + 1, 1.010, True)  # now 4 s holds LOCF(4 s) = 1.010
    np.testing.assert_allclose(eng.stats.values(), [1.0, 1.0, 1.010])
    assert st.mu == pytest.approx((2.0 + 1.010) / 3)


def test_z_clock_mode_invalid_and_fill_cap() -> None:
    eng = ZScoreEngine(ZConfig(window=10, min_periods=2, sample_mode="clock", max_clock_fill=2))
    eng.update(0, 1.0, True)
    eng.update(S // 2, 1.0, False)            # invalid from 0.5 s
    st = eng.update(5 * S + 1, 1.02, True)    # boundaries 1..5 s crossed while invalid
    assert not st.pushed and eng.stats.count == 1  # only boundary 0 s (held 1.0) was pushed
    st = eng.update(10 * S + 1, 1.03, True)   # 5 boundaries, capped at 2 pushes of 1.02
    assert eng.stats.count == 3 and eng.n_fill_capped == 1
    np.testing.assert_allclose(eng.stats.values(), [1.0, 1.02, 1.02])


def test_zconfig_validation() -> None:
    with pytest.raises(ValueError):
        ZConfig(window=1)
    with pytest.raises(ValueError):
        ZConfig(window=10, min_periods=11)
    with pytest.raises(ValueError):
        ZConfig(sample_mode="tick")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ZConfig(sample_mode="clock", include_current=True)
    with pytest.raises(ValueError):
        zscore_batch(np.ones(3), np.ones(3, bool), ZConfig(sample_mode="clock"))


# --------------------------------------------------------------------------- streaming = batch
def test_push_mask_hand_example() -> None:
    s = np.array([1.0, 1.0, 2.0, NAN, 2.0, 3.0, 3.0])
    valid = np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool)
    np.testing.assert_array_equal(push_mask(s, valid), [1, 0, 1, 0, 0, 0, 1])
    reset = np.zeros(7, dtype=bool)
    reset[4] = True  # first valid after a reset is always pushed
    np.testing.assert_array_equal(push_mask(s, valid, reset=reset), [1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(push_mask(s, valid, eps=-1.0), [1, 1, 1, 0, 1, 0, 1])


@pytest.mark.parametrize(
    "cfg",
    [
        ZConfig(window=100),
        ZConfig(window=100, min_periods=20, sigma_floor=0.002),
        ZConfig(window=20, include_current=True),
        ZConfig(window=40, min_periods=1, sigma_floor=0.0),
        ZConfig(window=250, change_eps=-1.0),
    ],
    ids=["default", "min_periods", "include_current", "no_floor", "push_all"],
)
def test_streaming_equals_batch(cfg: ZConfig) -> None:
    t_ns, s, valid = seeded_series()
    stream = run_engine(s, valid, cfg, t_ns=t_ns)
    batch = zscore_batch(s, valid, cfg)
    np.testing.assert_array_equal(stream["pushed"], batch["pushed"])
    for k in ("z", "mu", "sigma"):
        assert_same(stream[k], batch[k])
    assert np.isfinite(batch["z"]).sum() > 3000  # the comparison is not vacuous
    if cfg.change_eps >= 0:
        assert (valid & ~batch["pushed"] & np.isfinite(s)).sum() > 100  # unchanged S exercised


def test_streaming_equals_batch_with_resets() -> None:
    t_ns, s, valid = seeded_series(seed=3)
    reset = np.zeros(s.size, dtype=bool)
    reset[[1500, 1501, 4500]] = True
    cfg = ZConfig(window=60, min_periods=30)
    stream = run_engine(s, valid, cfg, reset=reset, t_ns=t_ns)
    batch = zscore_batch(s, valid, cfg, reset=reset)
    np.testing.assert_array_equal(stream["pushed"], batch["pushed"])
    for k in ("z", "mu", "sigma"):
        assert_same(stream[k], batch[k])
    assert np.isnan(batch["z"][1501:1530]).all()  # warm-up again after the reset


def test_prefix_invariance_no_lookahead() -> None:
    """Changing data after index t must not change any z, μ, σ or signal up to t."""
    t_ns, s, valid = seeded_series(seed=21)
    cfg = ZConfig(window=80)
    scfg = SignalConfig(z_entry=1.5, z_exit=0.3, max_hold_s=600.0)
    rng = np.random.default_rng(99)
    for cut in (1000, 2500, 5003):
        s2, valid2 = s.copy(), valid.copy()
        s2[cut + 1:] += 0.01 * rng.standard_normal(s.size - cut - 1)
        valid2[cut + 1:] = rng.random(s.size - cut - 1) < 0.8
        for fn in (lambda a, v: zscore_batch(a, v, cfg), lambda a, v: run_engine(a, v, cfg, t_ns=t_ns)):
            a, b = fn(s, valid), fn(s2, valid2)
            for k in ("z", "mu", "sigma", "pushed"):
                np.testing.assert_array_equal(a[k][:cut + 1], b[k][:cut + 1])  # bit-identical
            pa, sa = positions_from_z(t_ns, s, a["mu"], a["sigma"], a["z"], scfg)
            pb, sb = positions_from_z(t_ns, s2, b["mu"], b["sigma"], b["z"], scfg)
            np.testing.assert_array_equal(pa[:cut + 1], pb[:cut + 1])
            upto = t_ns[cut]
            assert [sig_key(x) for x in sa if x.t_ns <= upto] == [sig_key(x) for x in sb if x.t_ns <= upto]
            assert len(sa) > 4


def test_positions_from_z_equals_streaming_state_machine() -> None:
    t_ns, s, valid = seeded_series(seed=5)
    cfg = ZConfig(window=100)
    scfg = SignalConfig(z_entry=1.5, z_exit=0.0, z_stop=4.0, cooldown_s=30.0, exit_ref="frozen_entry_mu")
    eng, sm = ZScoreEngine(cfg), SignalStateMachine(scfg, cfg.sigma_floor)
    live_sigs, live_side = [], []
    for i in range(s.size):
        st = eng.update(int(t_ns[i]), float(s[i]), bool(valid[i]))
        sig = sm.step(int(t_ns[i]), st)
        if sig is not None:
            live_sigs.append(sig)
        live_side.append(int(sm.state))
    b = zscore_batch(s, valid, cfg)
    side, sigs = positions_from_z(t_ns, s, b["mu"], b["sigma"], b["z"], scfg, cfg.sigma_floor)
    np.testing.assert_array_equal(side, live_side)
    assert [(x.t_ns, x.action, x.reason) for x in sigs] == [(x.t_ns, x.action, x.reason) for x in live_sigs]
    assert {x.action for x in sigs} == {Action.ENTER_LONG, Action.ENTER_SHORT, Action.EXIT}
    assert side.dtype == np.int8


# --------------------------------------------------------------------------- state machine
E = (Action.ENTER_SHORT, "z_entry")
L = (Action.ENTER_LONG, "z_entry")


def X(reason: str) -> tuple[Action, str]:
    return (Action.EXIT, reason)


@pytest.mark.parametrize(
    "cfg, steps, expected",
    [
        # (t seconds, z) per step; expected (action, reason) or None per step
        (SignalConfig(), [(0, 0.5), (1, 2.5), (2, 1.0), (3, 0.1)], [None, E, None, X("revert")]),
        (SignalConfig(), [(0, -2.5), (1, -1.0), (2, -0.1)], [L, None, X("revert")]),
        (SignalConfig(), [(0, 2.0), (1, -2.0)], [None, None]),  # entry needs |z| > z_entry
        (SignalConfig(z_stop=3.0), [(0, 2.5), (1, 3.5)], [E, X("stop")]),
        (SignalConfig(z_stop=3.0), [(0, -2.5), (1, -3.5)], [L, X("stop")]),
        (SignalConfig(), [(0, 2.5), (1, -2.5), (2, -2.5)], [E, X("flip"), L]),
        (SignalConfig(), [(0, -2.5), (1, 2.5), (2, 2.5)], [L, X("flip"), E]),
        (SignalConfig(), [(0, 2.5), (1, -1.0)], [E, X("revert")]),  # overshoot past μ = reverted
        (SignalConfig(max_hold_s=10.0), [(0, 2.5), (5, 1.5), (10, 1.5), (11, 1.5)],
         [E, None, None, X("timeout")]),
        (SignalConfig(max_hold_s=10.0), [(0, -2.5), (11, NAN)], [L, X("timeout")]),  # NaN z still times out
        (SignalConfig(), [(0, NAN), (1, 2.5), (2, NAN), (3, NAN)], [None, E, None, None]),
        (SignalConfig(cooldown_s=5.0), [(0, 2.5), (1, 0.0), (2, 2.5), (5.9, 3.0), (6, 2.5)],
         [E, X("revert"), None, None, E]),
        (SignalConfig(allow_short=False), [(0, 2.5), (1, -2.5)], [None, L]),
        (SignalConfig(allow_long=False), [(0, -2.5), (1, 2.5)], [None, E]),
        # zero-crossing exit: z_exit == 0 exits when z touches/crosses 0 against the entry sign
        (SignalConfig(z_exit=0.0), [(0, 2.5), (1, 0.1), (2, 0.0)], [E, None, X("revert")]),
        (SignalConfig(z_exit=0.0), [(0, -2.5), (1, -0.1), (2, 0.3)], [L, None, X("revert")]),
        (SignalConfig(z_exit=0.5), [(0, 2.5), (1, 0.6), (2, 0.4)], [E, None, X("revert")]),
    ],
    ids=["short_revert", "long_revert", "strict_entry", "stop_short", "stop_long", "flip_short",
         "flip_long", "overshoot", "timeout", "timeout_nan", "nan_no_action", "cooldown",
         "no_short", "no_long", "zero_cross_short", "zero_cross_long", "wider_exit"],
)
def test_state_machine_transitions(cfg: SignalConfig, steps: list, expected: list) -> None:
    sm = SignalStateMachine(cfg)
    got = []
    for t, z in steps:
        sig = sm.step(int(t * S), zs(z))
        got.append(None if sig is None else (sig.action, sig.reason))
    assert got == expected


def test_state_machine_gate_and_entry_failure() -> None:
    calls: list[Side] = []

    def reject(side: Side) -> bool:
        calls.append(side)
        return False

    sm = SignalStateMachine(SignalConfig())
    assert sm.step(0, zs(0.5), reject) is None and calls == []  # gate only asked on a z signal
    assert sm.step(1, zs(2.5), reject) is None and calls == [Side.SHORT_BASKET]
    assert sm.step(2, zs(-2.5), reject) is None and calls[-1] == Side.LONG_BASKET
    sig = sm.step(3, zs(2.5), lambda side: side is Side.SHORT_BASKET)
    assert sig is not None and sig.action is Action.ENTER_SHORT and sm.state is Side.SHORT_BASKET
    sm.on_entry_failed()  # zero fill: flat again, no exit record, no cooldown
    assert sm.state is Side.FLAT and sm.entry_t_ns is None and sm.last_exit_t_ns is None
    assert sm.step(4, zs(2.5)) is not None


def test_state_machine_records_mu_and_force_flat() -> None:
    sm = SignalStateMachine(SignalConfig())
    ent = sm.step(0, ZState(0, 1.05, 1.00, 0.02, 2.5, True, 50))
    assert ent.mu_entry == 1.00 and ent.side_after is Side.SHORT_BASKET and sm.entry_z == 2.5
    ext = sm.step(S, ZState(S, 1.05, 1.048, 0.02, 0.1, True, 50))
    # 'revert' fired although S never moved: μ drifted up to S (baseline drift = 0.048)
    assert ext.reason == "revert" and ext.side_before is Side.SHORT_BASKET
    assert ext.baseline_drift == pytest.approx(0.048) and ext.side_after is Side.FLAT
    assert sm.force_flat(2 * S, "end_of_data") is None
    sm.step(3 * S, zs(-2.5))
    out = sm.force_flat(4 * S, "resolution")
    assert out.reason == "resolution" and out.side_before is Side.LONG_BASKET and math.isnan(out.z)
    assert sm.state is Side.FLAT and sm.last_exit_t_ns == 4 * S


def test_frozen_entry_mu_exit() -> None:
    """With a rolling μ that drifts toward S the rolling z 'reverts' without S moving; the
    frozen reference only exits once S itself returns to μ_entry."""
    path = [  # (s, rolling μ, σ, rolling z); frozen z = (s - 1.00) / 0.02
        (1.05, 1.000, 0.02, 2.5),    # enter short
        (1.05, 1.048, 0.02, 0.1),    # rolling z 0.1: μ caught up, S did not revert; frozen 2.5
        (1.03, 1.040, 0.02, -0.5),   # frozen 1.5
        (1.006, 1.03, 0.02, -1.2),   # frozen 0.3 -> still above z_exit
        (1.002, 1.03, 0.02, -1.4),   # frozen 0.1 -> revert
    ]
    roll = SignalStateMachine(SignalConfig(z_exit=0.2))
    frozen = SignalStateMachine(SignalConfig(z_exit=0.2, exit_ref="frozen_entry_mu"))
    got_roll, got_frozen = [], []
    for i, (s, mu, sd, z) in enumerate(path):
        st = ZState(i * S, s, mu, sd, z, True, 100)
        got_roll.append(roll.step(i * S, st))
        got_frozen.append(frozen.step(i * S, st))
    assert got_roll[1] is not None and got_roll[1].reason == "revert"
    assert [g is not None for g in got_frozen] == [True, False, False, False, True]
    assert got_frozen[-1].reason == "revert" and got_frozen[-1].mu_entry == 1.000


def test_frozen_entry_mu_zero_crossing_uses_sigma_floor() -> None:
    sm = SignalStateMachine(SignalConfig(z_exit=0.0, exit_ref="frozen_entry_mu"), sigma_floor=0.005)
    sm.step(0, ZState(0, 0.97, 1.0, 0.001, -6.0, True, 10))   # long (σ below the floor)
    assert sm.step(1, ZState(1, 0.999, 0.99, 0.001, 1.8, True, 10)) is None  # S < μ_entry
    assert sm.step(2, ZState(2, 1.0, 0.99, 0.001, 2.0, True, 10)).reason == "revert"  # touches


def test_signal_config_validation() -> None:
    for bad in (dict(z_exit=2.0), dict(z_exit=-0.1), dict(z_stop=2.0), dict(max_hold_s=0.0),
                dict(cooldown_s=-1.0), dict(exit_ref="mid")):
        with pytest.raises(ValueError):
            SignalConfig(**bad)  # type: ignore[arg-type]
    SignalConfig(z_exit=0.0, z_stop=2.5)


# --------------------------------------------------------------------------- BasketState
def strict_sums(bid: np.ndarray, ask: np.ndarray) -> tuple[float, float]:
    return (math.fsum(np.where(np.isfinite(bid), bid, 0.0).tolist()),
            math.fsum(np.where(np.isfinite(ask), ask, 1.0).tolist()))


@pytest.mark.parametrize("exact_every", [256, 10**9])
def test_basket_state_incremental_matches_fsum(exact_every: int) -> None:
    n = 12
    rng = np.random.default_rng(exact_every % 97)
    st = BasketState([f"L{i}" for i in range(n)], exact_every=exact_every)
    for i in range(n):
        st.set_leg(i, 0.05, 0.06, 10, 10, 0, True)
    for step in range(20_000):
        i = int(rng.integers(n))
        b = round(float(rng.uniform(0.01, 0.9)), 3)
        a = round(b + 0.001 * int(rng.integers(1, 30)), 3)
        if rng.random() < 0.03:
            b = NAN
        if rng.random() < 0.01:
            a = NAN
        st.set_leg(i, b, a, 1.0, 1.0, step, True)
        sb, sa = strict_sums(st.bid, st.ask)
        assert abs(st.s_bid - sb) < 1e-12 and abs(st.s_ask - sa) < 1e-12
        if st.valid:
            assert abs(st.s_mid - math.fsum(st.mid.tolist())) < 1e-12
    st.recompute()
    assert st.s_bid == strict_sums(st.bid, st.ask)[0]


def test_basket_state_one_sided_leg_and_dust_imputation() -> None:
    st = BasketState(["a", "b", "c"], dust_max_ask=0.02)
    snap = st.snapshot(0)
    assert not snap.valid and snap.max_age_s == math.inf and snap.s_ask == 3.0 and snap.s_bid == 0.0
    st.set_leg(0, 0.60, 0.62, 100, 50, 1 * S, True)
    st.set_leg(1, 0.35, 0.37, 80, 40, 2 * S, True)
    st.set_leg(2, NAN, 0.30, NAN, 10, 3 * S, True)  # one-sided, not dust
    snap = st.snapshot(4 * S)
    assert not snap.valid and math.isnan(snap.s_mid)
    assert snap.s_bid == pytest.approx(0.95) and snap.s_ask == pytest.approx(1.29)  # bid 0 counts 0
    st.set_leg(2, 0.29, NAN, 10, NAN, 3 * S, True)  # no ask counts 1
    assert st.snapshot(4 * S).s_ask == pytest.approx(1.99)
    st.set_leg(2, NAN, 0.01, NAN, 500, 3 * S, True)  # dust: no bid and ask <= 0.02
    snap = st.snapshot(4 * S)
    assert snap.valid and snap.n_imputed == 1 and bool(st.imputed[2])
    assert st.mid[2] == 0.005 and snap.s_mid == pytest.approx(0.61 + 0.36 + 0.005)
    assert snap.s_bid == pytest.approx(0.95) and snap.s_ask == pytest.approx(1.00)
    assert snap.spread_sum == pytest.approx(0.05) and snap.max_age_s == pytest.approx(3.0)
    st.set_leg(2, 0.004, 0.01, 5, 500, 5 * S, True)  # bid appears: no longer imputed
    assert st.snapshot(5 * S).n_imputed == 0 and not st.imputed[2]
    st.set_leg(2, NAN, 0.01, NAN, 500, 6 * S, False)  # unclean book -> S_mid invalid
    snap = st.snapshot(6 * S)
    assert not snap.valid and not snap.clean and snap.n_imputed == 0


def test_basket_state_tradable_flag() -> None:
    st = BasketState(["a", "b"])
    st.set_leg(0, 0.4, 0.42, 1, 1, 0, True)
    st.set_leg(1, 0.57, 0.6, 1, 1, 0, True, tradable=False)
    snap = st.snapshot(0)
    assert snap.valid and not snap.tradable and snap.s_mid == pytest.approx(0.995)
    st.set_tradable(1, True)
    assert st.snapshot(0).tradable


def test_basket_state_update_from_view_applies_frame_once() -> None:
    view = DictBookView({
        "y1": ([(0.50, 100), (0.49, 50)], [(0.52, 30)]),
        "y2": ([(0.30, 10)], [(0.33, 20), (0.34, 5)]),
        "y3": ([(0.15, 7)], [(0.17, 9)]),
        "other": ([(0.9, 1)], [(0.95, 1)]),
    })
    st = BasketState(["y1", "y2", "y3"])
    assert st.update_from_view(view, ["y1", "y2", "y3", "other", "NO-token"], 10 * S)
    snap = st.snapshot(10 * S)
    assert snap.valid and snap.clean and snap.n_legs == 3
    assert snap.s_bid == pytest.approx(0.95) and snap.s_ask == pytest.approx(1.02)
    assert snap.s_mid == pytest.approx(0.985) and snap.band_side is Side.FLAT
    assert st.bid_sz[0] == 100 and st.ask_sz[1] == 20 and st.t_last[2] == 10 * S
    assert not st.update_from_view(view, ["y2"], 11 * S)  # nothing reported changed
    assert not st.update_from_view(view, ["other"], 11 * S)
    view.set_book("y2", [(0.36, 10)], [(0.37, 20)])
    view.set_book("y3", [], [(0.17, 9)])  # bid side vanishes -> S_mid invalid
    assert st.update_from_view(view, {"y2", "y3"}, 12 * S)
    snap = st.snapshot(12 * S)
    assert not snap.valid and snap.s_bid == pytest.approx(0.86) and math.isnan(st.bid_sz[2])
    view.set_book("y3", [(0.16, 1)], [(0.17, 9)])
    view.set_dirty("y3")
    st.update_from_view(view, ["y3"], 13 * S)
    assert not st.snapshot(13 * S).clean and not st.valid
    view.set_dirty("y3", False)
    st.update_from_view(view, ["y3"], 14 * S)
    assert st.valid and st.snapshot(14 * S).band_side is Side.SHORT_BASKET  # S_bid = 1.02 > 1


def test_basket_state_remove_leg() -> None:
    st = BasketState(["a", "b", "c"], ticks=[0.01, 0.001, 0.01])
    for i, (b, a) in enumerate(zip(BIDS3, ASKS3, strict=True)):
        st.set_leg(i, b, a, 10, 10, i, True)
    st.set_leg(1, NAN, NAN, NAN, NAN, 5, True, tradable=False)
    st.remove_leg("b")  # leg b resolved NO early
    assert st.n == 2 and st.leg_ids == ("a", "c") and st.leg_index == {"a": 0, "c": 1}
    np.testing.assert_array_equal(st.ticks, [0.01, 0.01])
    snap = st.snapshot(10)
    assert snap.valid and snap.tradable and snap.n_legs == 2
    assert snap.s_bid == pytest.approx(0.70) and snap.s_ask == pytest.approx(0.72)
    st.set_leg(st.leg_index["c"], 0.16, 0.18, 1, 1, 11, True)
    assert st.s_mid == pytest.approx(0.555 + 0.17)
    with pytest.raises(KeyError):
        st.remove_leg("b")
    st.remove_leg("a")
    with pytest.raises(ValueError):
        st.remove_leg("c")


def test_basket_state_from_basket_and_validation() -> None:
    legs = tuple(
        Leg(leg_id=f"l{i}", label=f"L{i}", yes_token_id=f"SYNTHETIC-Y{i}", no_token_id=f"SYNTHETIC-N{i}",
            tick_size=0.001 if i == 2 else 0.01, accepting_orders=i != 1)
        for i in range(3)
    )
    st = BasketState.from_basket(Basket(basket_id="SYNTHETIC_t", title="t", legs=legs, synthetic=True))
    assert st.leg_ids == ("SYNTHETIC-Y0", "SYNTHETIC-Y1", "SYNTHETIC-Y2")
    np.testing.assert_array_equal(st.tradable, [True, False, True])
    np.testing.assert_array_equal(st.ticks, [0.01, 0.01, 0.001])
    with pytest.raises(ValueError):
        BasketState(["a", "a"])
    with pytest.raises(ValueError):
        BasketState(["a", "b"], ticks=[0.01])
    snap = BasketSnapshot(0, 0.97, 0.96, 0.99, True, 0.03, 3, 0, 0.0, True)
    assert snap.band_side is Side.LONG_BASKET and snap.clean


# --------------------------------------------------------------------------- payoffs / economics
@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_payoff_identities_exact(n: int) -> None:
    """Exactly one leg wins: a YES set pays 1 and a NO set pays n - 1 in every state."""
    for winner in range(n):
        yes_payoff = sum(Fraction(int(i == winner)) for i in range(n))
        no_payoff = sum(Fraction(int(i != winner)) for i in range(n))
        assert yes_payoff == 1 and no_payoff == n - 1
    # so both baskets are perfectly hedged: P&L does not depend on which leg wins
    rng = np.random.default_rng(n)
    b = [Fraction(int(x), 1000) for x in rng.integers(1, 900, n)]
    a = [x + Fraction(int(k), 1000) for x, k in zip(b, rng.integers(1, 20, n), strict=True)]
    for winner in range(n):
        long_pnl = sum(Fraction(int(i == winner)) for i in range(n)) - sum(a)
        short_pnl = sum(Fraction(int(i != winner)) for i in range(n)) - sum(1 - x for x in b)
        assert long_pnl == 1 - sum(a)
        assert short_pnl == sum(b) - 1


def test_three_leg_example_exact() -> None:
    b = [Fraction(55, 100), Fraction(33, 100), Fraction(15, 100)]
    a = [Fraction(56, 100), Fraction(34, 100), Fraction(16, 100)]
    n = 3
    no_cost = sum(1 - x for x in b)            # NO asks are 1 - b (mirrored books)
    assert no_cost == Fraction(197, 100)       # 1.97
    assert Fraction(n - 1) == 2                # convert / hold value of the NO set: 2.00
    assert (n - 1) - no_cost == Fraction(3, 100) == sum(b) - 1  # short edge 0.03 before fees
    assert 1 - sum(a) == Fraction(-6, 100)     # long edge 1 - 1.06 = -0.06
    # with fees r = 0.04 on every NO buy, the converted short realises exactly S_bid - 1 - fees
    r = Fraction(4, 100)
    fees = sum(r * (1 - x) * x for x in b)
    assert (n - 1) - (no_cost + fees) == sum(b) - 1 - fees == Fraction(3, 100) - Fraction(23844, 1_000_000)


def test_converted_short_inside_band_locks_in_a_loss() -> None:
    """A short entered at S_bid < 1 and converted realises exactly S_bid - 1 - fees whatever
    happens next: z-timing cannot rescue a taker short entered inside the band."""
    b = [Fraction(50, 100), Fraction(30, 100), Fraction(15, 100)]  # S_bid = 0.95
    r = Fraction(4, 100)
    cost = sum((1 - x) + r * x * (1 - x) for x in b)  # NO at 1 - b plus the symmetric fee
    pnl = 2 - cost  # convert: n - 1 = 2 collateral now (equals the NO-set payoff in every state)
    assert pnl == sum(b) - 1 - r * sum(x * (1 - x) for x in b) < 0
    _, short_edge = executable_edges(np.array([float(x) for x in b]), np.array([0.51, 0.31, 0.16]),
                                     [FeeSchedule(rate=0.04)] * 3)
    assert short_edge == pytest.approx(float(pnl), abs=1e-12)


def test_executable_edges_and_hurdle_hand_values() -> None:
    lo, sh = executable_edges(np.array(BIDS3), np.array(ASKS3))
    assert lo == pytest.approx(-0.06, abs=1e-12) and sh == pytest.approx(0.03, abs=1e-12)
    fees = [FeeSchedule(rate=0.04)] * 3
    lo, sh = executable_edges(np.array(BIDS3), np.array(ASKS3), fees)
    # Σ a(1-a) = .2464 + .2244 + .1344 = .6052 ; Σ b(1-b) = .2475 + .2211 + .1275 = .5961
    assert lo == pytest.approx(-0.06 - 0.04 * 0.6052, abs=1e-12)  # -0.084208
    assert sh == pytest.approx(0.03 - 0.04 * 0.5961, abs=1e-12)   # 0.006156
    h = basket_cost_hurdle(np.array(BIDS3), np.array(ASKS3), fees)
    assert h["spread_sum"] == pytest.approx(0.03)
    assert h["fee_entry_long"] == pytest.approx(0.024208) and h["fee_exit_long"] == pytest.approx(0.023844)
    assert h["fee_entry_short"] == pytest.approx(0.023844) and h["fee_exit_short"] == pytest.approx(0.024208)
    assert h["hurdle_long"] == pytest.approx(0.078052) and h["hurdle_short"] == pytest.approx(0.078052)
    # per-unit fees agree with the config helpers
    assert h["fee_entry_long"] == pytest.approx(basket_fee_per_unit(fees, np.array(ASKS3)))
    assert h["fee_entry_short"] == pytest.approx(sum(f.fee_per_unit(1 - x) for f, x in zip(fees, BIDS3, strict=True)))


def test_executable_edges_missing_sides_and_vectorised() -> None:
    fees = [FeeSchedule(rate=0.04), FeeSchedule(rate=0.07), FeeSchedule(rate=0.0)]
    bid = np.array([[0.55, 0.33, 0.15], [0.60, NAN, 0.15], [0.50, 0.30, 0.25]])
    ask = np.array([[0.56, 0.34, 0.16], [0.61, 0.30, NAN], [0.52, 0.31, 0.26]])
    lo, sh = executable_edges(bid, ask, fees)
    assert lo.shape == sh.shape == (3,)
    for k in range(3):
        l1, s1 = executable_edges(bid[k], ask[k], fees)
        assert (lo[k], sh[k]) == pytest.approx((l1, s1), abs=1e-15)
    # row 1: missing ask counts 1 (fee 0) and missing bid counts 0 (NO at 1, fee 0)
    assert lo[1] == pytest.approx(1 - (0.61 + 0.30 + 1.0) - 0.04 * 0.61 * 0.39 - 0.07 * 0.30 * 0.70)
    assert sh[1] == pytest.approx((0.60 + 0.0 + 0.15) - 1 - 0.04 * 0.6 * 0.4)
    h = basket_cost_hurdle(bid, ask, fees)
    assert h["hurdle_long"].shape == (3,) and h["spread_sum"][1] == pytest.approx(1.16)
    with pytest.raises(ValueError):
        executable_edges(bid, ask, fees[:2])


def test_fee_closed_form_one_minus_hhi() -> None:
    """Σ r p(1-p) = r (1 - Σ p²) when Σ p = 1 (exact), and the float helpers agree."""
    r = Fraction(4, 100)
    for p in ([Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)],
              [Fraction(7, 10), Fraction(1, 10), Fraction(1, 10), Fraction(1, 10)]):
        assert sum(p) == 1
        assert sum(r * x * (1 - x) for x in p) == r * (1 - sum(x * x for x in p))
    p = np.array([0.5, 0.3, 0.15, 0.05])
    fees = [FeeSchedule(rate=0.04)] * 4
    closed = 0.04 * (1 - float(np.sum(p ** 2)))
    assert basket_fee_per_unit(fees, p) == pytest.approx(closed, rel=1e-12)
    h = basket_cost_hurdle(p, p, fees)  # zero-spread book at p
    assert h["fee_entry_long"] == pytest.approx(closed, rel=1e-12) and h["spread_sum"] == 0.0


def test_edge_to_cost_ratio() -> None:
    s = np.array([1.0, 1.02, NAN, 0.98, 1.0])
    expected = np.std([1.0, 1.02, 0.98, 1.0], ddof=1) / 0.078052
    assert edge_to_cost_ratio(s, 0.078052) == pytest.approx(expected)
    assert math.isnan(edge_to_cost_ratio(np.array([1.0, NAN]), 0.05))
    with pytest.raises(ValueError):
        edge_to_cost_ratio(s, 0.0)


# --------------------------------------------------------------------------- hygiene / perf
def test_import_does_not_load_pandas() -> None:
    code = "import sys, src.arb_engine; sys.exit('pandas' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.slow
def test_micro_benchmark_basket_and_z_updates() -> None:
    """Non-failing report: 100k BasketState + ZScoreEngine + state machine updates."""
    n_legs, n_upd = 10, 100_000
    rng = np.random.default_rng(0)
    legs = rng.integers(n_legs, size=n_upd).tolist()
    bids = (np.round(rng.uniform(0.02, 0.15, n_upd), 3)).tolist()
    st = BasketState([f"L{i}" for i in range(n_legs)])
    for i in range(n_legs):
        st.set_leg(i, 0.09, 0.1, 1, 1, 0, True)
    eng, sm = ZScoreEngine(ZConfig(window=500)), SignalStateMachine()
    t0 = time.perf_counter()
    for k in range(n_upd):
        b = bids[k]
        st.set_leg(legs[k], b, b + 0.01, 10.0, 10.0, k, True)
        snap = st.snapshot(k)
        sm.step(k, eng.update(k, snap.s_mid, snap.valid))
    dt = time.perf_counter() - t0
    log.info("100k basket+z updates: %.3f s (%.2f us/update)", dt, dt / n_upd * 1e6)
    assert eng.stats.full
