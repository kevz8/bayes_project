"""Generated conclusion text must match what the live books actually showed."""
from __future__ import annotations

from src.research import band_sentence


def test_band_sentence_inside_band_says_no_trade():
    s = band_sentence(6.4, 1.013, 0.990, 0.0, 0.0, 0)
    assert "never cost less than $1.013" in s and "no risk-free trade was available" in s


def test_band_sentence_reports_band_crossing_honestly():
    s = band_sentence(151.6, 1.001, 1.021, 0.0, 0.009, 0)
    assert "never sold for more than" not in s
    assert "more than $1 only 0.9% of the time (at most $1.021)" in s
    assert "found no profitable risk-free trade" in s


def test_band_sentence_reports_arb_trades_when_found():
    assert "found 3 risk-free trades" in band_sentence(10, 0.99, 0.98, 0.05, 0.0, 3)


def test_band_sentence_without_backtest_uses_fee_comparison():
    assert "never by more than the taker fees" in band_sentence(10, 1.001, 1.021, 0.0, 0.009, share_beyond_fees=0.0)
    assert "more than the taker fees 0.50% of the time" in band_sentence(10, 1.001, 1.08, 0.0, 0.01, share_beyond_fees=0.005)
