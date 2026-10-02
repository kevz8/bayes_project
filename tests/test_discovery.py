"""Discovery against REAL captured Gamma responses (tests/fixtures/captured_*)."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from src import CONFIG_PATH
from src.config import Defaults, get_basket, load_markets
from src.discovery import (
    GammaClient,
    classify_market,
    decode_list_field,
    event_to_basket,
    format_slug,
    is_basket_candidate,
    main,
    market_asset_ids,
    rank_candidates,
)
from tests.conftest import FakeResponse, FakeSession

D = Defaults()
FED = "captured_gamma_event_fed_oct_2026.json"
HOUSE = "captured_gamma_event_house_2026_augmented.json"
MLB = "captured_gamma_event_mlb_ws_2026.json"
BOP = "captured_gamma_event_balance_of_power_2026.json"


def test_decode_list_field():
    assert decode_list_field('["a", "b"]') == ["a", "b"]
    assert decode_list_field(["a"]) == ["a"]
    assert decode_list_field(None) is None and decode_list_field("") is None
    assert decode_list_field("not json") is None


def test_market_asset_ids_v1_and_v2():
    m = {"clobTokenIds": '["111", "222"]'}
    assert market_asset_ids(m) == ("111", "222")
    v2 = {"version": "v2", "positionIds": ["333", "444"], "clobTokenIds": None}
    assert market_asset_ids(v2) == ("333", "444")


def test_fed_basket(fixture_json):
    ev = fixture_json(FED)
    b, rep = event_to_basket(ev, basket_id="fed-oct-2026", defaults=D)
    assert [l.label for l in b.legs] == ["50+ bps decrease", "25 bps decrease", "No change",
                                         "25 bps increase", "50+ bps increase"]
    for l, m in zip(b.legs, ev["markets"]):
        assert l.yes_token_id == json.loads(m["clobTokenIds"])[0] and l.yes_token_id.isdigit()
        assert l.fee.rate == 0.05 and l.fee.source == "gamma_feeSchedule"
    assert b.neg_risk and not b.neg_risk_augmented and not b.structural_discount_expected
    assert b.is_complete_partition and b.status == "active" and not b.excluded
    assert rep["n_legs"] == 5


def test_house_augmented_excludes_placeholders(fixture_json):
    b, rep = event_to_basket(fixture_json(HOUSE), basket_id="house-2026", defaults=D)
    assert [l.label for l in b.legs] == ["Democratic Party", "Republican Party"]
    assert rep["excluded"] == {"placeholder": 6, "inactive_other": 1}
    assert b.neg_risk_augmented and b.structural_discount_expected
    assert not b.is_complete_partition  # named legs do not cover "Other"
    assert any("structural" in w or "< 1" in w for w in rep["warnings"])


def test_mlb_closed_legs_are_resolved(fixture_json):
    b, rep = event_to_basket(fixture_json(MLB), basket_id="mlb", defaults=D)
    assert len(b.legs) == 9
    assert rep["excluded"] == {"resolved_no": 21, "inactive_other": 1}
    assert b.structural_discount_expected
    assert all(l.fee.rate == 0.03 for l in b.legs)


def test_balance_of_power_named_other_is_a_leg(fixture_json):
    b, _ = event_to_basket(fixture_json(BOP), basket_id="bop", defaults=D)
    labels = [l.label for l in b.legs]
    assert labels == ["Democrats Sweep", "D Senate, R House", "R Senate, D House", "Republicans Sweep", "Other"]
    assert not any(l.is_other for l in b.legs)  # named outcome, not the negRisk "Other" placeholder
    assert all(l.fee.rate == 0.04 for l in b.legs)


def test_paused_named_leg_is_kept(fixture_json):
    ev = copy.deepcopy(fixture_json(FED))
    ev["markets"][2]["acceptingOrders"] = False
    b, rep = event_to_basket(ev, basket_id="fed", defaults=D)
    assert len(b.legs) == 5 and not b.legs[2].accepting_orders
    assert any("paused" in w for w in rep["warnings"])


def test_classify_market():
    assert classify_market({"closed": True, "outcomePrices": '["1", "0"]'}) == "resolved_yes"
    assert classify_market({"closed": True, "outcomePrices": '["0", "1"]'}) == "resolved_no"
    assert classify_market({"active": False}) == "placeholder"
    assert classify_market({"active": False, "negRiskOther": True}) == "inactive_other"
    assert classify_market({"active": True}) == "leg"


def test_candidate_filter_and_hurdle():
    good = {"slug": "g", "title": "G", "negRisk": True, "enableOrderBook": True, "volume24hr": 1e6,
            "markets": [
                {"active": True, "bestBid": 0.59, "bestAsk": 0.61, "clobTokenIds": '["1","2"]',
                 "feeSchedule": {"rate": 0.04, "exponent": 1}},
                {"active": True, "bestBid": 0.39, "bestAsk": 0.41, "clobTokenIds": '["3","4"]',
                 "feeSchedule": {"rate": 0.04, "exponent": 1}},
            ]}
    cum = dict(good, slug="c", cumulativeMarkets=True)
    plain = dict(good, slug="p", negRisk=False)
    assert not is_basket_candidate(cum)[0] and not is_basket_candidate(plain)[0]
    (c,) = rank_candidates([good, cum, plain], defaults=D)
    assert c.slug == "g"
    assert c.spread_sum == pytest.approx(0.04)
    fee = 0.04 * (0.6 * 0.4 + 0.4 * 0.6)  # F = r Σ p(1-p)
    assert c.fee_per_basket == pytest.approx(fee)
    assert c.hurdle == pytest.approx(0.04 + 2 * fee)


def test_slug_404_falls_back_to_query(fixture_json):
    ev = fixture_json(FED)
    s = FakeSession({("GET", "/events/slug/foo"): FakeResponse({"error": "nf"}, status=404),
                     ("GET", "/events"): [[ev]]})
    got = GammaClient(session=s, min_interval_s=0).get_event_by_slug("foo")
    assert got["slug"] == ev["slug"]


def test_keyset_pagination_stops_at_null_cursor():
    pages = [{"events": [{"id": 1}], "next_cursor": "abc"}, {"events": [{"id": 2}], "next_cursor": None}]
    s = FakeSession({("GET", "/events/keyset"): pages})
    ids = [e["id"] for e in GammaClient(session=s, min_interval_s=0).iter_events()]
    assert ids == [1, 2]
    assert s.calls[1][2]["after_cursor"] == "abc"


def test_format_slug():
    assert format_slug("bitcoin-price-on-{month}-{day}-{year}", "2026-10-02") == "bitcoin-price-on-october-2-2026"


def _cfg_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "markets.json"
    shutil.copy(CONFIG_PATH, dst)
    return dst


def test_cli_add_and_refresh(tmp_path, fixture_json, capsys):
    cfg_path = _cfg_copy(tmp_path)
    ev = fixture_json(FED)
    slug = "fed-decision-in-october-20260617190323537"
    s = FakeSession({("GET", f"/events/slug/{slug}"): ev})
    assert main(["--config", str(cfg_path), "add", "--slug", slug, "--basket-id", "fed-oct-2026"], session=s) == 0
    cfg = load_markets(cfg_path)
    b = get_basket("fed-oct-2026", cfg)
    assert b.n_legs == 5 and b.status == "active"
    assert b.roles == ("record", "eda", "backtest")  # user fields preserved
    assert "Primary example" in b.notes

    ev2 = copy.deepcopy(ev)
    ev2["markets"][0]["closed"] = True
    ev2["markets"][0]["outcomePrices"] = '["0", "1"]'
    s2 = FakeSession({("GET", f"/events/slug/{slug}"): ev2})
    capsys.readouterr()
    assert main(["--config", str(cfg_path), "refresh", "--basket-id", "fed-oct-2026"], session=s2) == 0
    out = capsys.readouterr().out
    assert "removed=['50+ bps decrease']" in out
    assert get_basket("fed-oct-2026", load_markets(cfg_path)).n_legs == 4
