"""Basket discovery from Polymarket's Gamma API -> ``config/markets.json``.

Token IDs are never typed by hand: ``python -m src.discovery add --slug ... --basket-id ...``
fetches the event, keeps only the markets that are genuine tradable outcomes, and writes
them as basket legs. The rules below come from live data (2026-10-01):

* **Closed markets are resolved legs** (e.g. eliminated MLB teams: ``closed=true``,
  ``acceptingOrders=false``, best ask 0.001). They are excluded and recorded with the
  outcome, because a resolved-NO leg no longer contributes to the sum.
* **Inactive markets are placeholders** of an *augmented* negRisk event ("Party A-F",
  unnamed "Other" with ``negRiskOther``). They quote 0/1 and would wreck the sum, so they
  are excluded; the named legs then sum to < 1 by P(unnamed outcome) - a structural
  discount the notebooks report rather than trade.
* **Paused named legs are kept** (``accepting_orders=False``): dropping them would make
  the basket non-exhaustive and fake a discount.
* A *named, active* "Other" outcome (Balance of Power) is an ordinary leg.

Interview pro-tip 2 lives in :func:`rank_candidates`: only liquid, mutually exclusive
baskets are worth monitoring, because the n-leg spread plus taker fees (crypto carries the
highest rate, 0.07) is the hurdle every round trip must clear.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import requests

from . import CONFIG_PATH
from .clob_client import ClobHTTPError, ClobRestClient
from .config import (
    Basket,
    Defaults,
    Leg,
    MarketsConfig,
    load_markets,
    resolve_fee_schedule,
    save_markets,
    upsert_basket,
)

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
RULES_MAX_CHARS = 1500


class DiscoveryError(RuntimeError):
    """The event could not be found or is not usable as a basket."""


# --------------------------------------------------------------------------- client
class GammaClient:
    """Public Gamma API reads. Retries/backoff/throttling are shared with the CLOB client."""

    def __init__(self, base_url: str = GAMMA_URL, session: requests.Session | None = None,
                 timeout_s: float = 15.0, max_retries: int = 4, **kw: Any) -> None:
        self._http = ClobRestClient(base_url=base_url, session=session, timeout_s=timeout_s,
                                    max_retries=max_retries, **kw)

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return self._http._request("GET", path, params=params)

    def get_event_by_slug(self, slug: str) -> dict:
        """``GET /events/slug/{slug}``, falling back to ``GET /events?slug=`` (an array)."""
        try:
            ev = self._get(f"/events/slug/{slug}")
            if isinstance(ev, dict) and ev:
                return ev
        except ClobHTTPError as exc:
            if exc.status != 404:
                raise
        arr = self._get("/events", params={"slug": slug})
        if isinstance(arr, list) and arr:
            return arr[0]
        raise DiscoveryError(f"no Gamma event with slug {slug!r}")

    def iter_events(self, *, closed: bool = False, order: str = "volume24hr", ascending: bool = False,
                    limit: int = 50, tag_id: int | None = None, max_pages: int = 20) -> Iterator[dict]:
        """Keyset pagination over ``/events/keyset``; falls back to ``/events?offset=``."""
        params: dict[str, Any] = {"closed": str(closed).lower(), "order": order,
                                  "ascending": str(ascending).lower(), "limit": limit}
        if tag_id is not None:
            params["tag_id"] = tag_id
        try:
            cursor = None
            for _ in range(max_pages):
                p = dict(params, **({"after_cursor": cursor} if cursor else {}))
                page = self._get("/events/keyset", params=p)
                events = page.get("events", []) if isinstance(page, dict) else []
                yield from events
                cursor = page.get("next_cursor") if isinstance(page, dict) else None
                if not cursor or not events:
                    return
        except ClobHTTPError as exc:
            if exc.status not in (400, 404):
                raise
            logger.info("keyset pagination unavailable (%s); using offset pagination", exc.status)
        p = dict(params, active="true")
        for page_no in range(max_pages):
            events = self._get("/events", params=dict(p, offset=page_no * limit))
            if not isinstance(events, list) or not events:
                return
            yield from events

    def get_tags(self) -> list[dict]:
        tags = self._get("/tags")
        return tags if isinstance(tags, list) else []


# --------------------------------------------------------------------------- parsing helpers
def decode_list_field(v: Any) -> list | None:
    """Gamma encodes ``outcomes``/``outcomePrices``/``clobTokenIds`` as JSON *strings*."""
    if v is None or v == "":
        return None
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            out = json.loads(v)
        except ValueError:
            return None
        return list(out) if isinstance(out, (list, tuple)) else None
    return None


def market_asset_ids(m: Mapping[str, Any]) -> tuple[str, str]:
    """(YES, NO) CLOB asset ids. Index 0 is YES; ``version == "v2"`` markets use ``positionIds``."""
    key = "positionIds" if str(m.get("version", "v1")).lower() == "v2" and m.get("positionIds") else "clobTokenIds"
    ids = decode_list_field(m.get(key)) or []
    if len(ids) < 2:
        raise DiscoveryError(f"market {m.get('conditionId') or m.get('id')} has no {key}")
    return str(ids[0]), str(ids[1])


def _f(v: Any, default: float | None = None) -> float | None:
    try:
        return default if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return default


def slugify(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return s or "leg"


def classify_market(m: Mapping[str, Any]) -> str:
    """``leg`` | ``resolved_yes`` | ``resolved_no`` | ``placeholder`` | ``inactive_other``."""
    if m.get("closed"):
        prices = decode_list_field(m.get("outcomePrices")) or []
        yes = _f(prices[0]) if prices else None
        return "resolved_yes" if yes is not None and yes >= 0.99 else "resolved_no"
    if not m.get("active", True):
        return "inactive_other" if m.get("negRiskOther") else "placeholder"
    return "leg"


def event_category(event: Mapping[str, Any], defaults: Defaults) -> str | None:
    """First event tag that matches a fee category (politics, crypto, sports, ...)."""
    keys = {k.lower() for k in defaults.fee_category_rates}
    for tag in event.get("tags") or []:
        for field in ("slug", "label"):
            v = str(tag.get(field) or "").lower()
            if v in keys:
                return v
    return None


def is_basket_candidate(event: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Mutually exclusive + exhaustive + tradable on the CLOB."""
    reasons = []
    if not event.get("negRisk"):
        reasons.append("not negRisk (no sum-to-one guarantee)")
    if event.get("enableOrderBook") is False:
        reasons.append("order book disabled")
    if event.get("cumulativeMarkets"):
        reasons.append("cumulative/threshold markets are nested, not mutually exclusive")
    if event.get("closed"):
        reasons.append("event closed")
    return (not reasons, reasons)


# --------------------------------------------------------------------------- event -> basket
def event_to_basket(event: Mapping[str, Any], *, basket_id: str, defaults: Defaults,
                    category: str | None = None) -> tuple[Basket, dict[str, Any]]:
    """Build a ``Basket`` from a Gamma event. Returns ``(basket, report)``."""
    category = category or event_category(event, defaults)
    augmented = bool(event.get("negRiskAugmented"))
    legs: list[Leg] = []
    excluded: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for m in event.get("markets") or []:
        kind = classify_market(m)
        label = str(m.get("groupItemTitle") or m.get("question") or m.get("slug") or "?")
        try:
            yes_id, no_id = market_asset_ids(m)
        except DiscoveryError:
            excluded.append({"label": label, "condition_id": m.get("conditionId"), "yes_token_id": None,
                             "reason": "no_token_ids"})
            continue
        if kind != "leg":
            excluded.append({"label": label, "condition_id": m.get("conditionId"),
                             "yes_token_id": yes_id, "reason": kind})
            continue
        leg_id = slugify(label)
        while leg_id in seen_ids:
            leg_id += "-x"
        seen_ids.add(leg_id)
        legs.append(Leg(
            leg_id=leg_id,
            label=label,
            yes_token_id=yes_id,
            no_token_id=no_id,
            condition_id=str(m.get("conditionId") or ""),
            question=str(m.get("question") or ""),
            active=bool(m.get("active", True)),
            accepting_orders=bool(m.get("acceptingOrders", True)),
            closed=False,
            is_other=bool(m.get("negRiskOther")),
            is_placeholder=False,
            tick_size=_f(m.get("orderPriceMinTickSize"), 0.01) or 0.01,
            min_order_size=_f(m.get("orderMinSize"), 5.0) or 5.0,
            fee=resolve_fee_schedule(m, category=category, defaults=defaults),
            market_version=str(m.get("version") or "v1"),
            created_at=m.get("createdAt"),
            rules=str(m.get("description") or "")[:RULES_MAX_CHARS],
        ))
    resolved_yes = any(x["reason"] == "resolved_yes" for x in excluded)
    basket = Basket(
        basket_id=basket_id,
        title=str(event.get("title") or basket_id),
        legs=tuple(legs),
        event_slug=event.get("slug"),
        event_id=None if event.get("id") is None else str(event.get("id")),
        category=category,
        status="resolved" if resolved_yes else ("active" if len(legs) >= 2 else "stale"),
        neg_risk=bool(event.get("negRisk")),
        neg_risk_augmented=augmented,
        neg_risk_market_id=event.get("negRiskMarketID"),
        convert_fee_bips=_f(event.get("negRiskFeeBips"), 0.0) or 0.0,
        end_date=event.get("endDate"),
        structural_discount_expected=augmented and not any(l.is_other for l in legs),
        discovered_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        excluded=tuple(excluded),
    )
    counts: dict[str, int] = {}
    for x in excluded:
        counts[x["reason"]] = counts.get(x["reason"], 0) + 1
    ok, reasons = is_basket_candidate(event)
    warnings_ = list(reasons)
    if basket.structural_discount_expected:
        warnings_.append("augmented negRisk: named legs sum to < 1 by P(unnamed/Other outcome)")
    if any(not l.accepting_orders for l in legs):
        warnings_.append("some legs are paused (not accepting orders); kept so the basket stays exhaustive")
    report = {"n_markets": len(event.get("markets") or []), "n_legs": len(legs),
              "excluded": counts, "warnings": warnings_, "category": category}
    return basket, report


# --------------------------------------------------------------------------- ranking (pro-tip 2)
@dataclass(frozen=True)
class Candidate:
    slug: str
    title: str
    n_legs: int
    volume24hr: float
    liquidity: float
    augmented: bool
    fee_rate: float
    sum_bid: float
    sum_ask: float
    sum_mid: float
    spread_sum: float
    fee_per_basket: float
    hurdle: float
    ecr: float | None = None  # sigma(S_mid)/hurdle needs price history: computed in notebook 01


def rank_candidates(events: Sequence[Mapping[str, Any]], *, defaults: Defaults, min_volume24h: float = 50_000,
                    max_leg_spread: float = 0.05, min_legs: int = 2, max_legs: int = 40,
                    tag_slugs: Sequence[str] | None = None) -> list[Candidate]:
    """Liquid, mutually exclusive baskets ranked by 24 h volume.

    The cost hurdle of one taker round trip on the full basket is
    ``Σ spread_i + 2·F`` with ``F = Σ r_i p_i (1 - p_i)``: a dislocation of the basket sum
    smaller than that cannot be monetised. Crypto baskets trade often but carry the
    highest fee rate, so their hurdle is the largest for a given spread.
    """
    out: list[Candidate] = []
    wanted = {t.lower() for t in tag_slugs} if tag_slugs else None
    for ev in events:
        ok, _ = is_basket_candidate(ev)
        if not ok:
            continue
        if wanted is not None:
            tags = {str(t.get("slug") or "").lower() for t in ev.get("tags") or []}
            if not tags & wanted:
                continue
        vol = _f(ev.get("volume24hr"), 0.0) or 0.0
        if vol < min_volume24h:
            continue
        legs = [m for m in ev.get("markets") or [] if classify_market(m) == "leg"]
        if not (min_legs <= len(legs) <= max_legs):
            continue
        bids = [_f(m.get("bestBid"), 0.0) or 0.0 for m in legs]
        asks = [_f(m.get("bestAsk"), 1.0) or 1.0 for m in legs]
        if max(a - b for a, b in zip(asks, bids)) > max_leg_spread:
            continue
        mids = [(a + b) / 2 for a, b in zip(asks, bids)]
        rates = []
        for m in legs:
            fs = resolve_fee_schedule(m, category=event_category(ev, defaults), defaults=defaults)
            rates.append(fs)
        fee = sum(f.fee_per_unit(p) for f, p in zip(rates, mids))
        spread_sum = sum(a - b for a, b in zip(asks, bids))
        out.append(Candidate(
            slug=str(ev.get("slug")), title=str(ev.get("title")), n_legs=len(legs), volume24hr=vol,
            liquidity=_f(ev.get("liquidity"), 0.0) or 0.0, augmented=bool(ev.get("negRiskAugmented")),
            fee_rate=max((f.rate for f in rates), default=0.0), sum_bid=sum(bids), sum_ask=sum(asks),
            sum_mid=sum(mids), spread_sum=spread_sum, fee_per_basket=fee, hurdle=spread_sum + 2 * fee,
        ))
    return sorted(out, key=lambda c: -c.volume24hr)


# --------------------------------------------------------------------------- CLI
def format_slug(template: str, date: str) -> str:
    """``bitcoin-price-on-{month}-{day}-{year}`` + ``2026-10-02`` -> ``bitcoin-price-on-october-2-2026``."""
    d = datetime.strptime(date, "%Y-%m-%d")
    return template.format(month=d.strftime("%B").lower(), day=d.day, year=d.year)


def add_basket(cfg: MarketsConfig, client: GammaClient, *, slug: str, basket_id: str,
               category: str | None = None, roles: Sequence[str] | None = None) -> tuple[MarketsConfig, Basket, dict]:
    event = client.get_event_by_slug(slug)
    basket, report = event_to_basket(event, basket_id=basket_id, defaults=cfg.defaults, category=category)
    if roles:
        basket = replace(basket, roles=tuple(roles))
    old = cfg.baskets.get(basket_id)
    if old is not None and old.slug_template and not basket.slug_template:
        basket = replace(basket, slug_template=old.slug_template)
    return upsert_basket(cfg, basket), basket, report


def diff_legs(old: Basket, new: Basket) -> dict[str, list[str]]:
    o = {l.yes_token_id: l.label for l in old.legs}
    n = {l.yes_token_id: l.label for l in new.legs}
    return {"added": [n[k] for k in n if k not in o], "removed": [o[k] for k in o if k not in n]}


def _print_basket(b: Basket, report: Mapping[str, Any] | None = None) -> None:
    print(f"{b.basket_id}: {b.title}  [{b.status}] negRisk={b.neg_risk} augmented={b.neg_risk_augmented}")
    for l in b.legs:
        flags = ("" if l.accepting_orders else " [paused]") + (" [other]" if l.is_other else "")
        print(f"  - {l.label:<32} fee={l.fee.rate:.3f} tick={l.tick_size:g}{flags}")
    if b.excluded:
        print(f"  excluded {len(b.excluded)}: " + ", ".join(f"{x['label']} ({x['reason']})" for x in b.excluded[:12])
              + (" ..." if len(b.excluded) > 12 else ""))
    if report:
        for w in report.get("warnings", []):
            print(f"  ! {w}")


def main(argv: Sequence[str] | None = None, *, session: requests.Session | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.discovery", description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="rank liquid negRisk baskets")
    s.add_argument("--min-volume24h", type=float, default=50_000)
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--tag", action="append")
    a = sub.add_parser("add", help="fetch an event and write it as a basket")
    a.add_argument("--basket-id", required=True)
    a.add_argument("--slug")
    a.add_argument("--slug-template")
    a.add_argument("--date")
    a.add_argument("--category")
    a.add_argument("--roles")
    r = sub.add_parser("refresh", help="re-fetch baskets and report leg changes")
    r.add_argument("--basket-id")
    sh = sub.add_parser("show")
    sh.add_argument("--basket-id")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    cfg = load_markets(args.config)
    client = GammaClient(session=session)
    if args.cmd == "search":
        events = list(client.iter_events(limit=min(100, max(args.limit, 50))))
        for c in rank_candidates(events, defaults=cfg.defaults, min_volume24h=args.min_volume24h,
                                 tag_slugs=args.tag)[: args.limit]:
            print(f"{c.volume24hr:>12,.0f}  n={c.n_legs:<3} S_bid={c.sum_bid:.3f} S_ask={c.sum_ask:.3f} "
                  f"hurdle={c.hurdle:.3f} fee={c.fee_rate:.2f}{' aug' if c.augmented else ''}  {c.slug}")
        return 0
    if args.cmd == "add":
        slug = args.slug
        if not slug:
            if not (args.slug_template and args.date):
                ap.error("add needs --slug, or --slug-template with --date")
            slug = format_slug(args.slug_template, args.date)
        roles = [x for x in (args.roles or "").split(",") if x] or None
        cfg, basket, report = add_basket(cfg, client, slug=slug, basket_id=args.basket_id,
                                         category=args.category, roles=roles)
        save_markets(cfg, args.config)
        _print_basket(basket, report)
        return 0
    if args.cmd == "refresh":
        targets = [b for b in cfg.baskets.values() if b.event_slug and not b.synthetic
                   and (args.basket_id in (None, b.basket_id))]
        for old in targets:
            cfg, new, report = add_basket(cfg, client, slug=old.event_slug, basket_id=old.basket_id)
            d = diff_legs(old, new)
            print(f"{old.basket_id}: {len(new.legs)} legs; added={d['added']} removed={d['removed']} "
                  f"status={new.status}")
        save_markets(cfg, args.config)
        return 0
    for b in cfg.baskets.values():
        if args.basket_id in (None, b.basket_id):
            _print_basket(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
