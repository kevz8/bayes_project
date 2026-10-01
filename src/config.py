"""Load, validate and save ``config/markets.json``.

The config lists *baskets* (mutually-exclusive outcome groups, i.e. Polymarket negRisk
events), their legs (YES/NO token pairs), cross-market *pairings* used by notebook 02,
global simulation *defaults*, and seeded *synthetic* baskets.

Token IDs are written only by ``python -m src.discovery`` from Gamma API responses -
they are never typed by hand. A basket whose legs have not been discovered yet has
``status: "unresolved"`` and cannot be recorded or backtested.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import CONFIG_PATH

SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """Invalid or incomplete ``markets.json``."""


# --------------------------------------------------------------------------- fees
@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """Polymarket taker fee: ``fee = shares * rate * (p * (1 - p)) ** exponent``.

    Makers pay nothing. The curve is symmetric in ``p``, so buying NO at ``1 - b``
    costs the same fee as selling YES at ``b``. Fees are rounded to 5 decimals per order.
    """

    rate: float = 0.0
    exponent: float = 1.0
    taker_only: bool = True
    rebate_rate: float = 0.0
    source: str = "default"  # gamma_feeSchedule | clob_fd | category | fee_free | default

    def raw_fee(self, shares: float, price: float) -> float:
        if self.rate == 0.0 or shares <= 0.0:
            return 0.0
        p = min(max(float(price), 0.0), 1.0)
        return float(shares) * self.rate * (p * (1.0 - p)) ** self.exponent

    def taker_fee(self, shares: float, price: float) -> float:
        """Fee in USD for one taker order of ``shares`` at ``price`` (rounded to 5 dp)."""
        return round(self.raw_fee(shares, price), 5)

    def taker_fee_vec(self, shares: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Unrounded per-level fees (callers round the per-order total)."""
        p = np.clip(np.asarray(prices, dtype=float), 0.0, 1.0)
        return np.asarray(shares, dtype=float) * self.rate * (p * (1.0 - p)) ** self.exponent

    def fee_per_unit(self, price: float) -> float:
        """Unrounded fee for one share - used for hurdles and gross-ups."""
        return self.raw_fee(1.0, price)


FEE_FREE = FeeSchedule(rate=0.0, source="fee_free")


# --------------------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class Leg:
    leg_id: str
    label: str
    yes_token_id: str
    no_token_id: str
    condition_id: str = ""
    question: str = ""
    active: bool = True
    accepting_orders: bool = True
    closed: bool = False
    is_other: bool = False
    is_placeholder: bool = False
    tick_size: float = 0.01
    min_order_size: float = 5.0
    fee: FeeSchedule = FeeSchedule()
    market_version: str = "v1"
    created_at: str | None = None
    rules: str = ""


@dataclass(frozen=True)
class Basket:
    basket_id: str
    title: str
    legs: tuple[Leg, ...] = ()
    event_slug: str | None = None
    event_id: str | None = None
    category: str | None = None
    roles: tuple[str, ...] = ()
    status: str = "unresolved"  # unresolved | active | resolved | stale
    neg_risk: bool | None = None
    neg_risk_augmented: bool | None = None
    neg_risk_market_id: str | None = None
    convert_fee_bips: float = 0.0
    end_date: str | None = None
    structural_discount_expected: bool = False
    discovered_at: str | None = None
    notes: str = ""
    synthetic: bool = False
    slug_template: str | None = None

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def yes_ids(self) -> tuple[str, ...]:
        return tuple(l.yes_token_id for l in self.legs)

    @property
    def no_ids(self) -> tuple[str, ...]:
        return tuple(l.no_token_id for l in self.legs)

    @property
    def leg_index(self) -> dict[str, int]:
        """YES token id -> row index."""
        return {l.yes_token_id: i for i, l in enumerate(self.legs)}

    @property
    def no_to_yes(self) -> dict[str, str]:
        return {l.no_token_id: l.yes_token_id for l in self.legs}

    @property
    def fees(self) -> tuple[FeeSchedule, ...]:
        return tuple(l.fee for l in self.legs)

    @property
    def ticks(self) -> np.ndarray:
        return np.array([l.tick_size for l in self.legs], dtype=float)

    @property
    def is_resolved(self) -> bool:
        return self.status != "unresolved" and len(self.legs) >= 2

    @property
    def is_complete_partition(self) -> bool:
        """True when exactly one leg must resolve YES (needed for the sum-to-one identity
        and for the full-NO-set convert). Augmented events need their "Other" leg."""
        if self.synthetic:
            return True
        if not self.neg_risk:
            return False
        if self.neg_risk_augmented:
            return any(l.is_other for l in self.legs)
        return True

    def leg_by_id(self, leg_id: str) -> Leg:
        for l in self.legs:
            if l.leg_id == leg_id:
                return l
        raise KeyError(leg_id)

    def without_leg(self, yes_token_id: str) -> "Basket":
        """Basket after one leg resolved NO early (e.g. a team eliminated)."""
        return replace(self, legs=tuple(l for l in self.legs if l.yes_token_id != yes_token_id))


@dataclass(frozen=True)
class LegSelector:
    basket_id: str
    leg_label_regex: str
    coef: float = 1.0


@dataclass(frozen=True)
class Pairing:
    pairing_id: str
    type: str  # linear_identity | exploratory
    description: str
    lhs: tuple[LegSelector, ...]
    rhs: tuple[LegSelector, ...]
    theory_beta: float | None = None
    notes: str = ""


@dataclass(frozen=True)
class Defaults:
    fee_category_rates: Mapping[str, float] = field(default_factory=lambda: {
        "crypto": 0.07, "sports": 0.05, "economics": 0.05, "culture": 0.05, "weather": 0.05,
        "other": 0.05, "politics": 0.04, "finance": 0.04, "tech": 0.04, "mentions": 0.04,
        "geopolitics": 0.0,
    })
    fee_exponent: float = 1.0
    fees_effective_from: str = "2026-03-30"
    fee_mode: str = "shares"
    risk_free_rate: float = 0.042
    latency_ms: float = 500.0
    latency_jitter_ms: float = 250.0
    max_slip_ticks: int = 3
    participation_cap: float = 0.5
    max_position_frac: float = 0.2
    min_order_size: float = 5.0
    min_notional_usd: float = 1.0
    convert_enabled: bool = False
    gas: Mapping[str, Any] = field(default_factory=lambda: {
        "relayer_pays": True, "gwei": 600.0, "pol_usd": 0.11,
        "units": {"clob_fill": 0, "split": 200000, "merge": 200000, "redeem": 150000,
                  "convert_base": 100000, "convert_per_leg": 80000},
    })
    notes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketsConfig:
    defaults: Defaults
    baskets: Mapping[str, Basket]
    pairings: tuple[Pairing, ...] = ()
    synthetic_baskets: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    updated_at: str | None = None


# --------------------------------------------------------------------------- parsing
def _fee_from_dict(d: Mapping[str, Any] | None) -> FeeSchedule:
    if not d:
        return FeeSchedule()
    return FeeSchedule(
        rate=float(d.get("rate", 0.0) or 0.0),
        exponent=float(d.get("exponent", 1.0) or 1.0),
        taker_only=bool(d.get("taker_only", d.get("takerOnly", True))),
        rebate_rate=float(d.get("rebate_rate", d.get("rebateRate", 0.0)) or 0.0),
        source=str(d.get("source", "default")),
    )


def _leg_from_dict(d: Mapping[str, Any], *, synthetic: bool) -> Leg:
    for key in ("yes_token_id", "no_token_id"):
        v = d.get(key)
        if not isinstance(v, str) or not v:
            raise ConfigError(f"leg {d.get('leg_id')!r}: {key} must be a non-empty string (never a number)")
        if synthetic and not v.startswith("SYNTHETIC-"):
            raise ConfigError(f"synthetic leg token ids must start with 'SYNTHETIC-': {v!r}")
        if not synthetic and not v.isdigit():
            raise ConfigError(f"real token ids are decimal digit strings: {v!r}")
    return Leg(
        leg_id=str(d["leg_id"]),
        label=str(d.get("label", d["leg_id"])),
        yes_token_id=d["yes_token_id"],
        no_token_id=d["no_token_id"],
        condition_id=str(d.get("condition_id", "") or ""),
        question=str(d.get("question", "") or ""),
        active=bool(d.get("active", True)),
        accepting_orders=bool(d.get("accepting_orders", True)),
        closed=bool(d.get("closed", False)),
        is_other=bool(d.get("is_other", False)),
        is_placeholder=bool(d.get("is_placeholder", False)),
        tick_size=float(d.get("tick_size", 0.01) or 0.01),
        min_order_size=float(d.get("min_order_size", 5.0) or 5.0),
        fee=_fee_from_dict(d.get("fee_schedule")),
        market_version=str(d.get("market_version", "v1") or "v1"),
        created_at=d.get("created_at"),
        rules=str(d.get("rules", "") or ""),
    )


def basket_from_dict(d: Mapping[str, Any]) -> Basket:
    synthetic = bool(d.get("synthetic", False))
    legs = tuple(_leg_from_dict(x, synthetic=synthetic) for x in d.get("legs", []) or [])
    status = str(d.get("status", "unresolved"))
    if legs and status == "unresolved":
        status = "active"
    return Basket(
        basket_id=str(d["basket_id"]),
        title=str(d.get("title", d["basket_id"])),
        legs=legs,
        event_slug=d.get("event_slug"),
        event_id=None if d.get("event_id") is None else str(d.get("event_id")),
        category=d.get("category"),
        roles=tuple(d.get("roles", []) or []),
        status=status,
        neg_risk=d.get("neg_risk"),
        neg_risk_augmented=d.get("neg_risk_augmented"),
        neg_risk_market_id=d.get("neg_risk_market_id"),
        convert_fee_bips=float(d.get("convert_fee_bips", 0.0) or 0.0),
        end_date=d.get("end_date"),
        structural_discount_expected=bool(d.get("structural_discount_expected", False)),
        discovered_at=d.get("discovered_at"),
        notes=str(d.get("notes", "") or ""),
        synthetic=synthetic,
        slug_template=d.get("slug_template"),
    )


def _leg_to_dict(l: Leg) -> dict[str, Any]:
    d = asdict(l)
    fee = d.pop("fee")
    d["fee_schedule"] = fee
    return d


def basket_to_dict(b: Basket) -> dict[str, Any]:
    d = {
        "basket_id": b.basket_id,
        "title": b.title,
        "event_slug": b.event_slug,
        "event_id": b.event_id,
        "slug_template": b.slug_template,
        "category": b.category,
        "roles": list(b.roles),
        "status": b.status,
        "neg_risk": b.neg_risk,
        "neg_risk_augmented": b.neg_risk_augmented,
        "neg_risk_market_id": b.neg_risk_market_id,
        "convert_fee_bips": b.convert_fee_bips,
        "end_date": b.end_date,
        "structural_discount_expected": b.structural_discount_expected,
        "discovered_at": b.discovered_at,
        "notes": b.notes,
        "legs": [_leg_to_dict(l) for l in b.legs],
    }
    if b.synthetic:
        d["synthetic"] = True
    return d


def _selector(d: Mapping[str, Any]) -> LegSelector:
    return LegSelector(str(d["basket_id"]), str(d["leg_label_regex"]), float(d.get("coef", 1.0)))


def _defaults_from_dict(d: Mapping[str, Any] | None) -> Defaults:
    d = dict(d or {})
    base = Defaults()
    kw: dict[str, Any] = {}
    for f in base.__dataclass_fields__:
        if f in d:
            kw[f] = d[f]
    return replace(base, **kw)


def config_from_dict(raw: Mapping[str, Any]) -> MarketsConfig:
    if int(raw.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ConfigError(f"unsupported schema_version {raw.get('schema_version')}")
    defaults = _defaults_from_dict(raw.get("defaults"))
    baskets: dict[str, Basket] = {}
    seen_tokens: set[str] = set()
    for bd in raw.get("baskets", []) or []:
        b = basket_from_dict(bd)
        if b.basket_id in baskets:
            raise ConfigError(f"duplicate basket_id {b.basket_id!r}")
        for l in b.legs:
            for tid in (l.yes_token_id, l.no_token_id):
                if tid in seen_tokens:
                    raise ConfigError(f"token id appears twice: {tid}")
                seen_tokens.add(tid)
        if b.legs and b.neg_risk is False:
            warnings.warn(f"basket {b.basket_id}: neg_risk is False - sum-to-one is not guaranteed", stacklevel=2)
        if b.legs and b.neg_risk_augmented and not any(l.is_other for l in b.legs) and not b.structural_discount_expected:
            warnings.warn(f"basket {b.basket_id}: augmented negRisk without an 'Other' leg - expect a structural discount", stacklevel=2)
            b = replace(b, structural_discount_expected=True)
        baskets[b.basket_id] = b
    pairings = tuple(
        Pairing(
            pairing_id=str(p["pairing_id"]),
            type=str(p.get("type", "exploratory")),
            description=str(p.get("description", "")),
            lhs=tuple(_selector(x) for x in p.get("lhs", [])),
            rhs=tuple(_selector(x) for x in p.get("rhs", [])),
            theory_beta=p.get("theory_beta"),
            notes=str(p.get("notes", "") or ""),
        )
        for p in raw.get("pairings", []) or []
    )
    synth = {str(s["basket_id"]): dict(s) for s in raw.get("synthetic_baskets", []) or []}
    for sid in synth:
        if not sid.startswith("SYNTHETIC_"):
            raise ConfigError(f"synthetic basket ids must start with 'SYNTHETIC_': {sid!r}")
        if sid in baskets:
            raise ConfigError(f"synthetic basket id collides with a real basket: {sid!r}")
    return MarketsConfig(
        defaults=defaults, baskets=baskets, pairings=pairings, synthetic_baskets=synth,
        schema_version=SCHEMA_VERSION, updated_at=raw.get("updated_at"),
    )


def config_to_dict(cfg: MarketsConfig) -> dict[str, Any]:
    dd = asdict(cfg.defaults)
    return {
        "schema_version": cfg.schema_version,
        "updated_at": cfg.updated_at,
        "defaults": dd,
        "baskets": [basket_to_dict(b) for b in cfg.baskets.values()],
        "pairings": [
            {
                "pairing_id": p.pairing_id, "type": p.type, "description": p.description,
                "lhs": [asdict(s) for s in p.lhs], "rhs": [asdict(s) for s in p.rhs],
                "theory_beta": p.theory_beta, "notes": p.notes,
            }
            for p in cfg.pairings
        ],
        "synthetic_baskets": [dict(v) for v in cfg.synthetic_baskets.values()],
    }


def load_markets(path: Path | str = CONFIG_PATH) -> MarketsConfig:
    with open(path, encoding="utf-8") as fh:
        return config_from_dict(json.load(fh))


def save_markets(cfg: MarketsConfig, path: Path | str = CONFIG_PATH) -> None:
    """Atomic write (tmp file + ``os.replace``), stable key order, 2-space indent."""
    path = Path(path)
    cfg = replace(cfg, updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    data = json.dumps(config_to_dict(cfg), indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".markets.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def get_basket(basket_id: str, cfg: MarketsConfig | None = None, *, require_resolved: bool = True) -> Basket:
    """Look up a real or synthetic basket. Synthetic baskets are built from their params."""
    cfg = cfg or load_markets()
    if basket_id in cfg.synthetic_baskets:
        from .synthetic import SyntheticParams, synthetic_basket  # lazy: avoids heavy imports

        entry = cfg.synthetic_baskets[basket_id]
        return synthetic_basket(SyntheticParams.from_dict(entry.get("params", {}), name=basket_id, seed=entry.get("seed")))
    if basket_id not in cfg.baskets:
        raise ConfigError(f"unknown basket {basket_id!r}; known: {sorted(cfg.baskets) + sorted(cfg.synthetic_baskets)}")
    b = cfg.baskets[basket_id]
    if require_resolved and not b.is_resolved:
        raise ConfigError(
            f"basket {basket_id!r} has no legs yet (status={b.status}). Run:\n"
            f"  python -m src.discovery add --slug {b.event_slug or '<event-slug>'} --basket-id {basket_id}"
        )
    return b


def upsert_basket(cfg: MarketsConfig, basket: Basket) -> MarketsConfig:
    """Insert/replace a basket, keeping user-maintained fields (roles, notes) when present."""
    baskets = dict(cfg.baskets)
    old = baskets.get(basket.basket_id)
    if old is not None:
        basket = replace(
            basket,
            roles=old.roles or basket.roles,
            notes=old.notes or basket.notes,
            end_date=basket.end_date or old.end_date,
            category=basket.category or old.category,
        )
    baskets[basket.basket_id] = basket
    return replace(cfg, baskets=baskets)


# --------------------------------------------------------------------------- fee resolution
def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def resolve_fee_schedule(
    market: Mapping[str, Any],
    *,
    category: str | None,
    defaults: Defaults,
) -> FeeSchedule:
    """Fee schedule for one market, in order of preference:

    1. Gamma ``feeSchedule {rate, exponent, takerOnly, rebateRate}`` or CLOB ``fd {r, e, to}``;
    2. if ``feesEnabled`` (or unknown) and the market was created on/after
       ``defaults.fees_effective_from``: the category rate from ``defaults``;
    3. otherwise fee-free (markets created before 2026-03-30 are exempt).
    """
    fs = market.get("feeSchedule") or market.get("fee_schedule")
    if isinstance(fs, Mapping) and fs.get("rate") is not None:
        return FeeSchedule(
            rate=float(fs.get("rate") or 0.0),
            exponent=float(fs.get("exponent") or defaults.fee_exponent),
            taker_only=bool(fs.get("takerOnly", fs.get("taker_only", True))),
            rebate_rate=float(fs.get("rebateRate", fs.get("rebate_rate", 0.0)) or 0.0),
            source="gamma_feeSchedule",
        )
    fd = market.get("fd")
    if isinstance(fd, Mapping) and fd.get("r") is not None:
        return FeeSchedule(rate=float(fd["r"]), exponent=float(fd.get("e") or 1.0),
                           taker_only=bool(fd.get("to", True)), source="clob_fd")
    fees_enabled = market.get("feesEnabled", market.get("fees_enabled"))
    if fees_enabled is False:
        return FEE_FREE
    created = _parse_dt(market.get("createdAt") or market.get("created_at"))
    cutoff = _parse_dt(defaults.fees_effective_from + "T00:00:00+00:00")
    if created is not None and cutoff is not None and created < cutoff:
        return FEE_FREE
    if fees_enabled and category:
        rate = defaults.fee_category_rates.get(category.lower())
        if rate is not None:
            return FeeSchedule(rate=float(rate), exponent=defaults.fee_exponent, source="category")
    return FeeSchedule(rate=0.0, source="default")


def basket_fee_per_unit(fees: tuple[FeeSchedule, ...] | list[FeeSchedule], prices: np.ndarray) -> float:
    """Unrounded taker fee for one unit of every leg at ``prices``: ``Σ r_i p_i (1-p_i)``.

    With a common rate and Σp = 1 this equals ``r (1 - Σ p_i^2)`` (one minus the HHI)."""
    total = 0.0
    for f, p in zip(fees, np.asarray(prices, dtype=float)):
        if math.isfinite(p):
            total += f.fee_per_unit(p)
    return total
