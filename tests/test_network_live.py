"""Live smoke tests against Polymarket. Deselected by default; run with ``pytest -m network``."""
from __future__ import annotations

import asyncio

import pytest

from src.clob_client import ClobRestClient, MarketDataFeed
from src.config import get_basket, load_markets
from src.discovery import GammaClient

pytestmark = pytest.mark.network


def test_gamma_event_and_books():
    b = get_basket("fed-oct-2026", load_markets())
    ev = GammaClient().get_event_by_slug(b.event_slug)
    assert ev.get("negRisk")
    books = ClobRestClient().get_books(list(b.yes_ids))
    assert len(books) == b.n_legs and all("bids" in x for x in books)


def test_websocket_delivers_books():
    b = get_basket("fed-oct-2026", load_markets())

    async def go():
        feed = MarketDataFeed(list(b.yes_ids), rest=ClobRestClient())
        task = asyncio.create_task(feed.run())
        await asyncio.sleep(15)
        clean = all(feed.books.is_clean(a) for a in b.yes_ids)  # before stop(), which marks books dirty
        await feed.stop()
        task.cancel()
        return feed, clean

    feed, clean = asyncio.run(go())
    if feed.stats.connects == 0:
        pytest.skip(f"WebSocket upgrade refused here: {feed.stats.last_error}")
    assert feed.stats.data_frames > 0 and feed.stats.pongs > 0
    assert clean
