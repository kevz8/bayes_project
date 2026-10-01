"""The offline guarantee itself is tested: no test may reach Polymarket (or the proxy)."""
from __future__ import annotations

import os
import socket

import pytest
import requests

from tests.conftest import NetworkBlocked


def test_proxy_env_removed():
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        assert var not in os.environ


def test_live_rest_request_is_blocked():
    with pytest.raises((NetworkBlocked, requests.exceptions.ConnectionError)):
        requests.get("https://clob.polymarket.com/book", params={"token_id": "1"}, timeout=2)


def test_loopback_port_not_opened_by_test_is_blocked(monkeypatch):
    # Even with the proxy variable restored, the proxy's loopback port is not ours.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:39539")
    with pytest.raises((NetworkBlocked, requests.exceptions.ConnectionError)):
        requests.get("https://gamma-api.polymarket.com/events", timeout=2)
    s = socket.socket()
    try:
        with pytest.raises(NetworkBlocked):
            s.connect(("127.0.0.1", 39539))
    finally:
        s.close()


def test_in_process_server_is_allowed():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    cli = socket.socket()
    try:
        cli.connect(("127.0.0.1", port))
        conn, _ = srv.accept()
        conn.close()
    finally:
        cli.close()
        srv.close()
