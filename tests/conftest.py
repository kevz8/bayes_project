"""Shared fixtures. Tests NEVER touch the network (live tests are ``-m network``).

The guard has to be stricter than "allow loopback": in this container the HTTPS egress
proxy itself listens on 127.0.0.1, so a ``requests``/``websockets`` client that honours
``HTTPS_PROXY`` would pass a loopback-only guard and reach the live API. We therefore

1. remove every proxy environment variable for the duration of each test, and
2. allow socket connections only to loopback ports that were opened by a ``listen()``
   call **inside this test process** (i.e. our own fake servers), plus AF_UNIX sockets,
3. refuse DNS resolution of non-local host names.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Callable

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
_PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy",
               "WS_PROXY", "WSS_PROXY", "ws_proxy", "wss_proxy")
_LOCAL_NAMES = {"localhost", "127.0.0.1", "::1", "", None}


class NetworkBlocked(ConnectionError):
    """Raised when a test tries to reach anything but an in-process fake server."""


_ALLOWED_PORTS: set[int] = set()
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_listen = socket.socket.listen
_real_getaddrinfo = socket.getaddrinfo


def _is_loopback(host: Any) -> bool:
    return isinstance(host, str) and (host.startswith("127.") or host in ("::1", "localhost"))


def _check(sock: socket.socket, address: Any) -> None:
    if sock.family == getattr(socket, "AF_UNIX", object()):
        return
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        if _is_loopback(host) and int(port) in _ALLOWED_PORTS:
            return
    raise NetworkBlocked(f"network access blocked in tests: {address!r} (allowed loopback ports: {sorted(_ALLOWED_PORTS)})")


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    _check(self, address)
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    _check(self, address)
    return _real_connect_ex(self, address)


def _recording_listen(self: socket.socket, *args: Any) -> Any:
    result = _real_listen(self, *args)
    try:
        name = self.getsockname()
        if isinstance(name, tuple) and _is_loopback(name[0]):
            _ALLOWED_PORTS.add(int(name[1]))
    except OSError:
        pass
    return result


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    h = host.decode() if isinstance(host, bytes) else host
    if h not in _LOCAL_NAMES and not _is_loopback(h):
        raise NetworkBlocked(f"DNS lookup blocked in tests: {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("network"):
        yield
        return
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    monkeypatch.setattr(socket.socket, "listen", _recording_listen)
    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    try:
        yield
    finally:
        _ALLOWED_PORTS.clear()


# --------------------------------------------------------------------------- helpers
@pytest.fixture
def fixture_json() -> Callable[[str], Any]:
    def load(name: str) -> Any:
        with open(FIXTURES / name, encoding="utf-8") as fh:
            return json.load(fh)

    return load


@pytest.fixture
def fixture_text() -> Callable[[str], str]:
    def load(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")

    return load


class FakeResponse:
    def __init__(self, payload: Any = None, status: int = 200, text: str | None = None):
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)
        self.ok = 200 <= status < 300
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        if self._payload is None and self.text:
            return json.loads(self.text)
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            import requests

            raise requests.HTTPError(f"{self.status_code}", response=self)  # type: ignore[arg-type]


class FakeSession:
    """Tiny ``requests.Session`` stand-in.

    ``routes`` maps ``(METHOD, path_suffix)`` to either a payload, a ``FakeResponse``,
    an exception instance, a callable ``(method, url, params, json) -> payload|FakeResponse``,
    or a list of those (consumed in order; the last one repeats).
    Every call is appended to ``calls`` as ``(method, url, params, json)``.
    """

    def __init__(self, routes: dict[tuple[str, str], Any] | None = None):
        self.routes = dict(routes or {})
        self.calls: list[tuple[str, str, Any, Any]] = []
        self.trust_env = False
        self.headers: dict[str, str] = {}

    def _resolve(self, method: str, url: str, params: Any, json_body: Any) -> FakeResponse:
        self.calls.append((method, url, params, json_body))
        path = url.split("://", 1)[-1]
        path = path[path.find("/"):] if "/" in path else "/"
        match = None
        for (m, suffix), v in self.routes.items():
            if m == method and path.split("?", 1)[0].endswith(suffix):
                if match is None or len(suffix) > len(match[0][1]):
                    match = ((m, suffix), v)
        if match is None:
            return FakeResponse({"error": "not found"}, status=404)
        key, v = match
        if isinstance(v, list):
            item = v.pop(0) if len(v) > 1 else v[0]
        else:
            item = v
        if callable(item) and not isinstance(item, FakeResponse):
            item = item(method, url, params, json_body)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, FakeResponse) else FakeResponse(item)

    def get(self, url: str, params: Any = None, timeout: Any = None, **kw: Any) -> FakeResponse:
        return self._resolve("GET", url, params, None)

    def post(self, url: str, json: Any = None, params: Any = None, timeout: Any = None, **kw: Any) -> FakeResponse:  # noqa: A002
        return self._resolve("POST", url, params, json)

    def request(self, method: str, url: str, params: Any = None, json: Any = None, timeout: Any = None, **kw: Any) -> FakeResponse:  # noqa: A002
        return self._resolve(method.upper(), url, params, json)

    def close(self) -> None:
        pass


@pytest.fixture
def fake_session_factory() -> Callable[..., FakeSession]:
    return FakeSession


@pytest.fixture
def tmp_repo_dirs(tmp_path: Path) -> dict[str, Path]:
    d = {k: tmp_path / k for k in ("historical_books", "prices_history", "synthetic", "results")}
    for p in d.values():
        p.mkdir(parents=True)
    return d
