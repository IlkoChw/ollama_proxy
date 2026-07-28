"""Tests for the upstream-header denylist in :mod:`app.services.rotation`.

The proxy is meant to be a transparent passthrough: the denylist is
**minimal** and only strips headers that leak infrastructure or runtime
state about the upstream server. These tests assert exactly which
headers are dropped and which are forwarded.
"""

from __future__ import annotations

from app.services.rotation import (
    _BLOCKED_UPSTREAM_HEADERS,
    _TRANSPORT_HEADERS,
    _safe_headers,
)


class _FakeHeaders:
    """Minimal ``httpx.Headers``-like container supporting ``.items()``."""

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def items(self) -> list[tuple[str, str]]:
        return list(self._items)

    # ``httpx.Headers`` also supports iteration, mirroring ``items()``.
    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._items)


def test_blocked_headers_set_matches_documented_policy() -> None:
    """The denylist contains exactly the infrastructure-leaking headers."""
    expected = {
        "server",
        "x-powered-by",
        "x-aspnet-version",
        "x-aspnetmvc-version",
    }
    assert set(_BLOCKED_UPSTREAM_HEADERS) == expected, (
        f"denylist drift: {_BLOCKED_UPSTREAM_HEADERS!r}"
    )


def test_transport_headers_set_matches_documented_policy() -> None:
    """Transport-level headers stay stripped (HTTP/1.1 framing)."""
    assert set(_TRANSPORT_HEADERS) == {
        "content-length",
        "content-encoding",
        "transfer-encoding",
    }


def test_safe_headers_drops_blocked_and_transport_headers() -> None:
    """Headers in the deny- or transport-list must not appear in the output."""
    src = _FakeHeaders(
        [
            ("Server", "nginx/1.25"),
            ("X-Powered-By", "PHP/8.2"),
            ("X-AspNet-Version", "4.0.30319"),
            ("Content-Length", "1234"),
            ("Content-Encoding", "gzip"),
            ("Transfer-Encoding", "chunked"),
        ]
    )
    out = _safe_headers(src)
    assert out == {}, f"expected empty output, got {out!r}"


def test_safe_headers_preserves_useful_response_headers() -> None:
    """Standard response headers must be forwarded verbatim."""
    src = _FakeHeaders(
        [
            ("Content-Type", "application/json"),
            ("x-request-id", "req-abc-123"),
            ("x-ratelimit-remaining", "42"),
            ("retry-after", "60"),
            ("cache-control", "no-store"),
            ("vary", "Accept-Encoding"),
            ("www-authenticate", 'Bearer realm="ollama"'),
        ]
    )
    out = _safe_headers(src)
    expected = {
        "Content-Type": "application/json",
        "x-request-id": "req-abc-123",
        "x-ratelimit-remaining": "42",
        "retry-after": "60",
        "cache-control": "no-store",
        "vary": "Accept-Encoding",
        "www-authenticate": 'Bearer realm="ollama"',
    }
    assert out == expected, f"forwarded headers drift: {out!r}"


def test_safe_headers_is_case_insensitive_on_lookup() -> None:
    """The denylist comparison must not depend on the upstream's casing."""
    src = _FakeHeaders(
        [
            ("SERVER", "nginx"),  # upper-cased
            ("server", "nginx"),  # lower-cased
            ("X-Powered-By", "PHP"),  # mixed-case
        ]
    )
    out = _safe_headers(src)
    assert out == {}


def test_safe_headers_preserves_case_of_kept_headers() -> None:
    """The original casing of forwarded headers is preserved."""
    src = _FakeHeaders(
        [
            ("Content-Type", "text/event-stream"),
            ("X-Request-ID", "r-1"),
        ]
    )
    out = _safe_headers(src)
    assert "Content-Type" in out
    assert "X-Request-ID" in out
    # The denylist comparison lowercases for matching but does NOT
    # rewrite the original casing of the forwarded headers.
    assert out["Content-Type"] == "text/event-stream"
    assert out["X-Request-ID"] == "r-1"


def test_safe_headers_handles_empty_input() -> None:
    """Empty upstream header set yields an empty dict, not an error."""
    assert _safe_headers(_FakeHeaders([])) == {}
