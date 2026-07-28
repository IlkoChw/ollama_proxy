"""Tests for the :class:`DashboardClient` SDK.

The client wraps ``httpx.AsyncClient``; we drive the transport with
an ``httpx.MockTransport`` so the suite stays offline. The goal is
to verify that the SDK:

    * issues the right HTTP verb + path for every method,
    * sends ``Authorization: Bearer <token>`` on every request,
    * raises :class:`DashboardClientError` on non-2xx responses,
    * parses 204 (no content) as ``None``,
    * never logs the token (we don't capture logs here, but we
      assert the token isn't in the exception message).
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.services.dashboard_client import DashboardClient, DashboardClientError


class _Recorder:
    """Collects the requests the MockTransport receives."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []


def _build_transport(
    rec: _Recorder,
    routes: dict[tuple[str, str], list[httpx.Response]],
) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        rec.calls.append(request)
        key = (request.method.upper(), request.url.path)
        queue = routes.get(key)
        if not queue:
            return httpx.Response(599, content=f"no route for {key}".encode())
        return queue.pop(0)

    return httpx.MockTransport(handler)


def _make_client(
    rec: _Recorder,
    routes: dict[tuple[str, str], list[httpx.Response]],
    *,
    base_url: str = "http://proxy.test",
    token: str = "tok",
    timeout: float = 2.0,
) -> tuple[DashboardClient, httpx.AsyncClient]:
    """Build a DashboardClient + a real httpx client with a mock transport.

    The injected inner client carries the ``Authorization`` header so
    that the production request path (``_request_json`` → httpx) is
    the path under test. Returning the pair lets tests inspect
    ``rec.calls`` after the context exits.
    """
    transport = _build_transport(rec, routes)
    inner = httpx.AsyncClient(
        transport=transport,
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    c = DashboardClient(base_url, token, timeout=timeout, _client=inner)
    return c, inner


# --------------------------------------------------------------- lifecycle


def test_constructor_rejects_empty_token() -> None:
    with pytest.raises(ValueError):
        DashboardClient("http://x", "")


def test_use_outside_context_raises() -> None:
    c = DashboardClient("http://x", "tok")
    with pytest.raises(RuntimeError):
        # The ``_client`` is None outside ``async with``, so any
        # method that needs the transport will fail-fast.
        c._require()  # type: ignore[attr-defined]


async def test_context_manager_closes_inner_client() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/keys"): [httpx.Response(200, content=b"[]")],
    }
    c, inner = _make_client(rec, routes, token="tok-abc")
    async with c:
        await c.list_keys()
    # ``__aexit__`` closes the inner client and clears the cache.
    assert c._client is None
    # The inner client must be closed: a fresh request on it raises.
    with pytest.raises(RuntimeError):
        await inner.get("/admin/keys")


# --------------------------------------------------------------- auth header


async def test_bearer_token_attached_on_every_request() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/keys"): [httpx.Response(200, content=b"[]")],
    }
    c, _ = _make_client(rec, routes, token="the-secret-token")
    async with c:
        await c.list_keys()
    assert rec.calls, "no requests recorded"
    auth = rec.calls[0].headers.get("Authorization")
    assert auth == "Bearer the-secret-token", (
        f"Authorization header wrong: {rec.calls[0].headers!r}"
    )


# --------------------------------------------------------------- happy paths


async def test_list_keys_parses_json_array() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/keys"): [
            httpx.Response(
                200,
                content=json.dumps(
                    [
                        {"id": 1, "label": "acc1", "status": "active"},
                        {"id": 2, "label": None, "status": "depleted"},
                    ]
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        keys = await c.list_keys()
    assert keys == [
        {"id": 1, "label": "acc1", "status": "active"},
        {"id": 2, "label": None, "status": "depleted"},
    ]


async def test_get_key_parses_dict() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/keys/1"): [
            httpx.Response(
                200,
                content=json.dumps(
                    {"id": 1, "label": "acc1", "status": "active"}
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        body = await c.get_key(1)
    assert body["id"] == 1


async def test_create_key_returns_raw_key() -> None:
    rec = _Recorder()
    routes = {
        ("POST", "/admin/keys"): [
            httpx.Response(
                201,
                content=json.dumps(
                    {
                        "id": 1,
                        "label": "acc1",
                        "raw_key": "sk-real-key",
                        "status": "active",
                    }
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        body = await c.create_key("acc1", "sk-real-key")
    assert body["raw_key"] == "sk-real-key"
    payload = json.loads(rec.calls[0].content)
    assert payload == {"label": "acc1", "key": "sk-real-key"}


async def test_update_key_omits_absent_fields() -> None:
    rec = _Recorder()
    routes = {
        ("PATCH", "/admin/keys/1"): [
            httpx.Response(
                200,
                content=json.dumps({"id": 1, "label": "new", "status": "active"}).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        await c.update_key(1, label="new")
    payload = json.loads(rec.calls[0].content)
    # ``status`` is omitted when not provided.
    assert payload == {"label": "new"}


async def test_delete_key_returns_none_on_204() -> None:
    rec = _Recorder()
    routes = {
        ("DELETE", "/admin/keys/7"): [httpx.Response(204, content=b"")],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        result = await c.delete_key(7)
    assert result is None


async def test_health_returns_dict() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/health"): [
            httpx.Response(
                200,
                content=json.dumps(
                    {
                        "status": "ok",
                        "active_keys": 3,
                        "depleted_keys": 0,
                        "disabled_keys": 0,
                    }
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        h = await c.health()
    assert h["status"] == "ok"
    assert h["active_keys"] == 3


async def test_test_all_keys_returns_dict() -> None:
    rec = _Recorder()
    routes = {
        ("POST", "/admin/keys/test-all"): [
            httpx.Response(
                200,
                content=json.dumps(
                    {"total": 2, "results": [{"id": 1, "ok": True}]}
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        body = await c.test_all_keys()
    assert body["total"] == 2


async def test_reset_states_returns_dict() -> None:
    rec = _Recorder()
    routes = {
        ("POST", "/admin/keys/reset-states"): [
            httpx.Response(
                200,
                content=json.dumps(
                    {
                        "status": "ok",
                        "active_keys": 4,
                        "depleted_keys": 0,
                        "disabled_keys": 0,
                    }
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        body = await c.reset_states()
    assert body["active_keys"] == 4


# --------------------------------------------------------------- error paths


async def test_409_raises_dashboard_client_error_with_detail() -> None:
    rec = _Recorder()
    routes = {
        ("POST", "/admin/keys"): [
            httpx.Response(
                409,
                content=json.dumps(
                    {"detail": "API key with the same value already exists"}
                ).encode(),
            )
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        with pytest.raises(DashboardClientError) as exc_info:
            await c.create_key("acc1", "sk-dup")
    assert exc_info.value.status_code == 409
    assert "already exists" in str(exc_info.value)
    assert "already exists" in exc_info.value.short
    # Token must not appear in the error message.
    assert "tok" not in str(exc_info.value)


async def test_500_raises_with_text_body() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/keys"): [httpx.Response(500, content=b"internal error")],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        with pytest.raises(DashboardClientError) as exc_info:
            await c.list_keys()
    assert exc_info.value.status_code == 500
    assert "internal error" in str(exc_info.value)


async def test_200_with_non_object_non_list_body_raises() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/health"): [httpx.Response(200, content=b'"plain string"')],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        with pytest.raises(DashboardClientError):
            await c.health()


# ----------------------------------------------------------- usage methods


async def test_get_key_usage_happy_path() -> None:
    rec = _Recorder()
    body = {
        "id": 7,
        "label": "acc-7",
        "key_preview": "sk-test-…",
        "account": {"email": "x@example.com", "plan": "free", "fetched_at": "2026-07-27T00:00:00Z"},
        "official": {"session_usage": 1, "weekly_usage": 2, "models": [], "fetched_at": "2026-07-27T00:00:00Z"},
        "local_cumsum": {"session_prompt_tokens": 3, "session_completion_tokens": 4},
        "upstream_status": "ok",
    }
    routes = {
        ("GET", "/admin/keys/7/usage"): [
            httpx.Response(200, content=json.dumps(body).encode())
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        result = await c.get_key_usage(7)
    assert result == body
    assert rec.calls[0].method == "GET"
    assert rec.calls[0].url.path == "/admin/keys/7/usage"


async def test_refresh_key_usage_happy_path() -> None:
    rec = _Recorder()
    body = {"id": 7, "upstream_status": "ok"}
    routes = {
        ("POST", "/admin/keys/7/usage/refresh"): [
            httpx.Response(200, content=json.dumps(body).encode())
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        result = await c.refresh_key_usage(7)
    assert result["upstream_status"] == "ok"


async def test_refresh_key_usage_surfaces_unauthorised_status() -> None:
    """Per the admin endpoint contract, 401 from upstream returns 200
    with ``upstream_status="unauthorised"``; the SDK must not raise."""
    rec = _Recorder()
    body = {"id": 7, "upstream_status": "unauthorised"}
    routes = {
        ("POST", "/admin/keys/7/usage/refresh"): [
            httpx.Response(200, content=json.dumps(body).encode())
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        result = await c.refresh_key_usage(7)
    assert result["upstream_status"] == "unauthorised"


async def test_refresh_all_usage_happy_path() -> None:
    rec = _Recorder()
    body = {"total": 2, "results": [{"id": 1, "upstream_status": "ok"}, {"id": 2, "upstream_status": "ok"}]}
    routes = {
        ("POST", "/admin/keys/usage/refresh-all"): [
            httpx.Response(200, content=json.dumps(body).encode())
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        result = await c.refresh_all_usage()
    assert result["total"] == 2
    assert len(result["results"]) == 2


async def test_get_key_usage_404_raises() -> None:
    rec = _Recorder()
    routes = {
        ("GET", "/admin/keys/999/usage"): [
            httpx.Response(404, content=b'{"detail":"API key not found"}')
        ],
    }
    c, _ = _make_client(rec, routes)
    async with c:
        with pytest.raises(DashboardClientError) as exc_info:
            await c.get_key_usage(999)
    assert exc_info.value.status_code == 404
