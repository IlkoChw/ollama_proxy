"""Tests for the proxy endpoints: /v1/models, /v1/chat/completions, /api/tags."""

from __future__ import annotations

import ssl

import httpx

# ----------------------------------------------------------------- /v1/models


async def test_v1_models_static_fallback_when_no_keys(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient
) -> None:
    resp = await user_client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "minimax-m2.7" in ids


async def test_v1_models_returns_upstream_via_key(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-v1models-1"})
    mock_router.add(
        "GET",
        "/v1/models",
        status=200,
        json_body={
            "object": "list",
            "data": [
                {"id": "minimax-m3", "object": "model", "created": 0, "owned_by": "ollama"},
                {"id": "minimax-m2.7", "object": "model", "created": 0, "owned_by": "ollama"},
            ],
        },
    )
    resp = await user_client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    ids = sorted(m["id"] for m in body["data"])
    assert ids == ["minimax-m2.7", "minimax-m3"]


async def test_v1_models_falls_back_to_static_on_upstream_failure(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-5xx-v1models"})
    mock_router.add("GET", "/v1/models", status=503, text="upstream down")
    resp = await user_client.get("/v1/models")
    assert resp.status_code == 200
    ids = [m["id"] for m in resp.json()["data"]]
    assert "minimax-m2.7" in ids  # static last-resort


async def test_v1_models_caches_subsequent_calls(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-cache-1234"})
    # Register ONLY ONE response — if the cache works, the second call
    # should not hit the upstream.
    mock_router.add(
        "GET",
        "/v1/models",
        status=200,
        json_body={
            "object": "list",
            "data": [{"id": "minimax-m3", "object": "model"}],
        },
    )
    r1 = await user_client.get("/v1/models")
    r2 = await user_client.get("/v1/models")
    assert r1.status_code == r2.status_code == 200
    assert [m["id"] for m in r1.json()["data"]] == ["minimax-m3"]
    assert [m["id"] for m in r2.json()["data"]] == ["minimax-m3"]
    # Only one upstream call should have been made.
    upstream_calls = [c for c in mock_router.calls if c[1] == "/v1/models"]
    assert len(upstream_calls) == 1


async def test_v1_models_does_not_leak_in_flight_reservation(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """Regression: ``/v1/models`` cold path (``_fetch_models_via_rotation``)
    must release its ``KeyManager`` reservation on every exit path. Without
    the release, repeated cold calls would leak in-flight slots until the
    rotation pool saturates and ``/v1/chat/completions`` starts returning
    503.

    Asserts that after the request completes (success or upstream error),
    the process-wide ``KeyManager._in_flight`` set is empty.
    """
    from app.services import key_manager as km_module

    await client.post("/admin/keys", json={"label": "m1", "key": "sk-models-aaa"})
    mock_router.add(
        "GET",
        "/v1/models",
        status=200,
        json_body={
            "object": "list",
            "data": [{"id": "minimax-m3", "object": "model"}],
        },
    )

    km = km_module.get_key_manager()
    # Trigger the cold path (cache empty → ``_fetch_models_via_rotation`` runs).
    resp = await user_client.get("/v1/models")
    assert resp.status_code == 200

    # No leaked reservation. Without the ``finally`` in
    # ``_fetch_models_via_rotation`` this would be 1.
    assert km.in_flight_count() == 0


async def test_v1_models_releases_reservation_on_upstream_error(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """Anti-stacking release must run even when the upstream errors out —
    a leaked reservation from an error path would be invisible to tests
    that only exercise the happy path."""
    from app.services import key_manager as km_module

    await client.post("/admin/keys", json={"label": "err1", "key": "sk-models-err"})
    mock_router.add("GET", "/v1/models", status=503, text="upstream down")

    km = km_module.get_key_manager()
    resp = await user_client.get("/v1/models")
    # The endpoint itself succeeds (returns the static fallback), but
    # ``_fetch_models_via_rotation`` exited via the error path.
    assert resp.status_code == 200

    assert km.in_flight_count() == 0


# -------------------------------------------------------------- /api/tags


async def test_api_tags_static_fallback_when_no_keys(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient
) -> None:
    resp = await user_client.get("/api/tags")
    assert resp.status_code == 200
    body = resp.json()
    names = [m["name"] for m in body["models"]]
    assert "minimax-m2.7" in names


async def test_api_tags_proxies_through_active_key(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-tags-1"})
    mock_router.add(
        "GET",
        "/v1/models",  # list_tags goes through /v1/models now (not /api/tags)
        status=200,
        json_body={
            "object": "list",
            "data": [{"id": "minimax-m3", "object": "model"}],
        },
    )
    resp = await user_client.get("/api/tags")
    assert resp.status_code == 200
    body = resp.json()
    names = [m["name"] for m in body["models"]]
    assert "minimax-m3" in names


async def test_api_tags_falls_back_on_upstream_failure(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-5xx-tags-1"})
    mock_router.add("GET", "/v1/models", status=503, text="upstream down")

    resp = await user_client.get("/api/tags")
    assert resp.status_code == 200
    body = resp.json()
    assert any(m["name"] == "minimax-m2.7" for m in body["models"])


# ------------------------------------------------------ chat completions


async def test_chat_completions_passthrough(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-pass-1111111"})

    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
        },
    )

    resp = await user_client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "hi"


async def test_chat_completions_rotates_on_429(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    await client.post("/admin/keys", json={"label": "first", "key": "sk-rot-1111111"})
    await client.post("/admin/keys", json={"label": "second", "key": "sk-rot-2222222"})

    # First key → 429; second key → 200.
    mock_router.add("POST", "/v1/chat/completions", status=429, text="rate limited")
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={
            "id": "chatcmpl-2",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}}
            ],
        },
    )

    resp = await user_client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "ok"

    # First key now in 'depleted' state.
    listing = await client.get("/admin/keys")
    statuses = {item["label"]: item["status"] for item in listing.json()}
    assert statuses["first"] == "depleted"
    assert statuses["second"] == "active"


async def test_chat_completions_404_does_not_disable_key(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """A 404 (model unavailable) keeps the key active and rotates."""
    await client.post("/admin/keys", json={"label": "first", "key": "sk-404-1111111"})
    await client.post("/admin/keys", json={"label": "second", "key": "sk-404-2222222"})

    # First key: 404 (model unavailable); second key: 200.
    mock_router.add("POST", "/v1/chat/completions", status=404, text="model not found")
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={
            "id": "chatcmpl-3",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}}
            ],
        },
    )

    resp = await user_client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m3",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "ok"

    # Both keys must still be 'active' — 404 does NOT disable.
    listing = await client.get("/admin/keys")
    statuses = {item["label"]: item["status"] for item in listing.json()}
    assert statuses["first"] == "active"
    assert statuses["second"] == "active"


async def test_chat_completions_404_on_all_keys_returns_404(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """If every key returns 404, the proxy itself returns 404 (not 503)."""
    await client.post("/admin/keys", json={"label": "a1", "key": "sk-404-a-1111"})
    await client.post("/admin/keys", json={"label": "a2", "key": "sk-404-a-2222"})

    # Both keys → 404.
    mock_router.add("POST", "/v1/chat/completions", status=404, text="model not found")
    mock_router.add("POST", "/v1/chat/completions", status=404, text="model not found")

    resp = await user_client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m3",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 404
    body = resp.json()
    assert "not available" in body["detail"].lower()

    # Both keys must still be 'active' — no disable, no depletion.
    listing = await client.get("/admin/keys")
    statuses = {item["label"]: item["status"] for item in listing.json()}
    assert statuses["a1"] == "active"
    assert statuses["a2"] == "active"


async def test_chat_completions_401_still_disables_key(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """Regression: 401 (dead key) must still disable, while 404 does not."""
    await client.post("/admin/keys", json={"label": "first", "key": "sk-401-1111111"})
    await client.post("/admin/keys", json={"label": "second", "key": "sk-401-2222222"})

    # First key → 401 (dead); second key → 200.
    mock_router.add("POST", "/v1/chat/completions", status=401, text="unauthorized")
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={
            "id": "x",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        },
    )

    resp = await user_client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200

    listing = await client.get("/admin/keys")
    statuses = {item["label"]: item["status"] for item in listing.json()}
    assert statuses["first"] == "disabled"
    assert statuses["second"] == "active"


async def test_chat_completions_no_keys_returns_503(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient
) -> None:
    resp = await user_client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m2.7",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 503


async def test_chat_completions_missing_fields_returns_422(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient
) -> None:
    resp = await user_client.post(
        "/v1/chat/completions",
        json={"model": "minimax-m2.7"},  # missing messages
    )
    assert resp.status_code == 422


# ----------------------------------------- concurrent rotation (regression)


async def test_chat_completions_50_concurrent_with_5_valid_keys(
    client: httpx.AsyncClient,
    user_client: httpx.AsyncClient,
    mock_router,
) -> None:
    """End-to-end: 50 concurrent ``/v1/chat/completions`` requests against
    a pool of 5 active keys whose upstream always returns 200.

    Regression for the 2026-07-27 stress-test report: with the old
    pipeline 50 concurrent requests against 18 keys hit saturation
    (anti-stacking says "no active key" once ``len(_in_flight) ==
    pool_size``). The proxy must either (a) release the reservation
    after each request completes or (b) queue — but it must NEVER
    surface ``503 no active keys configured`` when there ARE active
    keys whose requests simply completed first.

    With ``MockRouter.set_default`` every upstream call returns 200
    immediately, so each request's "in-flight" reservation lasts
    only for the duration of the await. A correct implementation
    must serve all 50 with 200; a regression that leaks the
    reservation would cause 503s once the pool saturates.
    """
    import asyncio

    # Seed 5 valid keys via the admin API.
    for i in range(5):
        await client.post(
            "/admin/keys",
            json={"label": f"k{i}", "key": f"sk-pool-{i:0>16}"},
        )

    # Every upstream POST /v1/chat/completions → 200.
    mock_router.set_default(
        httpx.Response(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b'{"id":"chatcmpl","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        )
    )

    # Fire 50 concurrent requests through the same user-client.
    async def _one() -> int:
        r = await user_client.post(
            "/v1/chat/completions",
            json={
                "model": "minimax-m2.7",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        return r.status_code

    statuses = await asyncio.wait_for(
        asyncio.gather(*(_one() for _ in range(50))),
        timeout=10.0,
    )

    # No 503, no 5xx. Anti-stacking must release promptly.
    n_200 = sum(1 for s in statuses if s == 200)
    n_503 = sum(1 for s in statuses if s == 503)
    n_other = sum(1 for s in statuses if s not in (200, 503))

    assert n_200 == 50, (
        f"expected all 50 to succeed; got 200={n_200}, 503={n_503}, "
        f"other={n_other}, full={statuses}"
    )
    assert n_503 == 0, (
        f"anti-stacking saturation surfaced 503 on {n_503} of 50; "
        "this means reservations are leaked after the upstream call "
        "completes (regression of /v1/models fix)."
    )
    assert n_other == 0, f"unexpected statuses: {statuses}"

    # Pool must still hold 5 active keys (no spurious flip to disabled/depleted).
    listing = await client.get("/admin/keys")
    statuses_db = {item["label"]: item["status"] for item in listing.json()}
    assert all(s == "active" for s in statuses_db.values()), statuses_db
    # Every key got hit at least once (50 reqs on 5 keys, round-robin).
    total_reqs = sum(item["total_requests"] for item in listing.json())
    per_key = {item["label"]: item["total_requests"] for item in listing.json()}
    assert total_reqs == 50, f"per_key={per_key}"


# ----------------------------------------- low-level transport errors (regression)


async def test_ollama_client_wraps_ssl_error_as_request_error() -> None:
    """``OllamaClient.chat_completion`` must wrap raw ``ssl.SSLError``.

    Regression for: 2026-07-26 ``SSLV3_ALERT_BAD_RECORD_MAC`` in logs →
    client got ``500 Internal Server Error`` because
    ``httpcore 1.x`` lets ``ssl.SSLError`` bubble up unwrapped
    (it's a raw ``OSError`` subclass, not an ``httpx.RequestError``).

    The fix: ``OllamaClient`` now catches ``(ssl.SSLError, OSError)``
    and re-raises as ``httpx.RequestError`` so the caller's existing
    transport-error branch handles it.

    This is a focused unit test against ``OllamaClient.chat_completion``
    using an in-memory stub client. It does NOT use ``MockRouter``
    because ``httpx.MockTransport`` swallows handler exceptions and
    returns a 500 — we need the exception to actually propagate.
    """
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.services.ollama_client import OllamaClient

    # Stub client whose ``post`` always raises a raw ssl.SSLError,
    # matching what httpcore surfaces when a TLS record is corrupted.
    fake_client = AsyncMock()
    fake_client.post.side_effect = ssl.SSLError(
        "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] ssl/tls alert bad record mac"
    )

    settings = Settings(ollama_base_url="https://ollama.test")
    client = OllamaClient(http_client=fake_client, settings=settings)

    import pytest as _pytest

    with _pytest.raises(httpx.RequestError) as excinfo:
        await client.chat_completion(
            key="sk-test", key_preview="sk-test…", body={"model": "x", "stream": False}
        )
    # The original ssl.SSLError must be the cause (chained via 'from').
    assert isinstance(excinfo.value.__cause__, ssl.SSLError)
    assert "SSLV3_ALERT_BAD_RECORD_MAC" in str(excinfo.value.__cause__)


async def test_ollama_client_wraps_connection_reset_as_request_error() -> None:
    """``OllamaClient.chat_completion`` must wrap ``ConnectionResetError`` too.

    ``ConnectionResetError`` is a sibling subclass of ``ssl.SSLError``
    (both are ``OSError`` subclasses). Same fix applies: wrap as
    ``httpx.RequestError`` so rotation treats it as transient.
    """
    from unittest.mock import AsyncMock

    from app.core.config import Settings
    from app.services.ollama_client import OllamaClient

    fake_client = AsyncMock()
    fake_client.post.side_effect = ConnectionResetError("connection reset by peer")

    settings = Settings(ollama_base_url="https://ollama.test")
    client = OllamaClient(http_client=fake_client, settings=settings)

    import pytest as _pytest

    with _pytest.raises(httpx.RequestError) as excinfo:
        await client.chat_completion(
            key="sk-test", key_preview="sk-test…", body={"model": "x", "stream": False}
        )
    assert isinstance(excinfo.value.__cause__, ConnectionResetError)
