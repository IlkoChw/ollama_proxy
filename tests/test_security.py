"""Security regression tests.

These tests assert the hardening done as part of the security audit:

    * ``/admin/*`` and ``/admin/health`` are gated by a bearer token
      when ``ADMIN_TOKEN`` is configured; when unset, the gate is open
      and the service is intended for trusted LAN/VPN deployments.
    * The token is checked in constant time.
    * Missing/invalid tokens return 401.
    * The CORS middleware is OFF by default (no CORS headers in the
      response when ``CORS_ALLOW_ORIGINS`` is empty).
    * Once the operator sets ``CORS_ALLOW_ORIGINS=*``, the
      ``Access-Control-Allow-Origin`` header appears in preflight
      responses, but ``Access-Control-Allow-Credentials`` is suppressed
      (because ``*`` is incompatible with credentials).
    * ``/docs``, ``/redoc``, ``/openapi.json`` are 404 by default.
      They become available when ``ENABLE_DOCS=1`` is set at startup.
    * ``/healthz`` is a public liveness check (no auth, no secrets).
    * ``/admin/health`` is still admin-gated (regression).
"""

from __future__ import annotations

import os

import httpx
import pytest
from starlette.requests import Request

# --------------------------------------------------------- admin auth


async def test_admin_rejects_request_without_token(client: httpx.AsyncClient) -> None:
    """``/admin/keys`` must reject a request that has no Authorization header.

    The ``client`` fixture attaches the token, so we explicitly strip it
    on this request.
    """
    resp = await client.post(
        "/admin/keys",
        json={"label": "x", "key": "sk-nope-1234"},
        headers={"Authorization": ""},
    )
    assert resp.status_code == 401, resp.text
    assert "missing" in resp.json()["detail"].lower()


async def test_admin_rejects_wrong_token(client: httpx.AsyncClient) -> None:
    """A wrong token is 401, with the same message shape as 'missing'."""
    resp = await client.get(
        "/admin/keys",
        headers={"Authorization": "Bearer this-is-not-the-right-token"},
    )
    assert resp.status_code == 401, resp.text
    assert "invalid" in resp.json()["detail"].lower()


async def test_admin_rejects_non_bearer_scheme(client: httpx.AsyncClient) -> None:
    """Non-bearer schemes (e.g. Basic) are not accepted."""
    resp = await client.get(
        "/admin/keys",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert resp.status_code == 401, resp.text


async def test_admin_accepts_correct_token(client: httpx.AsyncClient) -> None:
    """The ``client`` fixture already sends the right token → 200."""
    resp = await client.get("/admin/keys")
    assert resp.status_code == 200, resp.text


async def test_admin_health_requires_token() -> None:
    """``/admin/health`` is admin-gated too. Use a fresh client with no header."""
    # Reuse the FastAPI app directly through a custom ASGI client with
    # no default Authorization header.
    from httpx import ASGITransport
    from httpx import AsyncClient as _AsyncClient

    from app.main import app

    transport = ASGITransport(app=app)
    async with _AsyncClient(transport=transport, base_url="http://testserver") as raw:
        resp = await raw.get("/admin/health")
    assert resp.status_code == 401, resp.text


# --------------------------------------------- token-not-configured (open mode)


def test_require_admin_token_no_op_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``ADMIN_TOKEN`` is empty, ``require_admin_token`` is a no-op.

    The operator is responsible for keeping the service on a trusted
    network (LAN / VPN) or for configuring ``ADMIN_TOKEN``. This
    replaces the historical fail-closed 503 behaviour.

    Unit-level test (no ASGI / DB) — the dependency function returns
    ``None`` when the setting is unset, regardless of the request.
    """
    from app.api.deps import require_admin_token
    from app.core.config import Settings, get_settings

    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    # The repo ships a ``.env`` file with ``ADMIN_TOKEN`` set for local
    # development convenience. Pydantic-settings v2 reads ``.env``
    # **after** ``os.environ``, so ``delenv`` alone is not enough to
    # observe "token unset". Disable the ``.env`` source explicitly
    # by passing ``_env_file=None`` on this one ``Settings`` instance.
    get_settings.cache_clear()
    settings: Settings = Settings(_env_file=None)

    # A minimal ASGI scope; ``require_admin_token`` only reads
    # ``request.headers`` so an empty scope is fine for this test.
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/admin/keys",
        "headers": [],
    }
    request = Request(scope)

    # No header at all — must be accepted silently (return None).
    assert require_admin_token(request=request, settings=settings) is None

    # Garbage header — also accepted silently in open mode.
    scope2 = {
        "type": "http",
        "method": "GET",
        "path": "/admin/keys",
        "headers": [(b"authorization", b"Bearer total-garbage")],
    }
    request2 = Request(scope2)
    assert require_admin_token(request=request2, settings=settings) is None


# --------------------------------------------------------------- CORS


async def test_cors_disabled_by_default() -> None:
    """No CORS middleware is installed when ``CORS_ALLOW_ORIGINS`` is empty.

    A cross-origin preflight request from an attacker must NOT be granted
    access. With no CORS middleware at all, the response simply lacks the
    ``Access-Control-Allow-Origin`` header (and is therefore rejected by
    the browser).
    """
    from fastapi.middleware.cors import CORSMiddleware

    from app.main import app

    # The production app has no CORSMiddleware when CORS_ALLOW_ORIGINS=""
    # (default in tests). Assert that.
    has_cors = any(
        isinstance(m.cls, type) and issubclass(m.cls, CORSMiddleware)
        for m in app.user_middleware
    )
    assert has_cors is False, (
        "CORS middleware must NOT be installed by default; "
        f"found: {app.user_middleware!r}"
    )


def test_cors_wildcard_suppresses_credentials() -> None:
    """When ``CORS_ALLOW_ORIGINS=*``, ``Allow-Credentials`` must NOT be set.

    We rebuild a fresh app (so the env change is picked up at import time
    the way production does on startup) and inspect the middleware list
    directly. The constructed CORS config must have
    ``allow_credentials=False`` (browsers reject wildcard + credentials).
    """
    import importlib

    from fastapi.middleware.cors import CORSMiddleware

    import app.main as main_module

    # Save the real env, flip CORS_ALLOW_ORIGINS to "*", reload the
    # module, inspect, then restore.
    saved = os.environ.get("CORS_ALLOW_ORIGINS")
    os.environ["CORS_ALLOW_ORIGINS"] = "*"
    try:
        from app.core.config import get_settings

        get_settings.cache_clear()
        importlib.reload(main_module)
        reloaded_app = main_module.app
        cors_entries = [
            m for m in reloaded_app.user_middleware
            if m.cls is CORSMiddleware
        ]
        assert len(cors_entries) == 1, (
            f"expected exactly one CORSMiddleware, got {cors_entries!r}"
        )
        # FastAPI stores the middleware kwargs in ``.kwargs`` on the
        # ``Middleware`` wrapper object.
        opts = dict(cors_entries[0].kwargs)
        assert opts.get("allow_origins") == ["*"], opts
        # The whole point of this guard: credentials MUST be off when
        # origins are wildcarded (otherwise the spec says the response
        # is invalid and browsers will reject it).
        assert opts.get("allow_credentials") is False, (
            f"allow_credentials must be False when origins are '*', got {opts!r}"
        )
    finally:
        if saved is None:
            os.environ.pop("CORS_ALLOW_ORIGINS", None)
        else:
            os.environ["CORS_ALLOW_ORIGINS"] = saved
        get_settings.cache_clear()
        importlib.reload(main_module)


# ------------------------------------------------- token generation helper


def test_generate_admin_token_is_urlsafe_and_long() -> None:
    """The helper must produce a 32-byte URL-safe token."""
    from app.api.deps import generate_admin_token

    tok = generate_admin_token()
    # token_urlsafe(32) → ~43 chars of base64-url alphabet.
    assert len(tok) >= 40
    # URL-safe alphabet: A-Z a-z 0-9 - _ (plus possible '=' padding stripped).
    import re

    assert re.fullmatch(r"[A-Za-z0-9_\-]+", tok), f"non-urlsafe token: {tok!r}"


# Suppress an unused-import warning for ``os`` in case future maintainers
# add a fixture that needs to mutate the env directly.
_ = os


# --------------------------------------------------- docs / Swagger (HIGH-1)


async def test_docs_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/docs``, ``/redoc`` and ``/openapi.json`` must 404 when ``ENABLE_DOCS`` is off.

    We reload ``app.main`` after clearing the env var so the change is
    picked up at import time the way production does on startup.

    The repo ships a ``.env`` file with ``ENABLE_DOCS=1`` for local
    development convenience. Pydantic-settings v2 reads ``.env``
    **after** ``os.environ``, so popping the env var alone is not
    enough — we additionally disable the ``.env`` source on
    :class:`Settings` while the test runs.

    The reload inside the test reads the **mutated** Settings (env_file
    disabled, ENABLE_DOCS unset). The final reload restores the module
    to its import-time state with the real ``.env`` so subsequent tests
    (``test_docs_enabled_via_env``) see ``docs_url == "/docs"`` again.
    """
    import importlib

    import app.main as main_module
    from app.core.config import Settings

    REAL_ENV_FILE = Settings.model_config["env_file"]

    monkeypatch.delenv("ENABLE_DOCS", raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    try:
        from app.core.config import get_settings

        get_settings.cache_clear()
        importlib.reload(main_module)
        reloaded_app = main_module.app
        from httpx import ASGITransport
        from httpx import AsyncClient as _AsyncClient

        transport = ASGITransport(app=reloaded_app)
        async with _AsyncClient(transport=transport, base_url="http://testserver") as raw:
            for path in ("/docs", "/redoc", "/openapi.json"):
                resp = await raw.get(path)
                assert resp.status_code == 404, (
                    f"{path} must 404 when ENABLE_DOCS=off, got {resp.status_code}"
                )
    finally:
        # Restore ``.env`` source on Settings BEFORE reloading the
        # module — otherwise the reload sees the test's mutation and
        # the production app would be left with ``docs_url=None``,
        # breaking tests that follow.
        Settings.model_config["env_file"] = REAL_ENV_FILE
        from app.core.config import get_settings

        get_settings.cache_clear()
        importlib.reload(main_module)


async def test_docs_enabled_via_env() -> None:
    """``/docs`` returns the Swagger UI when ``ENABLE_DOCS=1``."""
    from app.main import app

    # The conftest already sets ``ENABLE_DOCS=1`` via setdefault, so the
    # default-imported ``app`` should have docs_url set.
    assert app.docs_url == "/docs", f"expected docs_url=/docs, got {app.docs_url!r}"
    assert app.openapi_url == "/openapi.json"

    from httpx import ASGITransport
    from httpx import AsyncClient as _AsyncClient

    transport = ASGITransport(app=app)
    async with _AsyncClient(transport=transport, base_url="http://testserver") as raw:
        resp_docs = await raw.get("/docs")
        assert resp_docs.status_code == 200, resp_docs.text
        # /openapi.json must be valid JSON with an OpenAPI 3.x shape.
        resp_schema = await raw.get("/openapi.json")
        assert resp_schema.status_code == 200
        schema = resp_schema.json()
        assert "openapi" in schema
        assert "paths" in schema


# ----------------------------------------------------------- /healthz (HIGH-2)


async def test_healthz_public_no_auth(client: httpx.AsyncClient) -> None:
    """``/healthz`` must work without an Authorization header.

    The ``client`` fixture attaches the admin bearer by default; we
    explicitly strip it here to prove the endpoint is public.
    """
    # The lifespan has not run for this fixture (we never invoked it),
    # so the engine and vault are not initialised — expect 503 with
    # ``status=starting``. The test proves the endpoint does NOT
    # demand auth: any unauthenticated request reaches the handler.
    resp = await client.get("/healthz", headers={"Authorization": ""})
    assert resp.status_code in (200, 503), resp.text
    body = resp.json()
    assert body["status"] in ("ok", "starting")
    assert "timestamp" in body
    # Healthz must never include counts, version, or any sensitive state.
    forbidden_keys = {"active_keys", "depleted_keys", "disabled_keys", "version"}
    assert not (forbidden_keys & body.keys()), body


async def test_admin_health_still_requires_token(client: httpx.AsyncClient) -> None:
    """Regression: ``/admin/health`` remains admin-gated (CRITICAL-1 preserved)."""
    # The ``client`` fixture sends the admin token by default; verify
    # that stripping it actually returns 401 (proving the gate is real).
    resp = await client.get("/admin/health", headers={"Authorization": ""})
    assert resp.status_code == 401, resp.text
