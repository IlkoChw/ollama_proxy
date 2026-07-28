"""Integration tests for the in-process dashboard.

When ``ENABLE_DASHBOARD=1`` the dashboard lives in the same ASGI app
as the proxy. Tests boot the FastAPI app, run the lifespan with
mocked dependencies, and exercise the dashboard routes through an
``httpx.AsyncClient`` wired to the ASGI transport. State assertions
hit the in-memory SQLite the lifespan creates, so CRUD / health /
status-filter tests no longer need a separate ``MockProxy`` — they
read back what the dashboard routes just wrote.

HTTP round-trips that *do* exist (the ``/admin/keys/{id}/test`` and
``/admin/keys/{id}/usage*`` probes) still flow through the lifespan
``httpx.AsyncClient``, which the test fixture backs with a
``MockTransport``. That keeps the ``UsageService`` / ``OllamaClient``
calls scriptable without touching the real ollama.com.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest_asyncio
from cryptography.fernet import Fernet
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Set env *before* importing the app so the lifespan accepts the boot.
# ``ENABLE_DASHBOARD=1`` is what mounts the dashboard router in
# ``app.main``. The proxy env vars are managed by ``conftest.py``; we
# only add the dashboard-specific ones here so this file remains
# runnable in isolation (e.g. ``pytest tests/test_dashboard_routes.py``).
os.environ.setdefault("ENABLE_DASHBOARD", "1")
os.environ.setdefault("DASHBOARD_PASSWORD", "test-dashboard-pass")
os.environ.setdefault("DASHBOARD_SESSION_SECRET", "x" * 64)
os.environ.setdefault("DASHBOARD_COOKIE_SECURE", "0")

from app import models as _models  # noqa: F401, E402  # side-effect: ORM registry
from app.core.config import get_settings  # noqa: E402
from app.db import session as session_module  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.main import app as proxy_app  # noqa: E402
from app.services import key_manager as key_manager_module  # noqa: E402
from app.services import vault as vault_module  # noqa: E402
from app.services.dashboard_auth import (  # noqa: E402
    CSRF_COOKIE,
    FLASH_COOKIE,
    SESSION_COOKIE,
    DashboardAuth,
    set_dashboard_auth,
)
from app.services.usage_service import UsageService  # noqa: E402

AUTH_SECRET = "x" * 64
DASHBOARD_PASSWORD = "test-dashboard-pass"


# --------------------------------------------------------------- helpers


def _cookie_value(jar: httpx.Cookies, name: str) -> str | None:
    """Return the value of cookie ``name`` from the response jar, or ``None``.

    ``httpx.Cookies`` is a dict-like wrapper around the response
    ``Set-Cookie`` headers. ``jar.get(name)`` returns the value
    directly (str or None), not a Cookie object.
    """
    return jar.get(name)


async def _login(client: AsyncClient) -> dict[str, str]:
    """Log the client in. Returns ``{session, csrf}`` cookies."""
    resp = await client.post(
        "/dashboard/login",
        data={"password": DASHBOARD_PASSWORD},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text
    session = _cookie_value(client.cookies, SESSION_COOKIE)
    csrf = _cookie_value(client.cookies, CSRF_COOKIE)
    assert session, "session cookie was not set"
    assert csrf, "csrf cookie was not set"
    return {"session": session or "", "csrf": csrf or ""}


async def _create_key(
    client: AsyncClient,
    *,
    label: str,
    raw: str,
    cookies: dict[str, str],
) -> dict[str, Any]:
    """Create an API key via the dashboard route. Returns the API JSON."""
    resp = await client.post(
        "/dashboard/keys",
        data={"label": label, "key": raw, "csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303, resp.text
    return {}


# --------------------------------------------------------------- fixtures


@pytest_asyncio.fixture
async def dashboard_state() -> Any:
    """Yield a context-manager bundle: in-memory engine, mock router, etc.

    Tests use this directly when they need to seed data (e.g. status
    filter scenarios). The ``dashboard_client`` fixture below
    composes it into an ``httpx.AsyncClient`` that drives the
    FastAPI app in-process.
    """
    # Reset the cached settings so the env we set at the top of this
    # file actually wins.
    get_settings.cache_clear()

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    # Each test gets a fresh ephemeral vault.
    vault_module.set_vault(vault_module.Vault(fernet=Fernet(Fernet.generate_key())))
    key_manager_module.reset_key_manager_for_tests()
    session_module.override_session_factory(factory)

    class Bundle:
        pass

    bundle = Bundle()
    bundle.engine = engine
    bundle.factory = factory
    yield bundle

    session_module.reset_for_tests()
    await engine.dispose()


@pytest_asyncio.fixture
async def mock_router() -> Any:
    """Lightweight httpx mock router used by the in-process probes."""

    class MockRouter:
        def __init__(self) -> None:
            self.routes: dict[tuple[str, str], list[httpx.Response]] = {}
            self.default: httpx.Response | None = None

        def add(
            self,
            method: str,
            path: str,
            *,
            status: int = 200,
            json_body: Any = None,
            text: str | None = None,
            headers: dict[str, str] | None = None,
        ) -> None:
            if text is not None:
                body = text.encode("utf-8")
            elif json_body is not None:
                import json as _json

                body = _json.dumps(json_body).encode("utf-8")
            else:
                body = b""
            resp = httpx.Response(
                status_code=status, headers=headers or {}, content=body
            )
            self.routes.setdefault((method.upper(), path), []).append(resp)

        async def handle(self, request: httpx.Request) -> httpx.Response:
            queue = self.routes.get((request.method.upper(), request.url.path))
            if queue:
                return queue.pop(0)
            if self.default is not None:
                return self.default
            return httpx.Response(
                599, content=f"no mock for {request.method} {request.url.path}".encode()
            )

    return MockRouter()


@pytest_asyncio.fixture
async def dashboard_client(
    dashboard_state: Any, mock_router: Any
) -> AsyncClient:
    """Yield an ``httpx.AsyncClient`` driving the FastAPI app in-process.

    Wires the lifespan-shaped state onto ``app.state`` (engine, mock
    http_client, usage_service) and exposes the dashboard routes via
    the ASGI transport. ``DASHBOARD_PASSWORD`` /
    ``DASHBOARD_SESSION_SECRET`` are already in the env from the
    module-level ``os.environ.setdefault`` calls.
    """
    # Build the mock-backed httpx client that ``OllamaClient`` and
    # ``UsageService`` will use for upstream probes.
    transport = httpx.MockTransport(mock_router.handle)
    state_http = httpx.AsyncClient(transport=transport, base_url="https://ollama.test")

    # Usage service: lifetime is bound to the test engine via the
    # factory. The lifespan constructs one against ``app.state.engine``;
    # here we hand the same object to ``UsageService`` so the periodic
    # /api/usage refresh task would see the same data. We bypass the
    # lifespan entirely.
    vault = vault_module.get_vault()
    usage_service = UsageService(
        ollama=_ollama_via(state_http),
        vault=vault,
        ttl_seconds=300,
    )

    # Set the same ``app.state`` attributes the real lifespan does.
    state = proxy_app.state
    state.engine = dashboard_state.engine
    state.http_client = state_http
    state.model_cache = None
    state.usage_service = usage_service
    state.templates = _build_templates()

    # The real lifespan installs the DashboardAuth singleton; we
    # replicate that here so routes can resolve it.
    set_dashboard_auth(
        DashboardAuth(
            secret=AUTH_SECRET,
            session_ttl_seconds=3600,
            cookie_secure=False,
        )
    )

    transport_asgi = ASGITransport(app=proxy_app)
    try:
        async with AsyncClient(
            transport=transport_asgi, base_url="http://testserver"
        ) as ac:
            yield ac
    finally:
        await state_http.aclose()
        set_dashboard_auth(None)


def _ollama_via(http_client: httpx.AsyncClient) -> Any:
    """Build an :class:`OllamaClient` bound to ``http_client`` + current settings."""
    from app.services.ollama_client import OllamaClient

    return OllamaClient(http_client, get_settings())


def _build_templates() -> Any:
    """Construct a ``Jinja2Templates`` against the project's ``templates/`` dir."""
    from pathlib import Path

    from fastapi.templating import Jinja2Templates

    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    templates.env.auto_reload = False  # tests shouldn't re-read templates on disk
    from app.core.dashboard_filters import register_template_filters

    register_template_filters(templates)
    return templates


# --------------------------------------------------------------- seed helpers


async def _seed_keys_via_dashboard(
    client: AsyncClient,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    """Create each row through ``POST /dashboard/keys`` and return login cookies.

    After every successful ``POST`` the CSRF cookie is rotated. We
    re-login between rows so the next submission carries a fresh
    ``csrf_token`` matching the cookie httpx keeps on the jar.
    """
    cookies = await _login(client)
    for row in rows:
        # Re-login to refresh the CSRF pair; the previous POST rotated
        # the token.
        cookies = await _login(client)
        resp = await client.post(
            "/dashboard/keys",
            data={
                "label": row["label"],
                "key": row["key"],
                "csrf_token": cookies["csrf"],
            },
            follow_redirects=False,
        )
        assert resp.status_code == 303, resp.text
    return cookies


async def _set_key_status(
    factory: Any, key_id: int, status: str, **extra: Any
) -> None:
    """Override a key's status (and any extra columns) directly in the DB.

    Bypasses the dashboard route so tests can put rows into
    ``depleted`` / ``disabled`` without going through upstream.
    """
    from datetime import UTC, datetime

    from sqlalchemy import update as _update

    from app.models.api_key import ApiKey

    now = datetime.now(UTC)
    values: dict[str, Any] = {"status": status, "updated_at": now}
    if "cooldown_until" in extra:
        values["cooldown_until"] = extra["cooldown_until"]
    if "last_status_code" in extra:
        values["last_status_code"] = extra["last_status_code"]
    if "total_requests" in extra:
        values["total_requests"] = extra["total_requests"]
    if "total_failures" in extra:
        values["total_failures"] = extra["total_failures"]
    if "last_used_at" in extra:
        values["last_used_at"] = extra["last_used_at"]
    if "session_usage_fraction" in extra:
        values["last_usage_session_fraction"] = extra["session_usage_fraction"]
    if "weekly_usage_fraction" in extra:
        values["last_usage_weekly_fraction"] = extra["weekly_usage_fraction"]
    if "last_usage_fetch_at" in extra:
        values["last_usage_fetch_at"] = extra["last_usage_fetch_at"]
    async with factory() as session:
        await session.execute(
            _update(ApiKey).where(ApiKey.id == key_id).values(**values)
        )
        await session.commit()


# --------------------------------------------------------------- tests


async def test_root_redirects_to_login_when_unauthed(
    dashboard_client: AsyncClient,
) -> None:
    resp = await dashboard_client.get("/dashboard", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard/login"


async def test_login_form_is_public(dashboard_client: AsyncClient) -> None:
    resp = await dashboard_client.get("/dashboard/login")
    assert resp.status_code == 200
    body = resp.text.lower()
    assert "password" in body
    assert 'action="/dashboard/login"' in body


async def test_login_with_wrong_password_sets_error_flash(
    dashboard_client: AsyncClient,
) -> None:
    resp = await dashboard_client.post(
        "/dashboard/login",
        data={"password": "wrong"},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    # Follow the redirect to /dashboard/login and read the flash.
    resp2 = await dashboard_client.get("/dashboard/login")
    assert resp2.status_code == 200
    assert "invalid password" in resp2.text.lower()


async def test_login_with_correct_password_sets_cookies(
    dashboard_client: AsyncClient,
) -> None:
    cookies = await _login(dashboard_client)
    assert cookies["session"]
    assert cookies["csrf"]


async def test_logged_in_dashboard_renders_keys(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client,
        [{"label": "acc1", "key": "sk-test-1234567890"}],
    )
    await _login(dashboard_client)
    # Re-fetch CSRF after seeding (login above issued a fresh pair).
    resp = await dashboard_client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text
    assert "acc1" in body
    assert "sk-test-" in body  # masked prefix is 8 chars
    # The CSRF token is embedded in the form.
    assert 'name="csrf_token"' in body


async def test_create_key_routes_to_created_page_with_raw_key(
    dashboard_client: AsyncClient,
) -> None:
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys",
        data={
            "label": "fresh",
            "key": "sk-fresh-the-real-key-1234567890",
            "csrf_token": cookies["csrf"],
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard/keys/created"

    # The flash cookie now carries the raw key (single-use).
    flash = _cookie_value(dashboard_client.cookies, FLASH_COOKIE)
    assert flash, "flash cookie not set"

    # Follow the redirect — the /dashboard/keys/created page renders
    # the raw key ONCE.
    resp2 = await dashboard_client.get("/dashboard/keys/created")
    assert resp2.status_code == 200
    assert "sk-fresh-the-real-key-1234567890" in resp2.text

    # A second visit shows no key (flash is consumed).
    resp3 = await dashboard_client.get("/dashboard/keys/created")
    assert "sk-fresh-the-real-key-1234567890" not in resp3.text


async def test_create_key_without_csrf_returns_403(
    dashboard_client: AsyncClient,
) -> None:
    await _login(dashboard_client)
    # Send a request WITHOUT the csrf_token field.
    resp = await dashboard_client.post(
        "/dashboard/keys",
        data={"label": "x", "key": "sk-test-1234"},
        follow_redirects=False,
    )
    assert resp.status_code == 403


async def test_create_key_with_wrong_csrf_returns_403(
    dashboard_client: AsyncClient,
) -> None:
    await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys",
        data={"label": "x", "key": "sk-test-1234", "csrf_token": "garbage"},
        follow_redirects=False,
    )
    assert resp.status_code == 403


async def test_create_key_409_sets_error_flash(
    dashboard_client: AsyncClient,
) -> None:
    cookies = await _login(dashboard_client)
    # Seed one key directly via the dashboard.
    await dashboard_client.post(
        "/dashboard/keys",
        data={
            "label": "existing",
            "key": "sk-dup",
            "csrf_token": cookies["csrf"],
        },
        follow_redirects=False,
    )
    # Re-login to refresh CSRF (the previous POST rotated it).
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys",
        data={"label": "x", "key": "sk-dup", "csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard/keys/new"

    # Follow the redirect and check the flash is rendered.
    resp2 = await dashboard_client.get("/dashboard/keys/new")
    assert resp2.status_code == 200
    assert "already exists" in resp2.text.lower()


async def test_dashboard_renders_when_no_keys(
    dashboard_client: AsyncClient,
) -> None:
    """Empty DB is not a failure: the dashboard renders with an empty table."""
    await _login(dashboard_client)
    resp = await dashboard_client.get("/dashboard")
    assert resp.status_code == 200
    # No keys → "showing 0 of 0".
    assert "showing 0 of 0" in resp.text


async def test_health_refresh_flash(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client,
        [
            {"label": "k1", "key": "sk-1111111111111111"},
            {"label": "k2", "key": "sk-2222222222222222"},
            {"label": "k3", "key": "sk-3333333333333333"},
        ],
    )
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys/refresh-health",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard"
    resp2 = await dashboard_client.get("/dashboard")
    assert "active=3" in resp2.text


async def test_test_all_flash(
    dashboard_client: AsyncClient, dashboard_state: Any, mock_router: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client,
        [
            {"label": "k1", "key": "sk-1111111111111111"},
            {"label": "k2", "key": "sk-2222222222222222"},
        ],
    )
    # Two probes; mock the upstream so each comes back with the
    # chosen status code. The probe hits ``POST /v1/chat/completions``.
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={"choices": [{"message": {"role": "assistant", "content": "pong"}}]},
    )
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=429,
        headers={"Retry-After": "60"},
        json_body={"error": "rate limited"},
    )
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys/test-all",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    resp2 = await dashboard_client.get("/dashboard")
    assert "total=2" in resp2.text
    assert "ok=1" in resp2.text
    assert "fail=1" in resp2.text


async def test_reset_states_flash(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client,
        [
            {"label": f"k{i}", "key": f"sk-{i:0>16}"} for i in range(1, 5)
        ],
    )
    # Reset the keys to a non-active state via direct DB writes, then
    # verify the route brings them back.
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        for key in result.scalars().all():
            await _set_key_status(
                dashboard_state.factory,
                key.id,
                status="depleted",
                cooldown_until=datetime(2099, 1, 1, tzinfo=UTC),
                last_status_code=429,
            )
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys/reset-states",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    resp2 = await dashboard_client.get("/dashboard")
    assert "active=4" in resp2.text


async def test_delete_key_redirects_with_flash(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client,
        [{"label": "k1", "key": "sk-1111111111111111"}],
    )
    # Look up via the in-memory engine for a stable id.
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        key = result.scalars().first()
        key_id = int(key.id)
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        f"/dashboard/keys/{key_id}/delete",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard"
    resp2 = await dashboard_client.get("/dashboard")
    assert f"key {key_id} deleted" in resp2.text


async def test_test_key_redirects_with_flash(
    dashboard_client: AsyncClient, dashboard_state: Any, mock_router: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client, [{"label": "acc1", "key": "sk-1111111111111111"}]
    )
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={"choices": [{"message": {"role": "assistant", "content": "pong"}}]},
    )
    cookies = await _login(dashboard_client)
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        key_id = int(result.scalars().first().id)
    resp = await dashboard_client.post(
        f"/dashboard/keys/{key_id}/test",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    resp2 = await dashboard_client.get("/dashboard")
    assert "ok=True" in resp2.text or "ok=true" in resp2.text
    assert "status_code=200" in resp2.text


async def test_update_key_via_method_patch(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client, [{"label": "old", "key": "sk-1111111111111111"}]
    )
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        key_id = int(result.scalars().first().id)
    await _login(dashboard_client)
    resp = await dashboard_client.get(f"/dashboard/keys/{key_id}")
    assert resp.status_code == 200
    fresh_csrf = _cookie_value(dashboard_client.cookies, CSRF_COOKIE)
    assert fresh_csrf, "csrf cookie was not set by the edit page"
    resp2 = await dashboard_client.post(
        f"/dashboard/keys/{key_id}",
        data={
            "label": "renamed",
            "status": "active",
            "_method": "patch",
            "csrf_token": fresh_csrf,
        },
        follow_redirects=False,
    )
    assert resp2.status_code == 303
    assert resp2.headers["location"] == "/dashboard"
    resp3 = await dashboard_client.get("/dashboard")
    assert f"key {key_id} updated" in resp3.text


async def test_update_key_invalid_status_flash(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client, [{"label": "x", "key": "sk-1111111111111111"}]
    )
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        key_id = int(result.scalars().first().id)
    await _login(dashboard_client)
    await dashboard_client.get(f"/dashboard/keys/{key_id}")
    fresh_csrf = _cookie_value(dashboard_client.cookies, CSRF_COOKIE)
    assert fresh_csrf, "csrf cookie was not set by the edit page"
    resp = await dashboard_client.post(
        f"/dashboard/keys/{key_id}",
        data={
            "label": "x",
            "status": "garbage",
            "_method": "patch",
            "csrf_token": fresh_csrf,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/dashboard/keys/{key_id}"


async def test_logout_clears_cookies(dashboard_client: AsyncClient) -> None:
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/logout",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    # After logout the session cookie should be cleared.
    session = _cookie_value(dashboard_client.cookies, SESSION_COOKIE)
    assert session in (None, ""), f"session cookie not cleared: {session!r}"


async def test_static_css_is_served(dashboard_client: AsyncClient) -> None:
    resp = await dashboard_client.get("/static/dashboard/web1.css")
    assert resp.status_code == 200
    assert "background-color" in resp.text


# --------------------------------------------------------------- status filter


_FILTERED_ROWS: list[dict[str, Any]] = [
    {
        "label": "act-1",
        "key": "sk-act1111111111",
        "status": "active",
        "last_used_at": datetime(2026, 7, 26, 21, 43, 12, tzinfo=UTC),
        "last_status_code": 200,
        "session_usage_fraction": 0.07,
        "weekly_usage_fraction": 0.19,
        "last_usage_fetch_at": datetime(2026, 7, 27, 16, 0, 0, tzinfo=UTC),
    },
    {
        "label": "act-2",
        "key": "sk-act2222222222",
        "status": "active",
        "last_used_at": datetime(2026, 7, 26, 21, 43, 13, tzinfo=UTC),
        "last_status_code": 200,
        "session_usage_fraction": None,
        "weekly_usage_fraction": None,
        "last_usage_fetch_at": None,
    },
    {
        "label": "dep-1",
        "key": "sk-dep1111111111",
        "status": "depleted",
        "last_used_at": datetime(2026, 7, 26, 21, 0, 0, tzinfo=UTC),
        "last_status_code": 429,
        "cooldown_until": datetime(2026, 7, 26, 22, 0, 0, tzinfo=UTC),
        "session_usage_fraction": 0.91,
        "weekly_usage_fraction": 0.45,
        "last_usage_fetch_at": datetime(2026, 7, 27, 15, 50, 0, tzinfo=UTC),
    },
    {
        "label": "dis-1",
        "key": "sk-dis1111111111",
        "status": "disabled",
        "last_used_at": datetime(2026, 7, 25, 10, 0, 0, tzinfo=UTC),
        "last_status_code": 401,
        "session_usage_fraction": None,
        "weekly_usage_fraction": None,
        "last_usage_fetch_at": None,
    },
]


async def _seed_filtered(dashboard_client: AsyncClient, dashboard_state: Any) -> None:
    await _seed_keys_via_dashboard(
        dashboard_client, [{"label": r["label"], "key": r["key"]} for r in _FILTERED_ROWS]
    )
    for row in _FILTERED_ROWS:
        # Look up the key id (stable insertion order) and update its
        # status + usage fractions directly.
        from sqlalchemy import select as _select

        from app.models.api_key import ApiKey

        async with dashboard_state.factory() as session:
            result = await session.execute(
                _select(ApiKey).where(ApiKey.key_prefix == row["key"][:8])
            )
            key = result.scalars().one()
            key_id = int(key.id)
        await _set_key_status(
            dashboard_state.factory,
            key_id,
            row["status"],
            cooldown_until=row.get("cooldown_until"),
            last_status_code=row.get("last_status_code"),
            last_used_at=row.get("last_used_at"),
            session_usage_fraction=row.get("session_usage_fraction"),
            weekly_usage_fraction=row.get("weekly_usage_fraction"),
            last_usage_fetch_at=row.get("last_usage_fetch_at"),
        )


async def test_dashboard_default_filter_shows_only_active(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_filtered(dashboard_client, dashboard_state)
    await _login(dashboard_client)
    resp = await dashboard_client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text
    # Active keys are visible.
    assert "act-1" in body
    assert "act-2" in body
    # Non-active keys are filtered out.
    assert "dep-1" not in body
    assert "dis-1" not in body
    # Counter shows 2 of 4.
    assert "showing 2 of 4" in body
    # The active filter chip is marked active, others are links.
    assert '<span class="filter-active">[ active ]</span>' in body
    # All other statuses are still available as filter links.
    assert '<a class="btn-link" href="/dashboard?status=all">all</a>' in body
    assert (
        '<a class="btn-link" href="/dashboard?status=depleted">depleted</a>'
        in body
    )
    assert (
        '<a class="btn-link" href="/dashboard?status=disabled">disabled</a>'
        in body
    )


async def test_dashboard_status_query_param_shows_selected(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_filtered(dashboard_client, dashboard_state)
    await _login(dashboard_client)
    resp = await dashboard_client.get("/dashboard?status=disabled")
    assert resp.status_code == 200
    body = resp.text
    assert "dis-1" in body
    assert "act-1" not in body
    assert "act-2" not in body
    assert "dep-1" not in body
    assert "showing 1 of 4" in body
    assert '<span class="filter-active">[ disabled ]</span>' in body


async def test_dashboard_status_all_shows_everything(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_filtered(dashboard_client, dashboard_state)
    await _login(dashboard_client)
    resp = await dashboard_client.get("/dashboard?status=all")
    assert resp.status_code == 200
    body = resp.text
    for label in ("act-1", "act-2", "dep-1", "dis-1"):
        assert label in body
    assert "showing 4 of 4" in body
    assert '<span class="filter-active">[ all ]</span>' in body


async def test_dashboard_status_bogus_falls_back_to_active(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_filtered(dashboard_client, dashboard_state)
    await _login(dashboard_client)
    resp = await dashboard_client.get("/dashboard?status=bogus")
    assert resp.status_code == 200
    body = resp.text
    assert "act-1" in body
    assert "act-2" in body
    assert "dep-1" not in body
    assert "dis-1" not in body
    assert '<span class="filter-active">[ active ]</span>' in body


async def test_dashboard_home_renders_usage_column_with_progress_bars(
    dashboard_client: AsyncClient, dashboard_state: Any
) -> None:
    await _seed_filtered(dashboard_client, dashboard_state)
    await _login(dashboard_client)
    resp = await dashboard_client.get("/dashboard?status=all")
    assert resp.status_code == 200
    body = resp.text

    # Column header present.
    assert "<th>usage (session / weekly)</th>" in body

    # act-1: 7.0% session, 19.0% weekly → normal navy bars.
    assert 'class="progress compact"' in body
    # act-2: no fraction → «—» labels appear.
    assert "session: —" in body
    assert "weekly: —" in body
    # dep-1: 91% session → ``progress-near-full`` modifier is present.
    assert "progress-near-full" in body
    # Percent labels rendered.
    assert "7.0%" in body
    assert "19.0%" in body
    assert "91.0%" in body
    # raw_key never appears.
    assert "raw_key" not in body.lower()


# --------------------------------------------------------------- usage


async def test_dashboard_key_usage_view_unauthed_redirects_to_login(
    dashboard_client: AsyncClient,
) -> None:
    """Unauthed GET /dashboard/keys/4/usage → 303 to /dashboard/login."""
    resp = await dashboard_client.get(
        "/dashboard/keys/4/usage", follow_redirects=False
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard/login"


async def test_dashboard_key_usage_view_renders(
    dashboard_client: AsyncClient, dashboard_state: Any, mock_router: Any
) -> None:
    """GET /dashboard/keys/{id}/usage renders the snapshot from
    ``UsageService``. With the mock router scripted for ``/api/me`` +
    ``/api/usage``, the page shows the joined snapshot and never
    leaks ``raw_key``.
    """
    await _seed_keys_via_dashboard(
        dashboard_client, [{"label": "acc4", "key": "sk-4444444444444444"}]
    )
    # Mock upstream responses. The UsageService issues two calls.
    mock_router.add(
        "POST",
        "/api/me",
        json_body={
            "id": "acc-uuid",
            "email": "acc4@example.com",
            "name": "acc4",
            "plan": "free",
        },
    )
    mock_router.add(
        "GET",
        "/api/usage",
        json_body={
            "activity": {
                "period": {
                    "type": "rolling",
                    "starting_at": "2026-07-26T18:00:00Z",
                    "ending_at": "2026-07-27T18:00:00Z",
                }
            },
            "limits": {
                "session": {"usage": 0.07, "models": []},
                "weekly": {"usage": 0.19, "models": []},
            },
        },
    )
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        key_id = int(result.scalars().first().id)
    await _login(dashboard_client)
    resp = await dashboard_client.get(f"/dashboard/keys/{key_id}/usage")
    assert resp.status_code == 200, resp.text
    body = resp.text
    assert "acc4@example.com" in body
    assert "free" in body
    assert "Session usage" in body
    assert "Weekly usage" in body
    assert 'class="progress"' in body
    assert "progress-fill" in body
    assert "7.0%" in body
    assert "19.0%" in body
    assert f'action="/dashboard/keys/{key_id}/usage/refresh"' in body
    assert "raw_key" not in body.lower()


async def test_dashboard_key_usage_refresh_redirects_with_flash(
    dashboard_client: AsyncClient, dashboard_state: Any, mock_router: Any
) -> None:
    """POST /dashboard/keys/{id}/usage/refresh → 303 + flash."""
    await _seed_keys_via_dashboard(
        dashboard_client, [{"label": "acc4", "key": "sk-4444444444444444"}]
    )
    # Two cycles of /api/me + /api/usage: refresh + the follow-up GET.
    for _ in range(2):
        mock_router.add(
            "POST",
            "/api/me",
            json_body={
                "id": "acc-uuid",
                "email": "acc4@example.com",
                "name": "acc4",
                "plan": "free",
            },
        )
        mock_router.add(
            "GET",
            "/api/usage",
            json_body={
                "activity": {},
                "limits": {
                    "session": {"usage": 0.0, "models": []},
                    "weekly": {"usage": 0.0, "models": []},
                },
            },
        )
    from sqlalchemy import select as _select

    from app.models.api_key import ApiKey

    async with dashboard_state.factory() as session:
        result = await session.execute(_select(ApiKey).order_by(ApiKey.id))
        key_id = int(result.scalars().first().id)
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        f"/dashboard/keys/{key_id}/usage/refresh",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == f"/dashboard/keys/{key_id}/usage"
    resp2 = await dashboard_client.get(f"/dashboard/keys/{key_id}/usage")
    assert f"key {key_id}: usage refreshed" in resp2.text


async def test_dashboard_usage_refresh_all_redirects_with_flash(
    dashboard_client: AsyncClient, dashboard_state: Any, mock_router: Any
) -> None:
    """POST /dashboard/keys/usage/refresh-all → 303 + flash summary."""
    await _seed_keys_via_dashboard(
        dashboard_client,
        [
            {"label": "k1", "key": "sk-1111111111111111"},
            {"label": "k4", "key": "sk-4444444444444444"},
        ],
    )
    for _ in range(4):  # two keys × (me + usage)
        mock_router.add(
            "POST",
            "/api/me",
            json_body={
                "id": "acc-uuid",
                "email": "x@example.com",
                "name": "x",
                "plan": "free",
            },
        )
        mock_router.add(
            "GET",
            "/api/usage",
            json_body={
                "activity": {},
                "limits": {
                    "session": {"usage": 0.0, "models": []},
                    "weekly": {"usage": 0.0, "models": []},
                },
            },
        )
    cookies = await _login(dashboard_client)
    resp = await dashboard_client.post(
        "/dashboard/keys/usage/refresh-all",
        data={"csrf_token": cookies["csrf"]},
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/dashboard"
    resp2 = await dashboard_client.get("/dashboard")
    assert "refreshed 2 active keys" in resp2.text
