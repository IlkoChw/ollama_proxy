"""Test fixtures: in-memory SQLite, mock httpx client, ASGI client.

Strategy:
    * Each test session uses a fresh in-memory ``aiosqlite`` database.
    * The module-level ``get_session`` factory is rebound to that engine.
    * ``httpx.MockTransport`` substitutes the lifespan client so we can
      script upstream responses (status codes, headers, bodies).
    * ``httpx.ASGITransport`` drives the FastAPI app in-process, with
      ``app.dependency_overrides`` clearing the lifespan dependency on
      ``get_http_client`` (otherwise it would try to read ``app.state``
      which only the lifespan sets up).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Set a known env BEFORE any app imports so Settings picks it up.
os.environ.setdefault("OLLAMA_BASE_URL", "https://ollama.test")
os.environ.setdefault("DB_PATH", ":memory:")
# NOTE: tests in test_proxy.py expect the static last-resort model to be
# "minimax-m2.7" (the value baked into app/api/v1/proxy.py at import time).
# Keep the env in sync so the import-time default matches.
os.environ.setdefault("PROBE_MODEL", "minimax-m2.7")
os.environ.setdefault("TIMEOUT", "5")
# Tight safety margin in tests so the rotation loop matches the number
# of registered mock routes exactly.
os.environ.setdefault("MAX_ROTATION_ITERATIONS_SAFETY_MARGIN", "0")
# Stable admin token for the test session so existing tests keep working
# without an Authorization header (they all share this token via fixture).
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token-do-not-use-in-prod")
# Disable CORS middleware in tests — we hit the app in-process.
os.environ.setdefault("CORS_ALLOW_ORIGINS", "")
# Enable Swagger UI in tests by default so the schema is available if
# any test introspects it. Tests that exercise the off-by-default path
# override this via monkeypatch + ``importlib.reload(app.main)``.
os.environ.setdefault("ENABLE_DOCS", "1")

from cryptography.fernet import Fernet  # noqa: E402

from app.api.deps import get_http_client, get_model_cache  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.db import session as session_module  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.main import app  # noqa: E402
from app.services import key_manager as key_manager_module  # noqa: E402
from app.services import vault as vault_module  # noqa: E402
from app.services.model_cache import ModelCache  # noqa: E402

# ------------------------------------------------------------ settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Clear the lru_cache on ``get_settings`` so env changes take effect."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Return a freshly-built Settings (env-driven)."""
    return Settings()


# ------------------------------------------------------------ DB


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[Any]:
    """Yield a fresh in-memory async engine; drop tables after the test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session_factory(db_engine: Any) -> async_sessionmaker[AsyncSession]:
    """Return a sessionmaker bound to the in-memory engine."""
    return async_sessionmaker(db_engine, expire_on_commit=False, class_=AsyncSession)


@pytest_asyncio.fixture
async def db_session(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a single AsyncSession that is closed at the end of the test."""
    async with db_session_factory() as session:
        yield session


# ------------------------------------------------------------ http mock


class MockRouter:
    """Programmable httpx response router used as a ``MockTransport`` handler.

    Tests can register routes by (method, path) → ``(status, headers, body)``
    or call ``set_default`` for a catch-all. When no route matches and no
    default is set, a 599 ``RuntimeError`` is raised so tests fail loudly.

    Concurrency: ``httpx.Response`` objects are single-use — once httpx
    has streamed the body from a response it cannot be re-read by a
    second caller. Under burst load (e.g. 50 concurrent requests) the
    router therefore **clones the registered response on every call**
    by re-creating an ``httpx.Response`` with the same status, headers,
    and body bytes. The original template is preserved unchanged so
    the test can inspect it after the fact.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], list[httpx.Response]] = {}
        self.default: httpx.Response | None = None
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    @staticmethod
    def _clone(template: httpx.Response) -> httpx.Response:
        """Return a fresh ``httpx.Response`` with the same observable state.

        ``httpx.Response.content`` is the bytes that were passed in (or
        that the original was constructed with). Re-using the same
        bytes across clones is safe — only the response wrapper itself
        is single-use.
        """
        return httpx.Response(
            status_code=template.status_code,
            headers=dict(template.headers),
            content=template.content,
        )

    def add(
        self,
        method: str,
        path: str,
        status: int = 200,
        json_body: Any = None,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        if text is not None:
            body = text.encode("utf-8")
        elif json_body is not None:
            # Use httpx's own JSON encoder so the result is identical to
            # what the real client would emit.
            import json as _json

            body = _json.dumps(json_body).encode("utf-8")
        else:
            body = b""
        resp = httpx.Response(
            status_code=status,
            headers=headers or {},
            content=body,
        )
        self.routes.setdefault((method.upper(), path), []).append(resp)

    def set_default(self, response: httpx.Response) -> None:
        self.default = response

    async def handle(self, request: httpx.Request) -> httpx.Response:
        key = (request.method.upper(), request.url.path)
        self.calls.append((request.method, request.url.path, None))
        queue = self.routes.get(key)
        if queue:
            return MockRouter._clone(queue.pop(0))
        if self.default is not None:
            return MockRouter._clone(self.default)
        return httpx.Response(599, content=b"no mock route for " + str(key).encode())


@pytest_asyncio.fixture
async def mock_router() -> MockRouter:
    """Yield a fresh :class:`MockRouter` and reset its state per test."""
    return MockRouter()


@pytest_asyncio.fixture
async def mock_http_client(mock_router: MockRouter) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an httpx.AsyncClient whose transport is the MockRouter."""
    transport = httpx.MockTransport(mock_router.handle)
    async with httpx.AsyncClient(transport=transport, base_url="https://ollama.test") as client:
        yield client


# ------------------------------------------------------------ ASGI app


@pytest_asyncio.fixture
async def model_cache() -> ModelCache:
    """Yield a fresh :class:`ModelCache` (TTL = 300s) for tests."""
    return ModelCache(ttl_seconds=300)


# Stable admin token used by the ASGI test clients below. The value must
# match the ``ADMIN_TOKEN`` env var set at the top of this file so that
# every request the test client makes carries a valid bearer.
TEST_ADMIN_TOKEN = "test-admin-token-do-not-use-in-prod"


@pytest_asyncio.fixture
async def client(
    db_engine: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    mock_http_client: httpx.AsyncClient,
    model_cache: ModelCache,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an ``httpx.AsyncClient`` driving the FastAPI app in-process.

    Wires:
        * The session factory into ``get_session`` via dependency override.
        * The mock http client into ``get_http_client`` via dep override.
        * The per-test ``ModelCache`` into ``get_model_cache`` via override.
        * Resets the ``KeyManager`` singleton so the rotation counter
          starts from zero for each test.
    """
    session_module.override_session_factory(db_session_factory)
    key_manager_module.reset_key_manager_for_tests()
    # Each test gets a fresh ephemeral vault (no file system access).
    vault_module.set_vault(vault_module.Vault(fernet=Fernet(Fernet.generate_key())))

    async def _override_session() -> AsyncIterator[AsyncSession]:
        async with db_session_factory() as s:
            yield s

    async def _override_http_client() -> httpx.AsyncClient:
        return mock_http_client

    def _override_model_cache() -> ModelCache:
        return model_cache

    app.dependency_overrides[session_module.get_session] = _override_session
    app.dependency_overrides[get_http_client] = _override_http_client
    app.dependency_overrides[get_model_cache] = _override_model_cache

    # Auto-attach the admin bearer header. The ``client`` fixture is
    # used by every test that hits ``/admin/*`` or ``/admin/health``;
    # they all run as if the operator had configured ``ADMIN_TOKEN``.
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {TEST_ADMIN_TOKEN}"}
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=headers
        ) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        session_module.reset_for_tests()


@pytest_asyncio.fixture
async def user_client(
    client: httpx.AsyncClient,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield a fresh ``httpx.AsyncClient`` carrying an end-user bearer token.

    The proxy now requires ``Authorization: Bearer opk_…`` on every
    ``/v1/*`` and ``/api/tags`` request (see
    ``app/api/v1/proxy.py``'s router-level ``require_user_token``
    dependency). Tests that exercise proxy behaviour adopt this
    fixture so the auth dependency accepts the request; tests that
    specifically assert 401-on-missing-auth keep using ``client``
    (without a user header) instead.

    One user token is created per fixture lifetime and consumed by the
    test. The token is created via the admin-authenticated ``client``
    fixture; this fixture only depends on ``client`` for the setup
    POST and otherwise operates on its own ASGITransport so admin
    setup and proxy exercise are isolated.
    """
    # Mint a user token through the admin-authenticated client.
    resp = await client.post(
        "/admin/user-tokens", json={"label": "test-user"}
    )
    assert resp.status_code == 201, (resp.status_code, resp.text)
    raw_key: str = resp.json()["raw_key"]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {raw_key}"},
    ) as uclient:
        yield uclient
        vault_module.reset_vault_for_tests()
