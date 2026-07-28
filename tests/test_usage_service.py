"""Tests for ``app.services.usage_service.UsageService``.

Strategy:
    * In-memory SQLite + a fresh ``ApiKey`` row for each test.
    * ``httpx.MockTransport`` substitutes the lifespan client so we can
      script ``/api/me`` and ``/api/usage`` responses (ok / 401 /
      unreachable).
    * Each test uses its own ``Vault`` instance so encrypted blobs
      don't leak across tests.

Covers:
    * Happy path: ``fetch_snapshot`` parses both endpoints and returns
      ``upstream_status="ok"``.
    * Cache hit: a second call within TTL does NOT issue a second
      ``/api/me`` call.
    * Auth rejection: 401 → ``upstream_status="unauthorised"``.
    * Unreachable: connection error → ``upstream_status="unreachable"``
      with stale cache returned if available.
    * ``refresh_all_active`` runs against every active key in parallel.
    * Persistence: a successful snapshot writes account_* fields
      and JSON of models back to the ``api_keys`` row.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.core.config import Settings
from app.models.api_key import ApiKey
from app.services.ollama_client import OllamaClient
from app.services.usage_service import UsageService, _parse_official
from app.services.vault import Vault

# ----------------------------------------------------------- helpers


def _account_payload() -> dict[str, Any]:
    return {
        "ID": "acct-xyz",
        "Email": "operator@example.com",
        "Name": "test-account",
        "Plan": "free",
    }


def _usage_payload() -> dict[str, Any]:
    # Mirrors the real ollama.com /api/usage payload: ``usage`` is a
    # fraction in [0, 1] (share of the limit consumed in the window),
    # not an absolute count. ``models[].request_count`` is the only
    # absolute number upstream exposes per window.
    return {
        "activity": {
            "cost": "0.00000",
            "period": {
                "type": "last_4_weeks",
                "starting_at": "2026-07-06T00:00:00Z",
                "ending_at": "2026-07-27T10:00:00Z",
            },
            "models": [],
        },
        "limits": {
            "session": {
                "usage": 0.037,
                "models": [{"name": "minimax-m3", "request_count": 5}],
            },
            "weekly": {
                "usage": 0.014,
                "models": [{"name": "minimax-m3", "request_count": 7}],
            },
        },
    }


def _mock_transport_factory(
    *,
    account: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    status_code: int = 200,
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Build a MockTransport that serves ``/api/me`` and ``/api/usage``.

    Returns ``(transport, recorded_requests)`` so tests can assert on
    the upstream calls that were issued.
    """
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        path = request.url.path
        if path == "/api/me":
            payload = account if account is not None else _account_payload()
        elif path == "/api/usage":
            payload = usage if usage is not None else _usage_payload()
        else:
            return httpx.Response(599, content=b"unexpected path " + path.encode())
        body = json.dumps(payload).encode("utf-8")
        return httpx.Response(status_code, content=body, headers={"content-type": "application/json"})

    return httpx.MockTransport(handler), captured


# ----------------------------------------------------------- pure parser tests


def test_parse_official_handles_missing_fields() -> None:
    """``_parse_official`` must not crash when upstream omits optional keys."""
    snap = _parse_official({}, fetched_at=__import__("datetime").datetime.now(__import__("datetime").UTC))
    assert snap.session_usage == 0
    assert snap.weekly_usage == 0
    assert snap.session_usage_fraction is None
    assert snap.weekly_usage_fraction is None
    assert snap.models == []
    assert snap.period_type is None


def test_parse_official_fraction_is_optional() -> None:
    """When ``limits.session.usage`` is missing, the fraction is ``None``."""
    payload = {
        "limits": {
            "session": {"models": []},  # no ``usage`` key
            "weekly": {"usage": 0.5, "models": []},
        },
    }
    snap = _parse_official(payload, fetched_at=__import__("datetime").datetime.now(__import__("datetime").UTC))
    assert snap.session_usage_fraction is None
    assert snap.session_usage == 0
    assert snap.weekly_usage_fraction is not None
    assert abs(snap.weekly_usage_fraction - 0.5) < 1e-9


def test_parse_official_fraction_clamped_to_zero() -> None:
    """``usage`` may be returned as 0.0; both views must reflect that."""
    payload = {
        "limits": {
            "session": {"usage": 0, "models": []},
            "weekly": {"usage": 0.0, "models": []},
        },
    }
    snap = _parse_official(payload, fetched_at=__import__("datetime").datetime.now(__import__("datetime").UTC))
    assert snap.session_usage_fraction == 0.0
    assert snap.weekly_usage_fraction == 0.0
    assert snap.session_usage == 0
    assert snap.weekly_usage == 0


def test_parse_official_dedupes_models_across_windows() -> None:
    """Same model in ``session`` and ``weekly`` should produce one row."""
    payload = _usage_payload()
    snap = _parse_official(payload, fetched_at=__import__("datetime").datetime.now(__import__("datetime").UTC))
    assert len(snap.models) == 1
    assert snap.models[0].name == "minimax-m3"
    # Mirrors the updated _usage_payload(): session request_count=5,
    # weekly request_count=7.
    assert snap.models[0].session_request_count == 5
    assert snap.models[0].weekly_request_count == 7


# ----------------------------------------------------------- fixtures


@pytest_asyncio.fixture
async def vault() -> Vault:
    return Vault(fernet=Fernet(Fernet.generate_key()))


@pytest_asyncio.fixture
async def mock_client_factory():
    """Yield a callable that returns ``(OllamaClient, captured_requests)``.

    Tests configure the response by calling ``make(account=..., usage=...,
    status_code=...)``.
    """
    clients: list[tuple[OllamaClient, list[httpx.Request], httpx.MockTransport]] = []

    def make(**kwargs: Any) -> tuple[OllamaClient, list[httpx.Request]]:
        transport, captured = _mock_transport_factory(**kwargs)
        http = httpx.AsyncClient(transport=transport, base_url="https://ollama.test")
        client = OllamaClient(http, Settings())
        clients.append((client, captured, transport))
        return client, captured

    yield make

    for _client, _captured, _transport in clients:
        await _client._http.aclose()


@pytest_asyncio.fixture
async def db_factory(db_engine):
    return async_sessionmaker(db_engine, expire_on_commit=False)


# ----------------------------------------------------------- service tests


@pytest.mark.asyncio
async def test_fetch_snapshot_happy_path(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    ollama, _captured = mock_client_factory()
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            label="acc-1",
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        api_key = await session.get(ApiKey, api_key.id)
        snap = await svc.fetch_snapshot(api_key=api_key)

    assert snap.upstream_status == "ok"
    assert snap.account is not None
    assert snap.account.email == "operator@example.com"
    assert snap.account.plan == "free"
    assert snap.official is not None
    # Legacy int view: truncation of the fraction to int.
    assert snap.official.session_usage == 0
    assert snap.official.weekly_usage == 0
    # New float view: the raw upstream fraction.
    assert snap.official.session_usage_fraction is not None
    assert snap.official.weekly_usage_fraction is not None
    assert abs(snap.official.session_usage_fraction - 0.037) < 1e-9
    assert abs(snap.official.weekly_usage_fraction - 0.014) < 1e-9
    assert snap.official.models[0].name == "minimax-m3"


@pytest.mark.asyncio
async def test_fetch_snapshot_uses_cache(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    """Second call within TTL must not re-issue upstream requests."""
    ollama, captured = mock_client_factory()
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        k1 = await session.get(ApiKey, api_key.id)
        s1 = await svc.fetch_snapshot(api_key=k1)
        s2 = await svc.fetch_snapshot(api_key=k1)

    assert s1.upstream_status == "ok"
    assert s2.upstream_status == "ok"
    # Two endpoints (me, usage) on the first call only.
    assert len(captured) == 2
    assert {r.url.path for r in captured} == {"/api/me", "/api/usage"}


@pytest.mark.asyncio
async def test_fetch_snapshot_force_refresh_bypasses_cache(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    ollama, captured = mock_client_factory()
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        k = await session.get(ApiKey, api_key.id)
        await svc.fetch_snapshot(api_key=k)
        await svc.fetch_snapshot(api_key=k, force=True)

    # 2 endpoints x 2 calls = 4 upstream requests.
    assert len(captured) == 4


@pytest.mark.asyncio
async def test_fetch_snapshot_unauthorised(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    ollama, _captured = mock_client_factory(status_code=401)
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        k = await session.get(ApiKey, api_key.id)
        snap = await svc.fetch_snapshot(api_key=k)

    assert snap.upstream_status == "unauthorised"
    assert snap.account is None
    assert snap.official is None


@pytest.mark.asyncio
async def test_fetch_snapshot_unreachable_returns_stale_or_pending(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    """Connection errors must surface as ``unreachable`` with stale cache if any."""
    # Build a transport that always errors.
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated upstream down")

    http = httpx.AsyncClient(transport=httpx.MockTransport(boom), base_url="https://ollama.test")
    ollama = OllamaClient(http, Settings())
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        k = await session.get(ApiKey, api_key.id)
        snap = await svc.fetch_snapshot(api_key=k)

    assert snap.upstream_status == "unreachable"
    assert snap.upstream_error is not None
    assert snap.account is None
    assert snap.official is None
    await http.aclose()


@pytest.mark.asyncio
async def test_refresh_all_active_persists(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    ollama, _captured = mock_client_factory()
    async with db_factory() as session:
        for i in range(3):
            session.add(
                ApiKey(
                    key_hash=f"h{i}" + "0" * 63,
                    key_prefix=f"sk-test-{i:02d}",
                    key_encrypted=vault.encrypt(f"sk-test-{i:02d}"),
                    status="active",
                )
            )
        await session.commit()

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        snaps = await svc.refresh_all_active(session)

    assert len(snaps) == 3
    assert all(s.upstream_status == "ok" for s in snaps)

    # Persistence: re-read the rows and confirm account_email was written.
    async with db_factory() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(ApiKey).order_by(ApiKey.id))).scalars().all()
        assert len(rows) == 3
        for r in rows:
            assert r.account_email == "operator@example.com"
            assert r.account_plan == "free"
            # Legacy int view is the truncated fraction; both are now
            # 0 because the fixture uses 0.037 / 0.014 (mirroring the
            # real upstream format).
            assert r.last_usage_session == 0
            assert r.last_usage_weekly == 0
            assert r.last_usage_session_fraction is not None
            assert abs(r.last_usage_session_fraction - 0.037) < 1e-9
            assert r.last_usage_weekly_fraction is not None
            assert abs(r.last_usage_weekly_fraction - 0.014) < 1e-9
            assert r.last_usage_models_json is not None
            parsed = json.loads(r.last_usage_models_json)
            assert parsed[0]["name"] == "minimax-m3"


@pytest.mark.asyncio
async def test_refresh_all_active_only_active_keys(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    """Depleted/disabled keys must be skipped."""
    ollama, captured = mock_client_factory()
    async with db_factory() as session:
        session.add(
            ApiKey(
                key_hash="a" * 64,
                key_prefix="sk-active",
                key_encrypted=vault.encrypt("sk-active-xxxxxxxx"),
                status="active",
            )
        )
        session.add(
            ApiKey(
                key_hash="b" * 64,
                key_prefix="sk-disabled",
                key_encrypted=vault.encrypt("sk-disabled-xxxxxx"),
                status="disabled",
            )
        )
        session.add(
            ApiKey(
                key_hash="c" * 64,
                key_prefix="sk-depleted",
                key_encrypted=vault.encrypt("sk-depleted-xxxxxxx"),
                status="depleted",
            )
        )
        await session.commit()

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        snaps = await svc.refresh_all_active(session)

    # Only 1 active key → 2 upstream requests (me + usage).
    assert len(snaps) == 1
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_concurrent_refresh_coalesces(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    """Many concurrent fetch_snapshot calls share one upstream round-trip
    via the per-key lock + cache freshness re-check."""
    ollama, captured = mock_client_factory()
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)
    async with db_factory() as session:
        k = await session.get(ApiKey, api_key.id)
        results = await asyncio.gather(*(svc.fetch_snapshot(api_key=k) for _ in range(5)))

    assert all(r.upstream_status == "ok" for r in results)
    # The lock ensures only one upstream round-trip is issued despite
    # 5 concurrent callers (2 endpoints).
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_persist_keeps_fraction_when_snapshot_lacks_it(
    vault: Vault, mock_client_factory: Any, db_factory: Any
) -> None:
    """If upstream omits the fraction field on a later fetch, the
    previously persisted fraction must NOT be overwritten with ``None`` —
    otherwise the dashboard's progress bar would flicker between
    refreshes when the upstream payload is temporarily incomplete.

    ``fetch_snapshot`` alone does not persist anything; we have to call
    :meth:`UsageService.persist_snapshot` explicitly to seed the row
    before exercising the «degraded payload» path.
    """
    from app.schemas.api_key import OfficialUsage

    ollama, _captured = mock_client_factory()
    async with db_factory() as session:
        api_key = ApiKey(
            key_hash="h" * 64,
            key_prefix="sk-test-",
            key_encrypted=vault.encrypt("sk-test-1234567890"),
            status="active",
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    svc = UsageService(ollama=ollama, vault=vault, ttl_seconds=300)

    # 1) Seed the row with a known fraction via ``persist_snapshot``.
    async with db_factory() as session:
        k = await session.get(ApiKey, api_key.id)
        snap = await svc.fetch_snapshot(api_key=k)
        assert snap.upstream_status == "ok"
        assert snap.official is not None
        await svc.persist_snapshot(session=session, api_key=k, snapshot=snap)

    # 2) Persist a degraded snapshot whose fractions are ``None`` and
    # confirm the previously written values survive.
    async with db_factory() as session:
        k = await session.get(ApiKey, api_key.id)
        degraded_official = OfficialUsage(
            session_usage=0,
            weekly_usage=0,
            session_usage_fraction=None,
            weekly_usage_fraction=None,
            models=[],
            fetched_at=snap.official.fetched_at,
        )
        from app.schemas.api_key import AccountInfo

        degraded_snap = snap.model_copy(
            update={
                "official": degraded_official,
                "account": AccountInfo(
                    account_id=snap.account.account_id,
                    email=snap.account.email,
                    name=snap.account.name,
                    plan=snap.account.plan,
                    fetched_at=snap.account.fetched_at,
                )
                if snap.account is not None
                else None,
            }
        )
        await svc.persist_snapshot(session=session, api_key=k, snapshot=degraded_snap)

    # 3) Reload and confirm the fractions survived.
    async with db_factory() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(ApiKey).order_by(ApiKey.id))).scalars().all()
        assert len(rows) == 1
        r = rows[0]
        assert r.last_usage_session_fraction is not None
        assert abs(r.last_usage_session_fraction - 0.037) < 1e-9
        assert r.last_usage_weekly_fraction is not None
        assert abs(r.last_usage_weekly_fraction - 0.014) < 1e-9
