from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import logger
from app.db.session import get_session_factory
from app.models.api_key import ApiKey, ApiKeyStatus
from app.schemas.api_key import (
    ApiKeyOut,
    ApiKeyTestResultWithKey,
    KeyUsageSnapshot,
)
from app.services._keys_ops import (
    DashboardClientError,
    ProbeClassifier,
    ProbeModelMissingError,
    VaultDecryptError,
    build_health_snapshot,
    hash_key,
    prefix_of,
    probe_all_active,
    safe_decrypt,
)
from app.services.key_manager import KeyManager, get_key_manager
from app.services.ollama_client import OllamaClient
from app.services.usage_service import UsageService
from app.services.vault import get_vault


def _require_probe_model() -> str:
    from app.services._keys_ops import require_probe_model

    try:
        return require_probe_model()
    except ProbeModelMissingError as exc:
        raise DashboardClientError(
            status_code=503,
            body={
                "detail": (
                    "probe_model is not configured; set PROBE_MODEL env var "
                    "to a model that the upstream accounts have access to"
                )
            },
            endpoint="probe",
            source="in-process",
        ) from exc

class InProcessDashboardBackend:

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        usage_service: UsageService,
    ) -> None:
        self._http_client = http_client
        self._usage_service = usage_service

    # ----------------------------------------------------- async-context shape

    async def __aenter__(self) -> InProcessDashboardBackend:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    # ----------------------------------------------------------------- helpers

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        factory = get_session_factory()
        async with factory() as session:
            yield session

    def _ollama(self) -> OllamaClient:
        return OllamaClient(self._http_client, get_settings())

    def _key_manager(self) -> KeyManager:
        return get_key_manager()

    # ------------------------------------------------------------------ keys

    async def list_keys(self) -> list[dict[str, Any]]:
        async with self._session() as session:
            result = await session.execute(select(ApiKey).order_by(ApiKey.id))
            rows = list(result.scalars().all())
            return [ApiKeyOut.from_orm_key(k).model_dump(mode="json") for k in rows]

    async def get_key(self, key_id: int) -> dict[str, Any]:
        async with self._session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise DashboardClientError(
                    status_code=404,
                    body={"detail": "API key not found"},
                    endpoint=f"GET /admin/keys/{key_id}",
                    source="in-process",
                )
            return ApiKeyOut.from_orm_key(key).model_dump(mode="json")

    async def create_key(
        self, label: str | None, key: str
    ) -> dict[str, Any]:
        key_hash = hash_key(key)
        key_prefix = prefix_of(key)
        vault = get_vault()
        key_encrypted = vault.encrypt(key)

        async with self._session() as session:
            existing = await session.execute(
                select(ApiKey).where(ApiKey.key_hash == key_hash)
            )
            if existing.scalar_one_or_none() is not None:
                raise DashboardClientError(
                    status_code=409,
                    body={"detail": "API key with the same value already exists"},
                    endpoint="POST /admin/keys",
                    source="in-process",
                )
            new_key = ApiKey(
                key_hash=key_hash,
                key_prefix=key_prefix,
                key_encrypted=key_encrypted,
                label=label,
                status=ApiKeyStatus.ACTIVE.value,
            )
            session.add(new_key)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                raise DashboardClientError(
                    status_code=409,
                    body={"detail": "API key with the same value already exists"},
                    endpoint="POST /admin/keys",
                    source="in-process",
                ) from None
            await session.refresh(new_key)
            logger.info("dashboard: create_key id={} label={}", new_key.id, new_key.label)
            payload = ApiKeyOut.from_orm_key(new_key).model_dump(mode="json")
            # raw_key is exposed only to the dashboard's "created"
            # template (single-use) — never persisted beyond this dict.
            payload["raw_key"] = key
            return payload

    async def update_key(
        self,
        key_id: int,
        *,
        label: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        async with self._session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise DashboardClientError(
                    status_code=404,
                    body={"detail": "API key not found"},
                    endpoint=f"PATCH /admin/keys/{key_id}",
                    source="in-process",
                )
            if label is not None:
                key.label = label
            if status is not None:
                key.status = status
            key.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(key)
            return ApiKeyOut.from_orm_key(key).model_dump(mode="json")

    async def delete_key(self, key_id: int) -> None:
        async with self._session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise DashboardClientError(
                    status_code=404,
                    body={"detail": "API key not found"},
                    endpoint=f"DELETE /admin/keys/{key_id}",
                    source="in-process",
                )
            await session.delete(key)
            await session.commit()
            logger.info("dashboard: delete_key id={}", key_id)

    # ----------------------------------------------------------------- probes

    async def test_key(self, key_id: int) -> dict[str, Any]:
        probe_model = _require_probe_model()
        async with self._session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise DashboardClientError(
                    status_code=404,
                    body={"detail": "API key not found"},
                    endpoint=f"POST /admin/keys/{key_id}/test",
                    source="in-process",
                )
            vault = get_vault()
            try:
                raw_key = safe_decrypt(vault, key)
            except VaultDecryptError as exc:
                raise DashboardClientError(
                    status_code=500,
                    body={"detail": f"cannot decrypt stored key: {exc.inner}"},
                    endpoint=f"POST /admin/keys/{key_id}/test",
                    source="in-process",
                ) from exc

            ollama = self._ollama()
            probe = await ollama.probe(
                key=raw_key, key_preview=key.key_preview, model=probe_model
            )
            await ProbeClassifier.classify(session, self._key_manager(), key, probe)
            joined = ApiKeyTestResultWithKey(
                **probe.model_dump(),
                id=key.id,
                label=key.label,
                key_preview=key.key_preview,
            )
            return joined.model_dump(mode="json")

    async def test_all_keys(self) -> dict[str, Any]:
        probe_model = _require_probe_model()
        async with self._session() as session:
            total, probes = await probe_all_active(
                session,
                ollama=self._ollama(),
                vault=get_vault(),
                keys_manager=self._key_manager(),
                probe_model=probe_model,
            )
            return {
                "total": total,
                "results": [p.model_dump(mode="json") for p in probes],
            }

    # ----------------------------------------------------------------- reset

    async def reset_states(self) -> dict[str, Any]:
        async with self._session() as session:
            now = datetime.now(UTC)
            stmt = (
                update(ApiKey)
                .values(
                    status=ApiKeyStatus.ACTIVE.value,
                    cooldown_until=None,
                    last_status_code=None,
                    total_failures=0,
                    updated_at=now,
                )
                .execution_options(synchronize_session=False)
            )
            result = await session.execute(stmt)
            await session.commit()
            reset_count = int(result.rowcount or 0)
            logger.info("dashboard: reset_states reset={} keys", reset_count)

            keys_manager = self._key_manager()
            health_snap = await build_health_snapshot(session, keys_manager, now=now)
            return health_snap.model_dump(mode="json")

    # ----------------------------------------------------------------- health

    async def health(self) -> dict[str, Any]:
        async with self._session() as session:
            keys_manager = self._key_manager()
            health_snap = await build_health_snapshot(session, keys_manager)
            return health_snap.model_dump(mode="json")

    # ----------------------------------------------------------------- usage

    async def get_key_usage(self, key_id: int) -> dict[str, Any]:
        async with self._session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise DashboardClientError(
                    status_code=404,
                    body={"detail": "API key not found"},
                    endpoint=f"GET /admin/keys/{key_id}/usage",
                    source="in-process",
                )
            snap: KeyUsageSnapshot = await self._usage_service.fetch_snapshot(
                api_key=key
            )
            return snap.model_dump(mode="json")

    async def refresh_key_usage(self, key_id: int) -> dict[str, Any]:
        async with self._session() as session:
            key = await session.get(ApiKey, key_id)
            if key is None:
                raise DashboardClientError(
                    status_code=404,
                    body={"detail": "API key not found"},
                    endpoint=f"POST /admin/keys/{key_id}/usage/refresh",
                    source="in-process",
                )
            snap = await self._usage_service.fetch_snapshot(api_key=key, force=True)
            await self._usage_service.persist_snapshot(
                session=session, api_key=key, snapshot=snap
            )
            return snap.model_dump(mode="json")

    async def refresh_all_usage(self) -> dict[str, Any]:
        async with self._session() as session:
            snapshots = await self._usage_service.refresh_all_active(session)
            return {
                "total": len(snapshots),
                "results": [snap.model_dump(mode="json") for snap in snapshots],
            }
