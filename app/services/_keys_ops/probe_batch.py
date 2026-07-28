from __future__ import annotations

import asyncio

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey, ApiKeyStatus
from app.schemas.api_key import ApiKeyTestResult, ApiKeyTestResultWithKey
from app.services.key_manager import KeyManager
from app.services.ollama_client import OllamaClient
from app.services.vault import Vault

from .errors import VaultDecryptError
from .probe import ProbeClassifier
from .vault import safe_decrypt

__all__ = ["probe_all_active"]

async def probe_all_active(
    session: AsyncSession,
    *,
    ollama: OllamaClient,
    vault: Vault,
    keys_manager: KeyManager,
    probe_model: str,
) -> tuple[int, list[ApiKeyTestResultWithKey]]:
    rows = await session.execute(
        select(ApiKey)
        .where(ApiKey.status == ApiKeyStatus.ACTIVE.value)
        .order_by(ApiKey.id)
    )
    active_keys = list(rows.scalars().all())
    if not active_keys:
        return 0, []

    async def _run(k: ApiKey) -> tuple[ApiKey, ApiKeyTestResultWithKey]:
        try:
            raw_key = safe_decrypt(vault, k)
        except VaultDecryptError as exc:
            return k, ApiKeyTestResultWithKey(
                ok=False,
                status_code=None,
                error=f"cannot decrypt stored key: {exc.inner}",
                id=k.id,
                label=k.label,
                key_preview=k.key_preview,
            )

        try:
            probe = await ollama.probe(
                key=raw_key, key_preview=k.key_preview, model=probe_model
            )
        except (httpx.TimeoutException, httpx.RequestError, OSError) as exc:
            await keys_manager.record_server_error(session, k.id, 0)
            probe = ApiKeyTestResult(
                ok=False,
                status_code=None,
                error=str(exc),
            )
        joined = ApiKeyTestResultWithKey(
            **probe.model_dump(),
            id=k.id,
            label=k.label,
            key_preview=k.key_preview,
        )
        return k, joined

    pairs = await asyncio.gather(*(_run(k) for k in active_keys))

    for k, joined in pairs:
        await ProbeClassifier.classify(session, keys_manager, k, joined)

    results = [joined for _k, joined in pairs]
    return len(results), results
