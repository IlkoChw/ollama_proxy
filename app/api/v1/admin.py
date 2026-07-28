from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_ollama_client, get_usage_service, require_admin_token
from app.core.logging import logger
from app.db.session import get_session
from app.models.api_key import ApiKey, ApiKeyStatus
from app.schemas.api_key import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyHealth,
    ApiKeyOut,
    ApiKeyTestResultWithKey,
    ApiKeyUpdate,
    KeyUsageSnapshot,
)
from app.services._keys_ops import (
    ApiKeyNotFoundError,
    ProbeClassifier,
    ProbeModelMissingError,
    VaultDecryptError,
    build_health_snapshot,
    hash_key,
    load_or_404,
    prefix_of,
    probe_all_active,
    require_probe_model,
    safe_decrypt,
)
from app.services.key_manager import KeyManager, get_key_manager
from app.services.ollama_client import OllamaClient
from app.services.usage_service import UsageService
from app.services.vault import get_vault

router = APIRouter()
health_router = APIRouter()

# ----------------------------------------------------------------- helpers

def _probe_model_or_503() -> str:
    try:
        return require_probe_model()
    except ProbeModelMissingError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "probe_model is not configured; set PROBE_MODEL env var "
                "to a model that the upstream accounts have access to"
            ),
        ) from exc

# ----------------------------------------------------------------- CRUD

@router.post(
    "/keys",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    tags=["admin"],
    summary="Create an API key (returns raw_key once)",
    dependencies=[Depends(require_admin_token)],
)
async def create_key(
    payload: ApiKeyCreate,
    session: AsyncSession = Depends(get_session),
) -> ApiKeyCreated:
    key_hash = hash_key(payload.key)
    key_prefix = prefix_of(payload.key)
    key_encrypted = get_vault().encrypt(payload.key)

    # Optimistic existence check (also gives a clean 409 instead of relying
    # on the IntegrityError round-trip).
    existing = await session.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="API key with the same value already exists",
        )

    new_key = ApiKey(
        key_hash=key_hash,
        key_prefix=key_prefix,
        key_encrypted=key_encrypted,
        label=payload.label,
        status=ApiKeyStatus.ACTIVE.value,
    )
    session.add(new_key)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="API key with the same value already exists",
        ) from None
    await session.refresh(new_key)

    logger.info("create_key: id={} label={}", new_key.id, new_key.label)
    return ApiKeyCreated(
        **ApiKeyOut.from_orm_key(new_key).model_dump(),
        raw_key=payload.key,
    )

@router.get(
    "/keys", response_model=list[ApiKeyOut], tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)
async def list_keys(session: AsyncSession = Depends(get_session)) -> list[ApiKeyOut]:
    result = await session.execute(select(ApiKey).order_by(ApiKey.id))
    rows = list(result.scalars().all())
    return [ApiKeyOut.from_orm_key(k) for k in rows]

@router.get(
    "/keys/{key_id}", response_model=ApiKeyOut, tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)
async def get_key(
    key_id: int,
    session: AsyncSession = Depends(get_session),
) -> ApiKeyOut:
    try:
        key = await load_or_404(session, key_id)
    except ApiKeyNotFoundError as err:
        raise HTTPException(status_code=404, detail="API key not found") from err
    return ApiKeyOut.from_orm_key(key)

@router.patch(
    "/keys/{key_id}", response_model=ApiKeyOut, tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)
async def update_key(
    key_id: int,
    payload: ApiKeyUpdate,
    session: AsyncSession = Depends(get_session),
) -> ApiKeyOut:
    try:
        key = await load_or_404(session, key_id)
    except ApiKeyNotFoundError as err:
        raise HTTPException(status_code=404, detail="API key not found") from err

    if payload.label is not None:
        key.label = payload.label
    if payload.status is not None:
        key.status = payload.status.value
    key.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(key)
    return ApiKeyOut.from_orm_key(key)

@router.delete(
    "/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    tags=["admin"],
    dependencies=[Depends(require_admin_token)],
)
async def delete_key(
    key_id: int,
    session: AsyncSession = Depends(get_session),
) -> Response:
    try:
        key = await load_or_404(session, key_id)
    except ApiKeyNotFoundError as err:
        raise HTTPException(status_code=404, detail="API key not found") from err
    await session.delete(key)
    await session.commit()
    logger.info("delete_key: id={}", key_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ----------------------------------------------------------------- probes

@router.post(
    "/keys/{key_id}/test",
    response_model=ApiKeyTestResultWithKey,
    tags=["admin"],
    summary="Probe a single API key with a minimal LLM request",
    dependencies=[Depends(require_admin_token)],
)
async def test_one_key(
    key_id: int,
    session: AsyncSession = Depends(get_session),
    ollama: OllamaClient = Depends(get_ollama_client),
) -> ApiKeyTestResultWithKey:
    try:
        key = await load_or_404(session, key_id)
    except ApiKeyNotFoundError as err:
        raise HTTPException(status_code=404, detail="API key not found") from err

    try:
        raw_key = safe_decrypt(get_vault(), key)
    except VaultDecryptError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"cannot decrypt stored key: {exc.inner}",
        ) from exc

    probe_model = _probe_model_or_503()
    probe = await ollama.probe(
        key=raw_key, key_preview=key.key_preview, model=probe_model
    )
    keys_manager: KeyManager = get_key_manager()
    await ProbeClassifier.classify(session, keys_manager, key, probe)
    return ApiKeyTestResultWithKey(
        **probe.model_dump(),
        id=key.id,
        label=key.label,
        key_preview=key.key_preview,
    )

@router.post(
    "/keys/test-all",
    response_model=dict,
    tags=["admin"],
    summary="Probe every active API key concurrently",
    dependencies=[Depends(require_admin_token)],
)
async def test_all_keys(
    session: AsyncSession = Depends(get_session),
    ollama: OllamaClient = Depends(get_ollama_client),
) -> dict:
    probe_model = _probe_model_or_503()
    keys_manager: KeyManager = get_key_manager()
    total, probes = await probe_all_active(
        session,
        ollama=ollama,
        vault=get_vault(),
        keys_manager=keys_manager,
        probe_model=probe_model,
    )
    return {"total": total, "results": [p.model_dump() for p in probes]}

# ----------------------------------------------------------------- reset

@router.post(
    "/keys/reset-states",
    response_model=ApiKeyHealth,
    tags=["admin"],
    summary="Reset all keys back to 'active' (clears disabled/depleted + cooldowns)",
    dependencies=[Depends(require_admin_token)],
)
async def reset_key_states(
    session: AsyncSession = Depends(get_session),
) -> ApiKeyHealth:
    from sqlalchemy import update as _update  # local import to keep top tidy

    now = datetime.now(UTC)
    stmt = (
        _update(ApiKey)
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
    logger.info("reset_key_states: reset={} keys", reset_count)

    keys_manager: KeyManager = get_key_manager()
    return await build_health_snapshot(session, keys_manager, now=now)

# ----------------------------------------------------------------- usage

@router.get(
    "/keys/{key_id}/usage",
    response_model=KeyUsageSnapshot,
    tags=["admin"],
    summary="Get combined upstream + local usage snapshot for a key",
    dependencies=[Depends(require_admin_token)],
)
async def get_key_usage(
    key_id: int,
    session: AsyncSession = Depends(get_session),
    usage: UsageService = Depends(get_usage_service),
) -> KeyUsageSnapshot:
    try:
        key = await load_or_404(session, key_id)
    except ApiKeyNotFoundError as err:
        raise HTTPException(status_code=404, detail="API key not found") from err
    return await usage.fetch_snapshot(api_key=key)

@router.post(
    "/keys/{key_id}/usage/refresh",
    response_model=KeyUsageSnapshot,
    tags=["admin"],
    summary="Force-refresh upstream usage snapshot for a key",
    dependencies=[Depends(require_admin_token)],
)
async def refresh_key_usage(
    key_id: int,
    session: AsyncSession = Depends(get_session),
    usage: UsageService = Depends(get_usage_service),
) -> KeyUsageSnapshot:
    try:
        key = await load_or_404(session, key_id)
    except ApiKeyNotFoundError as err:
        raise HTTPException(status_code=404, detail="API key not found") from err
    snap = await usage.fetch_snapshot(api_key=key, force=True)
    await usage.persist_snapshot(session=session, api_key=key, snapshot=snap)
    return snap

@router.post(
    "/keys/usage/refresh-all",
    response_model=dict,
    tags=["admin"],
    summary="Force-refresh upstream usage for every active key",
    dependencies=[Depends(require_admin_token)],
)
async def refresh_all_usage(
    session: AsyncSession = Depends(get_session),
    usage: UsageService = Depends(get_usage_service),
) -> dict:
    snapshots = await usage.refresh_all_active(session)
    return {
        "total": len(snapshots),
        "results": [snap.model_dump(mode="json") for snap in snapshots],
    }

# ----------------------------------------------------------------- health

@health_router.get(
    "/admin/health",
    response_model=ApiKeyHealth,
    tags=["health"],
    dependencies=[Depends(require_admin_token)],
)
async def health(session: AsyncSession = Depends(get_session)) -> ApiKeyHealth:
    keys_manager: KeyManager = get_key_manager()
    return await build_health_snapshot(session, keys_manager)
