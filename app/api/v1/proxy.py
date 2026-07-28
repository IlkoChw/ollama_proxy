from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_model_cache,
    get_ollama_client,
    get_rotation,
    require_user_token,
)
from app.core.config import get_settings
from app.core.logging import logger
from app.db.session import get_session
from app.schemas.proxy import (
    ChatCompletionRequest,
    ModelInfo,
    ModelsResponse,
    ModelTag,
    TagsResponse,
)
from app.services._keys_ops import VaultDecryptError, safe_decrypt
from app.services.key_manager import KeyManager, get_key_manager
from app.services.model_cache import ModelCache
from app.services.ollama_client import OllamaClient
from app.services.rotation import Rotation
from app.services.vault import get_vault

router = APIRouter(
    dependencies=[Depends(require_user_token)],
)

_STATIC_MODEL_ID = get_settings().probe_model or ""
_STATIC_MODELS = ModelsResponse(
    data=[ModelInfo(id=_STATIC_MODEL_ID, created=1700000000, owned_by="ollama")]
)
_STATIC_TAGS = TagsResponse(
    models=[
        ModelTag(
            name=_STATIC_MODEL_ID,
            model=_STATIC_MODEL_ID,
            modified_at="2024-01-01T00:00:00Z",
            size=0,
            digest="",
            details={"family": "minimax", "parameter_size": "", "quantization_level": ""},
        )
    ]
)

# --------------------------------------------------------------- helpers

def _models_to_response(model_ids: list[str]) -> ModelsResponse:
    return ModelsResponse(
        data=[
            ModelInfo(id=m, created=0, owned_by="ollama") for m in model_ids
        ]
    )

def _models_to_tags(model_ids: list[str]) -> TagsResponse:
    return TagsResponse(
        models=[
            ModelTag(
                name=m,
                model=m,
                modified_at="",
                size=0,
                digest="",
                details={},
            )
            for m in model_ids
        ]
    )

async def _fetch_models_via_rotation(
    session: AsyncSession,
    ollama: OllamaClient,
) -> tuple[bool, list[str]]:
    keys: KeyManager = get_key_manager()
    key = await keys.pick_next_key(session)
    if key is None:
        return False, []
    try:
        try:
            raw_key = safe_decrypt(get_vault(), key)
        except VaultDecryptError as exc:
            logger.warning(
                "list_models_via_rotation: cannot decrypt key id={} err={}",
                exc.key_id,
                exc.inner,
            )
            return False, []
        try:
            model_ids = await ollama.list_models_v1(
                key=raw_key, key_preview=key.key_preview
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            await keys.record_server_error(session, key.id, 0)
            logger.warning(
                "list_models_via_rotation: network error key={} err={}",
                key.key_preview,
                exc,
            )
            return False, []
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in (401, 403):
                await keys.record_unauthorized(session, key.id, code)
            elif code == 429:
                retry_after = KeyManager.extract_retry_after(exc.response.headers)
                await keys.record_rate_limited(session, key.id, code, retry_after)
            elif code >= 500:
                await keys.record_server_error(session, key.id, code)
            # 404 is unusual for /v1/models; ignore and fall back.
            logger.info(
                "list_models_via_rotation: upstream unhappy key={} code={}",
                key.key_preview,
                code,
            )
            return False, []
        else:
            # 2xx — record success and return.
            await keys.record_success(session, key.id, 200)
            return True, model_ids
    finally:
        # Anti-stacking release: covers all exit paths above.
        await keys.release(key.id)

# -------------------------------------------------------------------- /v1/models

@router.get("/v1/models", response_model=ModelsResponse, tags=["proxy"])
async def list_models(
    session: AsyncSession = Depends(get_session),
    ollama: OllamaClient = Depends(get_ollama_client),
    cache: ModelCache = Depends(get_model_cache),
) -> ModelsResponse:
    cached = cache.get()
    if cached is None:
        # Try to refresh. If refresh fails, we get back whatever was in
        # the cache (possibly empty) or an empty list.
        async def _fetch() -> list[str]:
            ok, ids = await _fetch_models_via_rotation(session, ollama)
            return ids if ok else (cache.get_stale() or [])

        models = await cache.refresh(_fetch)
        if not models:
            # Last-resort static.
            return _STATIC_MODELS
        return _models_to_response(models)

    return _models_to_response(cached)

# -------------------------------------------------------------------- /api/tags

@router.get("/api/tags", tags=["proxy"])
async def list_tags(
    session: AsyncSession = Depends(get_session),
    ollama: OllamaClient = Depends(get_ollama_client),
    cache: ModelCache = Depends(get_model_cache),
) -> Response:
    cached = cache.get()
    if cached is None:
        async def _fetch() -> list[str]:
            ok, ids = await _fetch_models_via_rotation(session, ollama)
            return ids if ok else (cache.get_stale() or [])

        models = await cache.refresh(_fetch)
        if not models:
            return JSONResponse(_STATIC_TAGS.model_dump())
        return JSONResponse(_models_to_tags(models).model_dump())

    return JSONResponse(_models_to_tags(cached).model_dump())

# ------------------------------------------------------------ chat completions

@router.post(
    "/v1/chat/completions",
    tags=["proxy"],
    summary="OpenAI-compatible chat completions, proxied through rotation",
)
async def chat_completions(
    request: Request,
    session: AsyncSession = Depends(get_session),
    rotation: Rotation = Depends(get_rotation),
) -> Response:
    raw_body = await request.body()
    try:
        parsed: dict[str, Any] = json.loads(raw_body) if raw_body else {}
    except json.JSONDecodeError as exc:
        return Response(
            content=json.dumps({"detail": f"invalid JSON: {exc}"}).encode(),
            status_code=400,
            media_type="application/json",
        )

    try:
        validated = ChatCompletionRequest.model_validate(parsed)
    except Exception as exc:
        return Response(
            content=json.dumps({"detail": str(exc)}).encode(),
            status_code=422,
            media_type="application/json",
        )
    body = validated.model_dump(exclude_none=True)
    try:
        return await rotation.execute_with_rotation(session, body)
    except Exception as exc:  # pragma: no cover — safety net only
        logger.exception(
            "chat_completions: unexpected exception in execute_with_rotation: {}",
            exc,
        )
        return Response(
            content=json.dumps({"detail": "all backends failed"}).encode(),
            status_code=503,
            media_type="application/json",
        )
