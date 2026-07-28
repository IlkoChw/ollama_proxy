from __future__ import annotations

import hmac
import secrets as _py_secrets

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security.utils import get_authorization_scheme_param
from starlette.datastructures import State

from app.core.config import Settings, get_settings
from app.core.logging import logger
from app.db.session import get_session
from app.models.user_token import UserToken
from app.services.key_manager import KeyManager, get_key_manager
from app.services.model_cache import ModelCache
from app.services.ollama_client import OllamaClient
from app.services.rotation import Rotation
from app.services.usage_service import UsageService
from app.services.user_token_service import authenticate as authenticate_user_token
from app.services.user_token_service import has_any_active_token
from app.services.vault import Vault, get_vault


def get_settings_dep() -> Settings:
    return get_settings()

# ----------------------------------------------------------------- admin auth

def _extract_bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization")
    if not auth:
        return None
    scheme, param = get_authorization_scheme_param(auth)
    if scheme.lower() != "bearer" or not param:
        return None
    return param

def require_admin_token(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> None:
    configured = settings.admin_token
    if not configured:
        return
    presented = _extract_bearer(request)
    if presented is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization: Bearer <ADMIN_TOKEN>",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )
    if not hmac.compare_digest(presented.encode("utf-8"), configured.encode("utf-8")):
        # Use the same code as "missing" to avoid disclosing whether the
        # header was the only thing missing.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid admin token",
            headers={"WWW-Authenticate": 'Bearer realm="admin"'},
        )

def generate_admin_token() -> str:
    return _py_secrets.token_urlsafe(32)

def generate_dashboard_session_secret() -> str:
    return _py_secrets.token_urlsafe(32)

# ----------------------------------------------------------------- user tokens

async def require_user_token(
    request: Request,
    session = Depends(get_session),
) -> UserToken | None:
    presented = _extract_bearer(request)
    if presented is None:
        # No bearer → check whether we're in dev mode (table empty).
        if not await has_any_active_token(session):
            _log_dev_mode_once()
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization: Bearer <opk_…>",
            headers={"WWW-Authenticate": 'Bearer realm="proxy"'},
        )
    try:
        token = await authenticate_user_token(session, presented)
    except ValueError:
        # Malformed bearer; treat as invalid without revealing why.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": 'Bearer realm="proxy"'},
        ) from None
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired bearer token",
            headers={"WWW-Authenticate": 'Bearer realm="proxy"'},
        )
    return token

_dev_mode_warned: bool = False

def _log_dev_mode_once() -> None:
    global _dev_mode_warned  # noqa: PLW0603
    if _dev_mode_warned:
        return
    _dev_mode_warned = True
    logger.warning(
        "require_user_token: user_tokens table empty, /v1/* endpoints are OPEN. "
        "Mint a token via POST /admin/user-tokens to lock the proxy.",
    )

def get_http_client(request: Request) -> httpx.AsyncClient:
    state: State = request.app.state
    client = getattr(state, "http_client", None)
    if client is not None:
        return client
    raise RuntimeError("HTTP client is not initialised; lifespan did not run.")

def get_ollama_client(
    client: httpx.AsyncClient = Depends(get_http_client),
    settings: Settings = Depends(get_settings_dep),
) -> OllamaClient:
    return OllamaClient(client, settings)

def get_key_manager_dep() -> KeyManager:
    return get_key_manager()

def get_vault_dep() -> Vault:
    return get_vault()

def get_rotation(
    client: httpx.AsyncClient = Depends(get_http_client),
    key_manager: KeyManager = Depends(get_key_manager_dep),
    vault: Vault = Depends(get_vault_dep),
    settings: Settings = Depends(get_settings_dep),
) -> Rotation:
    return Rotation(
        key_manager=key_manager,
        vault=vault,
        ollama_client=OllamaClient(client, settings),
        settings=settings,
    )

def get_model_cache(request: Request, settings: Settings = Depends(get_settings_dep)) -> ModelCache:
    state: State = request.app.state
    cache = getattr(state, "model_cache", None)
    if cache is not None:
        return cache
    # If the lifespan hasn't initialised it (e.g. some test paths),
    # create a throwaway one — still useful, just not shared.
    return ModelCache(ttl_seconds=settings.model_cache_ttl_seconds)

def get_usage_service(
    request: Request,
    ollama: OllamaClient = Depends(get_ollama_client),
    vault: Vault = Depends(get_vault_dep),
    settings: Settings = Depends(get_settings_dep),
) -> UsageService:
    state: State = request.app.state
    svc = getattr(state, "usage_service", None)
    if svc is not None:
        return svc
    # Fallback for tests that don't run the lifespan: spin up a fresh
    # service per request. Still functional, just no shared cache.
    return UsageService(
        ollama=ollama,
        vault=vault,
        ttl_seconds=settings.usage_refresh_interval_seconds,
    )
