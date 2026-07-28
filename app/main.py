from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import State

from app import models as _models  # noqa: F401
from app.api.v1.admin import health_router
from app.api.v1.admin import router as admin_router
from app.api.v1.admin_user_tokens import router as admin_user_tokens_router
from app.api.v1.proxy import router as proxy_router
from app.core.config import Settings, get_settings
from app.core.dashboard_filters import register_template_filters
from app.core.logging import logger
from app.db.session import dispose_engine, get_session_factory, init_engine
from app.services.dashboard_auth import DashboardAuth, set_dashboard_auth
from app.services.model_cache import ModelCache
from app.services.ollama_client import OllamaClient
from app.services.usage_service import UsageService
from app.services.vault import build_vault, get_vault, set_vault

_settings: Settings = get_settings()

@asynccontextmanager
async def lifespan(application: FastAPI):
    settings: Settings = get_settings()
    logger.info(
        "startup: ollama_base_url={} dashboard={}",
        settings.ollama_base_url,
        "on" if settings.enable_dashboard else "off",
    )

    # Build the encryption vault from the configured db_path so the
    # master key file ends up next to the SQLite DB.
    set_vault(build_vault(db_path=settings.db_path))

    engine = await init_engine(settings)
    state: State = application.state
    state.engine = engine
    state.http_client = httpx.AsyncClient(timeout=settings.timeout)
    state.model_cache = ModelCache(ttl_seconds=settings.model_cache_ttl_seconds)
    state.usage_service = UsageService(
        ollama=OllamaClient(state.http_client, settings),
        vault=get_vault(),
        ttl_seconds=settings.usage_refresh_interval_seconds,
    )

    dashboard_enabled = settings.enable_dashboard
    if dashboard_enabled:
        missing: list[str] = []
        if not settings.dashboard_password:
            missing.append("DASHBOARD_PASSWORD")
        if not settings.dashboard_session_secret:
            missing.append("DASHBOARD_SESSION_SECRET")
        if missing:
            raise RuntimeError(
                "dashboard cannot start: missing required env vars: "
                + ", ".join(missing)
                + ". Set them in docker-compose.yml or .env."
            )

        set_dashboard_auth(
            DashboardAuth(
                secret=settings.dashboard_session_secret,
                session_ttl_seconds=settings.dashboard_session_ttl_seconds,
                cookie_secure=settings.dashboard_cookie_secure,
            )
        )

        templates_dir = Path(settings.dashboard_templates_dir)
        state.templates = Jinja2Templates(directory=str(templates_dir))
        state.templates.env.auto_reload = True
        register_template_filters(state.templates)

    # Periodic /api/usage refresh task. Started AFTER the engine so the
    # very first iteration has a real DB session to write to.
    refresh_task: asyncio.Task[None] | None = None
    if settings.usage_refresh_interval_seconds > 0:
        refresh_task = asyncio.create_task(
            _usage_refresh_loop(state.usage_service, settings.usage_refresh_interval_seconds),
            name="usage-refresh",
        )

    try:
        yield
    finally:
        if refresh_task is not None:
            refresh_task.cancel()
            try:
                await refresh_task
            except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
                logger.info("usage-refresh: cancelled ({})", exc)
        if dashboard_enabled:
            set_dashboard_auth(None)
        client: httpx.AsyncClient | None = getattr(state, "http_client", None)
        if client is not None:
            await client.aclose()
        await dispose_engine()
        logger.info("shutdown: complete")

async def _usage_refresh_loop(
    usage: UsageService,
    interval_seconds: int,
) -> None:
    factory = get_session_factory()
    logger.info("usage-refresh: loop starting interval={}s", interval_seconds)
    while True:
        try:
            async with factory() as session:
                await usage.refresh_all_active(session)
        except asyncio.CancelledError:
            logger.info("usage-refresh: loop cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("usage-refresh: cycle failed err={}", exc)
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("usage-refresh: loop cancelled during sleep")
            raise

app = FastAPI(
    title="ollama_proxy",
    version="0.1.0",
    description=(
        "FastAPI proxy for ollama.com with API-key rotation. "
        "See ``AGENTS.md`` for the full architecture overview."
    ),
    lifespan=lifespan,
    docs_url="/docs" if _settings.enable_docs else None,
    redoc_url="/redoc" if _settings.enable_docs else None,
    openapi_url="/openapi.json" if _settings.enable_docs else None,
)

_cors_origins_raw = get_settings().cors_allow_origins.strip()
if _cors_origins_raw:
    if _cors_origins_raw == "*":
        _origins: list[str] = ["*"]
    else:
        _origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        # ``*`` is incompatible with credentials per the CORS spec; force
        # credentials off whenever the operator has wildcarded origins.
        allow_credentials=(_origins != ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    logger.info("startup: CORS middleware disabled (CORS_ALLOW_ORIGINS not set)")

# Routers — prefixes are baked into the router definitions to keep
# endpoint paths self-documenting in Swagger.
app.include_router(proxy_router)
app.include_router(admin_router, prefix="/admin")
app.include_router(admin_user_tokens_router)
app.include_router(health_router)

if _settings.enable_dashboard:
    from app.api.dashboard_routes import router as dashboard_router  # noqa: E402

    app.include_router(dashboard_router)

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("startup: dashboard mounted at /dashboard")

@app.get("/", tags=["health"], include_in_schema=False)
async def root() -> dict:
    return {
        "service": "ollama_proxy",
        "version": app.version,
        "docs": "/docs" if _settings.enable_docs else None,
        "health": "/healthz",
    }

@app.get("/healthz", tags=["health"], include_in_schema=False)
async def healthz() -> Response:
    from fastapi.responses import JSONResponse

    from app.services.vault import get_vault, is_vault_initialised

    state: State = app.state
    engine_ready = getattr(state, "engine", None) is not None
    vault_ready = get_vault().is_persistent or is_vault_initialised()
    if engine_ready and vault_ready:
        return JSONResponse(
            {"status": "ok", "timestamp": datetime.now(UTC).isoformat()},
            status_code=200,
        )
    return JSONResponse(
        {"status": "starting", "timestamp": datetime.now(UTC).isoformat()},
        status_code=503,
    )
