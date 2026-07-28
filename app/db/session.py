from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import logger
from app.db.base import Base

_engine: AsyncEngine | None = None
_AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None

def _build_db_url(db_path: str) -> str:
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{path.as_posix()}"

async def init_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine, _AsyncSessionLocal  # noqa: PLW0603
    if _engine is not None:
        return _engine

    cfg = settings or get_settings()
    url = _build_db_url(cfg.db_path)
    logger.info("init_engine: db={}", url)

    _engine = create_async_engine(
        url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
    )
    _AsyncSessionLocal = async_sessionmaker(
        _engine, expire_on_commit=False, class_=AsyncSession
    )

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _ensure_usage_columns(conn)

    return _engine

_USAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("account_email", "VARCHAR(255)"),
    ("account_name", "VARCHAR(255)"),
    ("account_plan", "VARCHAR(32)"),
    ("account_id", "VARCHAR(64)"),
    ("last_usage_fetch_at", "TIMESTAMP"),
    ("last_usage_session", "INTEGER"),
    ("last_usage_weekly", "INTEGER"),
    ("last_usage_models_json", "TEXT"),
    ("session_prompt_tokens", "INTEGER DEFAULT 0 NOT NULL"),
    ("session_completion_tokens", "INTEGER DEFAULT 0 NOT NULL"),
    ("session_window_started_at", "TIMESTAMP"),
    ("weekly_prompt_tokens", "INTEGER DEFAULT 0 NOT NULL"),
    ("weekly_completion_tokens", "INTEGER DEFAULT 0 NOT NULL"),
    ("weekly_window_started_at", "TIMESTAMP"),
    ("last_token_at", "TIMESTAMP"),
    ("last_usage_session_fraction", "REAL"),
    ("last_usage_weekly_fraction", "REAL"),
)

async def _ensure_usage_columns(conn: Any) -> None:
    def _has_column(sync_conn: Any) -> set[str]:
        inspector = inspect(sync_conn)
        if "api_keys" not in inspector.get_table_names():
            return set()
        return {c["name"] for c in inspector.get_columns("api_keys")}

    existing = await conn.run_sync(_has_column)
    missing = [(name, ddl) for name, ddl in _USAGE_COLUMNS if name not in existing]
    if not missing:
        return
    for name, ddl in missing:
        logger.info("migration: api_keys.ADD COLUMN {} {}", name, ddl)
        await conn.execute(
            text(f"ALTER TABLE api_keys ADD COLUMN {name} {ddl}")
        )

async def dispose_engine() -> None:
    global _engine, _AsyncSessionLocal  # noqa: PLW0603
    if _engine is not None:
        logger.info("dispose_engine")
        await _engine.dispose()
    _engine = None
    _AsyncSessionLocal = None

def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Database engine is not initialised. Call init_engine() first.")
    return _engine

def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _AsyncSessionLocal is None:
        raise RuntimeError(
            "AsyncSession factory is not initialised. Call init_engine() first."
        )
    return _AsyncSessionLocal

async def get_session() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

def override_session_factory(new_factory: async_sessionmaker[AsyncSession]) -> None:
    global _AsyncSessionLocal  # noqa: PLW0603
    _AsyncSessionLocal = new_factory

def reset_for_tests() -> None:
    global _engine, _AsyncSessionLocal  # noqa: PLW0603
    _engine = None
    _AsyncSessionLocal = None
