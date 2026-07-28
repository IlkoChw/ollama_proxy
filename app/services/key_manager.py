from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.api_key import ApiKey, ApiKeyStatus

_SESSION_WINDOW = timedelta(hours=24)
_WEEKLY_WINDOW = timedelta(days=7)

def _ensure_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value

class KeyManager:

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._next_idx: int = 0
        self._in_flight: set[int] = set()

    # ------------------------------------------------------------------ select

    async def pick_next_key(self, session: AsyncSession) -> ApiKey | None:
        async with self._lock:
            active = await self._load_active(session)
            if not active:
                await self._reactivate_locked(session)
                active = await self._load_active(session)
            if not active:
                return None

            # Anti-stacking scan: starting from ``_next_idx``, find the
            # first key whose ``id`` is not currently in flight.
            n = len(active)
            start = self._next_idx
            chosen: ApiKey | None = None
            chosen_offset = 0
            for offset in range(n):
                idx = (start + offset) % n
                candidate = active[idx]
                if candidate.id not in self._in_flight:
                    chosen = candidate
                    chosen_offset = offset
                    break

            if chosen is None:
                return None

            # Reserve the key for the duration of the upstream call.
            self._in_flight.add(chosen.id)
            self._next_idx = (start + chosen_offset + 1) % n
        return chosen

    async def release(self, key_id: int) -> None:
        # Lock-free: asyncio is single-threaded, ``set.discard`` is
        # atomic from the event loop's perspective.
        self._in_flight.discard(key_id)

    def in_flight_count(self) -> int:
        return len(self._in_flight)

    async def _load_active(self, session: AsyncSession) -> list[ApiKey]:
        stmt = (
            select(ApiKey)
            .where(ApiKey.status == ApiKeyStatus.ACTIVE.value)
            .order_by(ApiKey.id)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    # ----------------------------------------------------- reactivation / count

    async def reactivate_expired_depleted(self, session: AsyncSession) -> int:
        async with self._lock:
            return await self._reactivate_locked(session)

    async def _reactivate_locked(self, session: AsyncSession) -> int:
        now = datetime.now(UTC)
        stmt = (
            update(ApiKey)
            .where(
                ApiKey.status == ApiKeyStatus.DEPLETED.value,
                ApiKey.cooldown_until.is_not(None),
                ApiKey.cooldown_until <= now,
            )
            .values(status=ApiKeyStatus.ACTIVE.value, cooldown_until=None)
            .execution_options(synchronize_session=False)
        )
        result = await session.execute(stmt)
        await session.commit()
        reactivated = int(result.rowcount or 0)
        if reactivated:
            logger.info("reactivate_expired_depleted: reactivated={}", reactivated)
        return reactivated

    async def count_by_status(self, session: AsyncSession) -> dict[str, int]:
        stmt = select(ApiKey.status)
        result = await session.execute(stmt)
        counts: dict[str, int] = {s.value: 0 for s in ApiKeyStatus}
        for (status,) in result.all():
            counts[status] = counts.get(status, 0) + 1
        return counts

    # -------------------------------------------------------------- classifiers

    async def _update_key(
        self,
        session: AsyncSession,
        key_id: int,
        **values: Any,
    ) -> ApiKey:
        # Reload first so the caller always sees the post-update state.
        key = await session.get(ApiKey, key_id)
        if key is None:
            raise LookupError(f"ApiKey id={key_id} not found")
        for attr, value in values.items():
            setattr(key, attr, value)
        await session.commit()
        await session.refresh(key)
        return key

    async def record_success(
        self, session: AsyncSession, key_id: int, status_code: int
    ) -> ApiKey:
        now = datetime.now(UTC)
        return await self._update_key(
            session,
            key_id,
            status=ApiKeyStatus.ACTIVE.value,
            last_used_at=now,
            last_status_code=status_code,
            cooldown_until=None,
            total_requests=ApiKey.total_requests + 1,
            updated_at=now,
        )

    async def record_success_with_usage(
        self,
        session: AsyncSession,
        key_id: int,
        status_code: int,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> ApiKey:
        now = datetime.now(UTC)
        # Reload once so we see the current window state before mutating.
        key = await session.get(ApiKey, key_id)
        if key is None:
            raise LookupError(f"ApiKey id={key_id} not found")

        if prompt_tokens is not None and prompt_tokens > 0:
            # SQLite drops tz info on read; treat naive datetimes as UTC
            # so the arithmetic below doesn't blow up.
            session_started = _ensure_aware(key.session_window_started_at)
            weekly_started = _ensure_aware(key.weekly_window_started_at)
            if session_started is None or (now - session_started) >= _SESSION_WINDOW:
                key.session_window_started_at = now
                key.session_prompt_tokens = 0
                key.session_completion_tokens = 0
            if weekly_started is None or (now - weekly_started) >= _WEEKLY_WINDOW:
                key.weekly_window_started_at = now
                key.weekly_prompt_tokens = 0
                key.weekly_completion_tokens = 0
            key.session_prompt_tokens += int(prompt_tokens)
            key.weekly_prompt_tokens += int(prompt_tokens)
            if completion_tokens is not None and completion_tokens > 0:
                key.session_completion_tokens += int(completion_tokens)
                key.weekly_completion_tokens += int(completion_tokens)
            key.last_token_at = now

        # Single atomic UPDATE that includes ``ApiKey.total_requests + 1``
        # SQL arithmetic, so concurrent writers can't lose increments.
        stmt = (
            update(ApiKey)
            .where(ApiKey.id == key_id)
            .values(
                status=ApiKeyStatus.ACTIVE.value,
                last_used_at=now,
                last_status_code=status_code,
                cooldown_until=None,
                total_requests=ApiKey.total_requests + 1,
                updated_at=now,
                session_window_started_at=key.session_window_started_at
                if prompt_tokens
                else ApiKey.session_window_started_at,
                session_prompt_tokens=key.session_prompt_tokens,
                session_completion_tokens=key.session_completion_tokens,
                weekly_window_started_at=key.weekly_window_started_at
                if prompt_tokens
                else ApiKey.weekly_window_started_at,
                weekly_prompt_tokens=key.weekly_prompt_tokens,
                weekly_completion_tokens=key.weekly_completion_tokens,
                last_token_at=key.last_token_at if prompt_tokens else ApiKey.last_token_at,
            )
            .execution_options(synchronize_session=False)
        )
        await session.execute(stmt)
        await session.commit()
        return key

    async def record_unauthorized(
        self, session: AsyncSession, key_id: int, status_code: int
    ) -> ApiKey:
        now = datetime.now(UTC)
        return await self._update_key(
            session,
            key_id,
            status=ApiKeyStatus.DISABLED.value,
            last_status_code=status_code,
            cooldown_until=None,
            total_failures=ApiKey.total_failures + 1,
            updated_at=now,
        )

    async def record_rate_limited(
        self,
        session: AsyncSession,
        key_id: int,
        status_code: int,
        retry_after_seconds: int,
    ) -> ApiKey:
        now = datetime.now(UTC)
        cooldown = now + timedelta(seconds=max(1, retry_after_seconds))
        return await self._update_key(
            session,
            key_id,
            status=ApiKeyStatus.DEPLETED.value,
            last_status_code=status_code,
            cooldown_until=cooldown,
            total_failures=ApiKey.total_failures + 1,
            updated_at=now,
        )

    async def record_server_error(
        self, session: AsyncSession, key_id: int, status_code: int
    ) -> ApiKey:
        now = datetime.now(UTC)
        return await self._update_key(
            session,
            key_id,
            last_status_code=status_code,
            total_failures=ApiKey.total_failures + 1,
            updated_at=now,
        )

    # ----------------------------------------------------------------- utility

    @staticmethod
    def extract_retry_after(headers: Iterable[tuple[str, str]] | Any) -> int:
        if headers is None:
            return 60

        # Path 1: httpx.Headers-like or any object with case-insensitive .get
        get_lower: Any = None
        if hasattr(headers, "get"):
            get_lower = headers.get
            for name in ("retry-after", "Retry-After", "RETRY-AFTER"):
                try:
                    raw = get_lower(name)
                except Exception:  # noqa: BLE001
                    raw = None
                if raw is not None:
                    return _parse_retry_after_value(raw)

        # Path 2: list of tuples (case-insensitive match)
        if isinstance(headers, list | tuple):
            for entry in headers:
                if not isinstance(entry, tuple) or len(entry) < 2:
                    continue
                name, value = entry[0], entry[1]
                if isinstance(name, str) and name.lower() == "retry-after":
                    return _parse_retry_after_value(value)

        # Path 3: dict-like
        if hasattr(headers, "items"):
            try:
                items = headers.items()
            except Exception:  # noqa: BLE001
                items = ()
            for name, value in items:
                if isinstance(name, str) and name.lower() == "retry-after":
                    return _parse_retry_after_value(value)

        return 60

def _parse_retry_after_value(raw: Any) -> int:
    if raw is None:
        return 60
    try:
        return max(1, int(float(str(raw))))
    except (TypeError, ValueError):
        return 60

# A process-wide singleton, instantiated on first use.
_manager: KeyManager | None = None

def get_key_manager() -> KeyManager:
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = KeyManager()
    return _manager

def reset_key_manager_for_tests() -> None:
    global _manager  # noqa: PLW0603
    _manager = None
