from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.logging import logger

T = TypeVar("T")

class ModelCache:

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = max(1, int(ttl_seconds))
        self._lock = asyncio.Lock()
        self._models: list[str] | None = None
        self._last_refresh: float = 0.0

    # ---------------------------------------------------------------- read

    def get(self) -> list[str] | None:
        if self._models is None or not self._models:
            return None
        if (time.monotonic() - self._last_refresh) > self._ttl:
            return None
        return list(self._models)

    def get_stale(self) -> list[str] | None:
        if self._models is None:
            return None
        return list(self._models)

    # -------------------------------------------------------------- write

    async def refresh(self, fetcher: Callable[[], Awaitable[list[str]]]) -> list[str]:
        async with self._lock:
            # Re-check inside the lock: another coroutine may have just
            # refreshed while we were waiting.
            if self._models is not None and self._models and (time.monotonic() - self._last_refresh) <= self._ttl:
                return list(self._models)
            try:
                models = await fetcher()
            except Exception as exc:  # noqa: BLE001
                logger.warning("model_cache: refresh failed: {}", exc)
                # Return whatever we have, even if stale, so the caller
                # can decide whether to fall back.
                return list(self._models) if self._models is not None else []
            # Sanity: dedupe, drop empties, preserve order.
            seen: set[str] = set()
            cleaned: list[str] = []
            for name in models:
                if isinstance(name, str) and name and name not in seen:
                    seen.add(name)
                    cleaned.append(name)
            if not cleaned:
                # Empty result: do NOT poison the cache. Keep whatever we
                # had before, even if stale, so the next call retries.
                logger.warning(
                    "model_cache: upstream returned empty model list; preserving cache"
                )
                return list(self._models) if self._models is not None else []
            self._models = cleaned
            self._last_refresh = time.monotonic()
            logger.info("model_cache: refreshed count={}", len(cleaned))
            return list(cleaned)

    def reset(self) -> None:
        self._models = None
        self._last_refresh = 0.0
