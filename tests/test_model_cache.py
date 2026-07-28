"""Tests for the in-memory ModelCache (TTL, concurrency, fallback)."""

from __future__ import annotations

import asyncio

import pytest

from app.services.model_cache import ModelCache


async def test_get_returns_none_when_empty() -> None:
    cache = ModelCache(ttl_seconds=300)
    assert cache.get() is None
    assert cache.get_stale() is None


async def test_refresh_stores_data_and_get_returns_it() -> None:
    cache = ModelCache(ttl_seconds=300)
    result = await cache.refresh(lambda: _async(["minimax-m3", "minimax-m2.7"]))
    assert result == ["minimax-m3", "minimax-m2.7"]
    assert cache.get() == ["minimax-m3", "minimax-m2.7"]


async def test_get_returns_none_when_cached_list_is_empty() -> None:
    """An empty upstream response is treated as 'no data' — the cache
    is not populated, so subsequent calls will retry the fetch instead
    of returning an empty list for the full TTL.
    """
    cache = ModelCache(ttl_seconds=300)
    result = await cache.refresh(lambda: _async([]))
    # refresh returns [] because nothing was cached before.
    assert result == []
    assert cache.get() is None
    # get_stale() also returns None — no data to fall back on.
    assert cache.get_stale() is None


async def test_empty_refresh_does_not_overwrite_existing_data() -> None:
    """A subsequent empty fetch must not clobber a previously good cache."""
    cache = ModelCache(ttl_seconds=300)
    await cache.refresh(lambda: _async(["a", "b"]))
    assert cache.get() == ["a", "b"]

    # Now imagine a transient upstream hiccup returning empty.
    result = await cache.refresh(lambda: _async([]))
    # Result is whatever the cache had before.
    assert result == ["a", "b"]
    assert cache.get() == ["a", "b"]


async def test_refresh_dedupes_and_drops_empty() -> None:
    cache = ModelCache(ttl_seconds=300)
    await cache.refresh(lambda: _async(["a", "b", "a", "", "c"]))
    assert cache.get() == ["a", "b", "c"]


async def test_get_returns_none_after_ttl_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ModelCache(ttl_seconds=10)
    await cache.refresh(lambda: _async(["x"]))
    # Move monotonic clock forward past the TTL.
    base = cache._last_refresh  # noqa: SLF001 — internal but acceptable in tests
    monkeypatch.setattr(
        "app.services.model_cache.time.monotonic", lambda: base + 11.0
    )
    assert cache.get() is None
    # Stale is still available.
    assert cache.get_stale() == ["x"]


async def test_refresh_returns_stale_on_error() -> None:
    cache = ModelCache(ttl_seconds=300)
    await cache.refresh(lambda: _async(["a", "b"]))

    async def _boom() -> list[str]:
        raise RuntimeError("upstream down")

    result = await cache.refresh(_boom)
    # Refresh failed, but we still return whatever was cached.
    assert result == ["a", "b"]


async def test_refresh_returns_empty_when_no_prior_data_and_error() -> None:
    cache = ModelCache(ttl_seconds=300)

    async def _boom() -> list[str]:
        raise RuntimeError("upstream down")

    result = await cache.refresh(_boom)
    assert result == []


async def test_concurrent_refresh_issues_one_fetch() -> None:
    """10 concurrent refreshes should result in only 1 actual fetcher call."""
    cache = ModelCache(ttl_seconds=300)
    call_count = 0

    async def _slow_fetch() -> list[str]:
        nonlocal call_count
        call_count += 1
        # Yield so other coroutines pile up on the lock.
        await asyncio.sleep(0.05)
        return ["a", "b"]

    results = await asyncio.gather(*(cache.refresh(_slow_fetch) for _ in range(10)))
    assert call_count == 1
    # All callers receive the same data.
    for r in results:
        assert r == ["a", "b"]


async def test_concurrent_refresh_with_fresh_cache_skips_fetch() -> None:
    """If the cache is already fresh, concurrent refresh is a no-op."""
    cache = ModelCache(ttl_seconds=300)
    await cache.refresh(lambda: _async(["a"]))

    call_count = 0

    async def _would_fetch() -> list[str]:
        nonlocal call_count
        call_count += 1
        return ["should not be called"]

    results = await asyncio.gather(*(cache.refresh(_would_fetch) for _ in range(5)))
    assert call_count == 0
    for r in results:
        assert r == ["a"]


def _async(value: list[str]):
    """Helper: turn a list literal into an awaitable that returns it."""

    async def _f() -> list[str]:
        return value

    return _f()
