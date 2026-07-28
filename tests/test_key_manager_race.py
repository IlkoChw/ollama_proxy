"""Lock-contract tests for :class:`KeyManager`.

These tests verify the invariant that ``pick_next_key`` and
``reactivate_expired_depleted`` hold ``self._lock`` across the entire
selection / reactivation path — not just the ``_next_idx`` increment.

The lock is non-reentrant (``asyncio.Lock``), so any regression that
introduces a nested acquire would deadlock the test event loop. We
catch that with a short timeout on the gather.

Anti-stacking
=============

The bottom of this file exercises the ``_in_flight`` reservation set
introduced for the anti-stacking rotation policy. The tests check:

* ``pick_next_key`` skips keys that are currently in-flight.
* ``pick_next_key`` returns ``None`` when every active key is in-flight.
* ``release`` makes a reserved key available again.
* Concurrent pickers never receive the same ``key_id``.
* The in-flight set is empty after a ``try/finally``-style release path.

The tests rely on ``db_session`` from ``tests/conftest.py``.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey, ApiKeyStatus
from app.services.key_manager import KeyManager


async def _add(session: AsyncSession, label: str, status: str = "active") -> ApiKey:
    """Create + persist a fresh ApiKey for use in tests."""
    from app.services.vault import get_vault

    fake_raw = f"sk-{label}-raw"
    key = ApiKey(
        key_hash=f"{label}_hash".ljust(64, "0")[:64],
        key_prefix=label[:8],
        key_encrypted=get_vault().encrypt(fake_raw),
        label=label,
        status=status,
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)
    return key


# ---------------------------------------------------------- lock acquisition


async def test_pick_next_key_holds_lock_for_whole_round(db_session: AsyncSession) -> None:
    """``pick_next_key`` acquires ``self._lock`` exactly once and holds it
    across ``_load_active`` / ``_reactivate_locked`` / idx computation."""
    km = KeyManager()
    await _add(db_session, "alpha")
    await _add(db_session, "beta")

    acquires = 0
    releases = 0
    original_acquire = km._lock.acquire
    original_release = km._lock.release

    def _track_acquire() -> asyncio.Future:  # type: ignore[override]
        nonlocal acquires
        acquires += 1
        return original_acquire()

    def _track_release() -> None:
        nonlocal releases
        releases += 1
        original_release()

    km._lock.acquire = _track_acquire  # type: ignore[method-assign]
    km._lock.release = _track_release  # type: ignore[method-assign]

    try:
        picked = await km.pick_next_key(db_session)
    finally:
        km._lock.acquire = original_acquire  # type: ignore[method-assign]
        km._lock.release = original_release  # type: ignore[method-assign]

    assert picked is not None
    # Exactly one acquire / release per call, covering the whole round.
    assert acquires == 1, f"expected 1 acquire, got {acquires}"
    assert releases == 1, f"expected 1 release, got {releases}"


async def test_reactivate_expired_depleted_holds_lock(db_session: AsyncSession) -> None:
    """Standalone ``reactivate_expired_depleted`` takes the lock once."""
    km = KeyManager()
    await _add(db_session, "alpha", status="depleted")
    db_session.expire_all()
    key = await db_session.get(ApiKey, 1)
    key.cooldown_until = datetime.now(UTC) - timedelta(seconds=5)
    await db_session.commit()

    acquires = 0
    original_acquire = km._lock.acquire

    def _track() -> asyncio.Future:  # type: ignore[override]
        nonlocal acquires
        acquires += 1
        return original_acquire()

    km._lock.acquire = _track  # type: ignore[method-assign]

    try:
        reactivated = await km.reactivate_expired_depleted(db_session)
    finally:
        km._lock.acquire = original_acquire  # type: ignore[method-assign]

    assert reactivated == 1
    assert acquires == 1, f"expected 1 acquire, got {acquires}"


# --------------------------------------------- concurrent pick + reactivation


async def test_concurrent_picks_dont_duplicate_reactivation(
    db_session: AsyncSession,
) -> None:
    """10 concurrent ``pick_next_key`` on a pool of 5 expired-depleted keys.

    The pool has 5 keys; with anti-stacking the first 5 picks each get
    a distinct key, the remaining 5 hit saturation and return ``None``.
    The original purpose of this test is preserved: no deadlock from a
    non-reentrant lock, the pool is reactived exactly once, and every
    picked id comes from the seeded set.
    """
    km = KeyManager()

    # Seed 5 expired-depleted keys.
    expired = []
    for label in ("a", "b", "c", "d", "e"):
        k = await _add(db_session, label, status="depleted")
        k.cooldown_until = datetime.now(UTC) - timedelta(seconds=1)
        expired.append(k)
    await db_session.commit()

    picked_ids: list[int] = []
    none_count = 0

    async def _pick() -> None:
        nonlocal none_count
        picked = await km.pick_next_key(db_session)
        if picked is None:
            none_count += 1
        else:
            picked_ids.append(picked.id)

    # 10 concurrent calls, each guarded by a 2s timeout — deadlocks would
    # trigger the timeout instead of stalling the test runner.
    await asyncio.wait_for(
        asyncio.gather(*(_pick() for _ in range(10))), timeout=2.0
    )

    # Anti-stacking: 5 picks get keys, 5 hit saturation.
    assert len(picked_ids) == 5
    assert none_count == 5
    # All picks are over the same 5 reactivated ids — no spurious rows.
    assert set(picked_ids) == {k.id for k in expired}
    # And every active key is now in-flight (anti-stacking reservation).
    assert km.in_flight_count() == 5


async def test_pick_next_key_pool_grows_after_reactivation(
    db_session: AsyncSession,
) -> None:
    """Pool starts empty (only expired-depleted); after reactivation,
    ``_next_idx`` is computed against the new size — index resets
    cleanly to 0 even when several pending callers race on the same
    empty pool.

    With anti-stacking the first 2 concurrent calls each get a distinct
    reactivated key, and any remaining calls hit saturation (``None``).
    The original guarantee — no deadlock, no crash — is preserved.
    """
    km = KeyManager()

    # Seed 2 expired-depleted keys (no active). The first concurrent
    # pick_next_key triggers reactivation; the second sees the now-2-key
    # pool and picks the next slot.
    keys = []
    for label in ("exp1", "exp2"):
        k = await _add(db_session, label, status="depleted")
        k.cooldown_until = datetime.now(UTC) - timedelta(seconds=1)
        keys.append(k)
    await db_session.commit()

    # 4 concurrent calls over a 2-key pool: 2 get keys, 2 hit saturation.
    results = await asyncio.wait_for(
        asyncio.gather(*(km.pick_next_key(db_session) for _ in range(4))),
        timeout=2.0,
    )

    picked = [r for r in results if r is not None]
    assert len(picked) == 2
    # The two picked ids must be distinct (anti-stacking guarantee).
    assert picked[0].id != picked[1].id
    # And both come from the reactivated set.
    assert {p.id for p in picked} == {k.id for k in keys}


# --------------------------------------------------------------- reentrancy


async def test_lock_is_not_reentrant(db_session: AsyncSession) -> None:
    """Smoke check: ``pick_next_key`` must NOT call ``reactivate_expired_depleted``
    while holding the lock, or the non-reentrant ``asyncio.Lock`` would
    deadlock. We trigger the reactivation path (all active gone) and
    assert the call returns within a tight timeout."""
    km = KeyManager()
    started = time.monotonic()

    # Empty pool → reactivation path is taken.
    result = await asyncio.wait_for(
        km.pick_next_key(db_session), timeout=1.0
    )
    elapsed = time.monotonic() - started

    assert result is None
    assert elapsed < 0.5, f"reactivation took {elapsed:.2f}s — possible deadlock"


def test_key_manager_status_enum_unchanged() -> None:
    """Spot check: the public surface of :class:`KeyManager` did not lose
    any methods during the lock refactor."""
    expected = {
        "pick_next_key",
        "reactivate_expired_depleted",
        "count_by_status",
        "record_success",
        "record_unauthorized",
        "record_rate_limited",
        "record_server_error",
        "extract_retry_after",
        "release",
        "in_flight_count",
    }
    public = {
        name
        for name in dir(KeyManager)
        if not name.startswith("__") and not name.startswith("_")
    }
    assert expected.issubset(public), f"missing: {expected - public}"


# ----------------------------------------------------------- count_by_status


async def test_count_by_status_does_not_block_under_load(
    db_session: AsyncSession,
) -> None:
    """``count_by_status`` is read-only and intentionally does not take
    ``self._lock``. Two concurrent calls cannot deadlock."""
    km = KeyManager()
    await _add(db_session, "a1", status=ApiKeyStatus.ACTIVE.value)
    await _add(db_session, "d1", status=ApiKeyStatus.DEPLETED.value)
    await _add(db_session, "x1", status=ApiKeyStatus.DISABLED.value)

    counts_a, counts_b = await asyncio.wait_for(
        asyncio.gather(km.count_by_status(db_session), km.count_by_status(db_session)),
        timeout=1.0,
    )
    assert counts_a == counts_b
    assert counts_a["active"] == 1
    assert counts_a["depleted"] == 1
    assert counts_a["disabled"] == 1


@pytest.mark.asyncio
async def test_pick_next_key_lock_released_on_db_error(db_session: AsyncSession) -> None:
    """If the DB raises inside the lock (e.g. broken session), the lock
    must be released. Otherwise the manager would deadlock every future
    request — silent production footgun.
    """
    km = KeyManager()

    class _BrokenSession:
        async def execute(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        async def commit(self) -> None:
            return None

    # First call — fails inside the lock body.
    with pytest.raises(RuntimeError, match="boom"):
        await km.pick_next_key(_BrokenSession())  # type: ignore[arg-type]

    # Lock must NOT be held now. Second normal call on a fresh pool
    # should succeed instantly.
    await _add(db_session, "fresh")
    started = time.monotonic()
    picked = await asyncio.wait_for(km.pick_next_key(db_session), timeout=0.5)
    elapsed = time.monotonic() - started

    assert picked is not None
    assert elapsed < 0.2, f"second pick took {elapsed:.2f}s — lock leaked"


# =====================================================================
# Anti-stacking: per-key concurrency = 1
# =====================================================================


async def test_pick_next_key_marks_in_flight(db_session: AsyncSession) -> None:
    """``pick_next_key`` reserves the chosen key by adding its id to
    ``_in_flight``. ``release`` removes it. The pair is the contract
    that the rotation loop relies on."""
    km = KeyManager()
    await _add(db_session, "alpha")
    await _add(db_session, "beta")

    assert km.in_flight_count() == 0
    key = await km.pick_next_key(db_session)
    assert key is not None
    assert km.in_flight_count() == 1
    assert key.id in km._in_flight  # type: ignore[attr-defined]

    await km.release(key.id)
    assert km.in_flight_count() == 0
    assert key.id not in km._in_flight  # type: ignore[attr-defined]


async def test_pick_next_key_skips_in_flight(db_session: AsyncSession) -> None:
    """When the round-robin pointer lands on a reserved key, the
    scanner must skip it and pick the next free key."""
    km = KeyManager()
    await _add(db_session, "alpha")
    await _add(db_session, "beta")
    await _add(db_session, "gamma")

    # Manually reserve id=1 (alpha) — pick_next_key should skip it.
    alpha = await db_session.get(ApiKey, 1)
    assert alpha is not None
    km._in_flight.add(alpha.id)  # type: ignore[attr-defined]

    picked = await km.pick_next_key(db_session)
    assert picked is not None
    assert picked.id != alpha.id
    # The picked key itself is now also in-flight.
    assert km.in_flight_count() == 2


async def test_pick_next_key_returns_none_when_all_in_flight(
    db_session: AsyncSession,
) -> None:
    """If every active key is reserved, ``pick_next_key`` returns
    ``None``. This is the anti-stacking saturation signal."""
    km = KeyManager()
    a = await _add(db_session, "a")
    b = await _add(db_session, "b")

    km._in_flight.update([a.id, b.id])  # type: ignore[attr-defined]
    picked = await km.pick_next_key(db_session)
    assert picked is None


async def test_release_is_idempotent(db_session: AsyncSession) -> None:
    """Calling ``release`` twice (or for a never-picked id) must not
    raise. ``set.discard`` is silent on missing entries."""
    km = KeyManager()
    await km.release(99999)  # never picked
    await _add(db_session, "k")
    key = await km.pick_next_key(db_session)
    assert key is not None
    await km.release(key.id)
    await km.release(key.id)  # second call: no-op
    assert km.in_flight_count() == 0


async def test_concurrent_picks_never_duplicate(
    db_session: AsyncSession,
) -> None:
    """``asyncio.gather`` of N pickers over a pool of size M must yield
    at most M distinct ``key_id`` values. This is the central
    anti-stacking invariant."""
    km = KeyManager()
    keys = [await _add(db_session, f"k{i}") for i in range(3)]

    # Pick 5 times concurrently — pool has only 3 keys, so at least
    # two callers should hit saturation and return None.
    results = await asyncio.wait_for(
        asyncio.gather(*(km.pick_next_key(db_session) for _ in range(5))),
        timeout=2.0,
    )

    picked_ids = [r.id for r in results if r is not None]
    none_count = sum(1 for r in results if r is None)

    # Every picked id is one of the three known keys.
    assert set(picked_ids) <= {k.id for k in keys}
    # No duplicates — anti-stacking is the central guarantee here.
    assert len(picked_ids) == len(set(picked_ids))
    # Saturation observed (5 picks on a 3-key pool).
    assert none_count == 2
    assert km.in_flight_count() == 3


async def test_release_after_pick_makes_key_available_again(
    db_session: AsyncSession,
) -> None:
    """End-to-end: pick → release → pick again should return a key
    (could be the same id, could be the next round-robin slot — but
    the pool is non-empty)."""
    km = KeyManager()
    await _add(db_session, "alpha")

    first = await km.pick_next_key(db_session)
    assert first is not None
    await km.release(first.id)
    assert km.in_flight_count() == 0

    second = await km.pick_next_key(db_session)
    assert second is not None
    assert km.in_flight_count() == 1


async def test_release_in_finally_unblocks_saturation(
    db_session: AsyncSession,
) -> None:
    """Simulate a caller that releases its key in a ``finally`` block
    (the canonical rotation pattern). After the release, the next
    ``pick_next_key`` should find a free slot even if the previous
    state was fully saturated."""
    km = KeyManager()
    a = await _add(db_session, "a")
    b = await _add(db_session, "b")

    # Saturate the pool.
    km._in_flight.update([a.id, b.id])  # type: ignore[attr-defined]
    assert await km.pick_next_key(db_session) is None

    # Simulate a finished upstream call releasing one key.
    await km.release(a.id)
    assert km.in_flight_count() == 1

    # Next picker must find the freed slot.
    picked = await km.pick_next_key(db_session)
    assert picked is not None
    assert picked.id == a.id
    assert km.in_flight_count() == 2


async def test_pick_next_key_does_not_mutate_in_flight_on_no_active(
    db_session: AsyncSession,
) -> None:
    """When the pool is empty (no active keys), ``pick_next_key``
    returns ``None`` and leaves ``_in_flight`` untouched."""
    km = KeyManager()
    picked = await km.pick_next_key(db_session)
    assert picked is None
    assert km.in_flight_count() == 0
