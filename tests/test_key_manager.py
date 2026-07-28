"""Tests for KeyManager: round-robin, status classification, reactivation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey, ApiKeyStatus
from app.services.key_manager import KeyManager


def _make_key(label: str, status: str = ApiKeyStatus.ACTIVE.value) -> ApiKey:
    """Build a transient ApiKey with a deterministic hash so tests are stable."""
    return ApiKey(
        key_hash=ApiKey.__table__.c.key_hash.type.python_type(
            f"{label}_hash"
        ),  # placeholder; overridden below
        key_prefix=label[:8],
        label=label,
        status=status,
    )


async def _add(session: AsyncSession, label: str, status: str = "active") -> ApiKey:
    from app.services.vault import get_vault

    fake_raw = f"sk-{label}-raw"
    k = ApiKey(
        key_hash=f"{label}_hash".ljust(64, "0")[:64],
        key_prefix=label[:8],
        key_encrypted=get_vault().encrypt(fake_raw),
        label=label,
        status=status,
    )
    session.add(k)
    await session.commit()
    await session.refresh(k)
    return k


# ----------------------------------------------------------------- round robin


async def test_pick_next_key_round_robin(db_session: AsyncSession) -> None:
    km = KeyManager()
    k1 = await _add(db_session, "alpha")
    k2 = await _add(db_session, "beta")
    k3 = await _add(db_session, "gamma")

    # Anti-stacking: release after each pick so the next pick can take
    # the next slot in the round-robin. Without ``release`` the pool
    # would saturate after the 3rd pick and subsequent calls would
    # return ``None`` (which is the correct anti-stacking behaviour).
    seen: list[int] = []
    for _ in range(6):
        picked = await km.pick_next_key(db_session)
        assert picked is not None
        seen.append(picked.id)
        await km.release(picked.id)

    # Round-robin over [k1, k2, k3] twice: [1, 2, 3, 1, 2, 3]
    assert seen == [k1.id, k2.id, k3.id, k1.id, k2.id, k3.id]


async def test_pick_next_key_no_active_returns_none(db_session: AsyncSession) -> None:
    km = KeyManager()
    await _add(db_session, "alpha", status="disabled")
    await _add(db_session, "beta", status="depleted")

    picked = await km.pick_next_key(db_session)
    assert picked is None


async def test_pick_next_key_skips_disabled(db_session: AsyncSession) -> None:
    km = KeyManager()
    k1 = await _add(db_session, "alpha")
    k2 = await _add(db_session, "beta", status="disabled")
    k3 = await _add(db_session, "gamma")

    # Anti-stacking: release after each pick so we can verify both
    # active keys are reachable. The disabled key is filtered by the
    # ``_load_active`` query, not by the in-flight scan.
    picked_ids = set()
    for _ in range(2):
        picked = await km.pick_next_key(db_session)
        assert picked is not None
        picked_ids.add(picked.id)
        await km.release(picked.id)

    assert picked_ids == {k1.id, k3.id}
    assert k2.id not in picked_ids


# ------------------------------------------------------ classification codes


async def test_record_success_updates_counters(db_session: AsyncSession) -> None:
    km = KeyManager()
    k = await _add(db_session, "alpha")
    await km.record_success(db_session, k.id, 200)
    refreshed = await db_session.get(ApiKey, k.id)
    assert refreshed is not None
    assert refreshed.status == "active"
    assert refreshed.last_status_code == 200
    assert refreshed.total_requests == 1
    assert refreshed.last_used_at is not None
    assert refreshed.cooldown_until is None


async def test_record_unauthorized_disables_key(db_session: AsyncSession) -> None:
    km = KeyManager()
    k = await _add(db_session, "alpha")
    await km.record_unauthorized(db_session, k.id, 401)
    refreshed = await db_session.get(ApiKey, k.id)
    assert refreshed is not None
    assert refreshed.status == "disabled"
    assert refreshed.total_failures == 1
    assert refreshed.cooldown_until is None


async def test_record_rate_limited_marks_depleted_with_cooldown(
    db_session: AsyncSession,
) -> None:
    km = KeyManager()
    k = await _add(db_session, "alpha")
    await km.record_rate_limited(db_session, k.id, 429, retry_after_seconds=120)
    refreshed = await db_session.get(ApiKey, k.id)
    assert refreshed is not None
    assert refreshed.status == "depleted"
    assert refreshed.cooldown_until is not None
    cooldown = refreshed.cooldown_until
    # SQLite may return naive datetimes (no tzinfo). Compare in UTC; if
    # the stored value is naive we strip tz from our reference too.
    now = datetime.now(UTC)
    if cooldown.tzinfo is None:
        now = now.replace(tzinfo=None)
    assert cooldown > now
    assert cooldown <= now + timedelta(seconds=130)


async def test_record_server_error_keeps_active(db_session: AsyncSession) -> None:
    km = KeyManager()
    k = await _add(db_session, "alpha")
    await km.record_server_error(db_session, k.id, 503)
    refreshed = await db_session.get(ApiKey, k.id)
    assert refreshed is not None
    assert refreshed.status == "active"
    assert refreshed.last_status_code == 503
    assert refreshed.total_failures == 1


# --------------------------------------------------------- reactivation


async def test_reactivate_expired_depleted_moves_to_active(
    db_session: AsyncSession,
) -> None:
    km = KeyManager()
    past = datetime.now(UTC) - timedelta(seconds=10)
    k = await _add(db_session, "alpha", status="depleted")
    k.cooldown_until = past
    await db_session.commit()

    reactivated = await km.reactivate_expired_depleted(db_session)
    assert reactivated == 1

    picked = await km.pick_next_key(db_session)
    assert picked is not None
    assert picked.id == k.id


async def test_reactivate_skips_keys_with_future_cooldown(
    db_session: AsyncSession,
) -> None:
    km = KeyManager()
    future = datetime.now(UTC) + timedelta(minutes=10)
    await _add(db_session, "alpha", status="depleted")
    db_session.expire_all()
    k = await db_session.get(ApiKey, 1)
    k.cooldown_until = future
    await db_session.commit()

    reactivated = await km.reactivate_expired_depleted(db_session)
    assert reactivated == 0


async def test_pick_next_key_falls_back_to_depleted_after_reactivation(
    db_session: AsyncSession,
) -> None:
    """All active gone, depleted with expired cooldown → still returns a key."""
    km = KeyManager()
    k = await _add(db_session, "alpha", status="depleted")
    k.cooldown_until = datetime.now(UTC) - timedelta(seconds=5)
    await db_session.commit()

    picked = await km.pick_next_key(db_session)
    assert picked is not None
    assert picked.id == k.id


# ----------------------------------------------------------------- counters


async def test_count_by_status(db_session: AsyncSession) -> None:
    km = KeyManager()
    await _add(db_session, "a1", status="active")
    await _add(db_session, "a2", status="active")
    await _add(db_session, "d1", status="depleted")
    await _add(db_session, "x1", status="disabled")

    counts = await km.count_by_status(db_session)
    assert counts["active"] == 2
    assert counts["depleted"] == 1
    assert counts["disabled"] == 1


# ----------------------------------------------------------------- helpers


def test_extract_retry_after_uses_60_when_missing() -> None:
    assert KeyManager.extract_retry_after(None) == 60
    assert KeyManager.extract_retry_after([]) == 60


def test_extract_retry_after_parses_seconds() -> None:
    headers = [("retry-after", "120")]
    assert KeyManager.extract_retry_after(headers) == 120


def test_extract_retry_after_clamps_to_min_one() -> None:
    headers = [("Retry-After", "0")]
    assert KeyManager.extract_retry_after(headers) == 1
