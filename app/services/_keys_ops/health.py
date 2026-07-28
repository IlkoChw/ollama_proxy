from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKeyStatus
from app.schemas.api_key import ApiKeyHealth
from app.services.key_manager import KeyManager

__all__ = ["build_health_snapshot"]

async def build_health_snapshot(
    session: AsyncSession,
    keys_manager: KeyManager,
    *,
    now: datetime | None = None,
) -> ApiKeyHealth:
    counts = await keys_manager.count_by_status(session)
    active = counts.get(ApiKeyStatus.ACTIVE.value, 0)
    depleted = counts.get(ApiKeyStatus.DEPLETED.value, 0)
    disabled = counts.get(ApiKeyStatus.DISABLED.value, 0)
    if active > 0:
        overall = "ok"
    elif (depleted + disabled) > 0:
        overall = "degraded"
    else:
        overall = "down"
    return ApiKeyHealth(
        status=overall,  # type: ignore[arg-type]
        active_keys=active,
        depleted_keys=depleted,
        disabled_keys=disabled,
        timestamp=now or datetime.now(UTC),
    )
