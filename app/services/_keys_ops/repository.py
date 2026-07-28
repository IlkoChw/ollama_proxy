from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey

from .errors import ApiKeyNotFoundError


async def load_or_404(session: AsyncSession, key_id: int) -> ApiKey:
    key = await session.get(ApiKey, key_id)
    if key is None:
        raise ApiKeyNotFoundError(key_id)
    return key

__all__ = ["load_or_404"]
