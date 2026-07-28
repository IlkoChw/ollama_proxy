from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.api_key import ApiKey
from app.schemas.api_key import ApiKeyTestResult
from app.services.key_manager import KeyManager

from .errors import ProbeModelMissingError

__all__ = ["ProbeClassifier", "require_probe_model"]

def require_probe_model() -> str:
    settings = get_settings()
    model = (settings.probe_model or "").strip()
    if not model:
        raise ProbeModelMissingError()
    return model

class ProbeClassifier:

    @staticmethod
    async def classify(
        session: AsyncSession,
        keys: KeyManager,
        key: ApiKey,
        result: ApiKeyTestResult | Any,
    ) -> None:
        code = getattr(result, "status_code", None)
        if code is None:
            return
        if 200 <= code < 300:
            await keys.record_success(session, key.id, code)
        elif code in (401, 403, 404):
            await keys.record_unauthorized(session, key.id, code)
        elif code == 429:
            retry_after = KeyManager.extract_retry_after(
                getattr(result, "ratelimit_headers", None)
            )
            await keys.record_rate_limited(session, key.id, code, retry_after)
        elif code >= 500:
            await keys.record_server_error(session, key.id, code)
