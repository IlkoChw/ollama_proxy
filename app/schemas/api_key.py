from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.api_key import ApiKeyStatus as _ModelApiKeyStatus

# Re-export so API layer never imports the ORM enum directly.
ApiKeyStatus = _ModelApiKeyStatus

class ApiKeyCreate(BaseModel):

    label: str | None = Field(default=None, max_length=255)
    key: str = Field(..., min_length=1, max_length=512)

class ApiKeyUpdate(BaseModel):

    label: str | None = Field(default=None, max_length=255)
    status: ApiKeyStatus | None = None

class ApiKeyOut(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str | None
    key_preview: str
    status: ApiKeyStatus
    last_used_at: datetime | None
    last_status_code: int | None
    cooldown_until: datetime | None
    total_requests: int
    total_failures: int
    created_at: datetime
    updated_at: datetime
    # ----- usage fraction (from ollama.com /api/usage; nullable) -----
    session_usage_fraction: float | None = None
    weekly_usage_fraction: float | None = None
    last_usage_fetch_at: datetime | None = None

    @classmethod
    def from_orm_key(cls, key: Any) -> ApiKeyOut:
        return cls(
            id=key.id,
            label=key.label,
            key_preview=key.key_preview,
            status=ApiKeyStatus(key.status),
            last_used_at=key.last_used_at,
            last_status_code=key.last_status_code,
            cooldown_until=key.cooldown_until,
            total_requests=key.total_requests,
            total_failures=key.total_failures,
            created_at=key.created_at,
            updated_at=key.updated_at,
            session_usage_fraction=key.last_usage_session_fraction,
            weekly_usage_fraction=key.last_usage_weekly_fraction,
            last_usage_fetch_at=key.last_usage_fetch_at,
        )

class ApiKeyCreated(ApiKeyOut):

    raw_key: str

class ApiKeyTestResult(BaseModel):

    ok: bool
    status_code: int | None = None
    latency_ms: float | None = None
    ratelimit_headers: dict[str, str] = Field(default_factory=dict)
    error: str | None = None

class ApiKeyTestResultWithKey(ApiKeyTestResult):

    id: int
    label: str | None
    key_preview: str

class ApiKeyHealth(BaseModel):

    status: Literal["ok", "degraded", "down"]
    active_keys: int
    depleted_keys: int
    disabled_keys: int
    timestamp: datetime

# ----------------------------------------------------------- usage schemas

class AccountInfo(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    account_id: str | None = None
    email: str | None = None
    name: str | None = None
    plan: str | None = None
    fetched_at: datetime

class ModelUsageRow(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    name: str
    session_request_count: int = 0
    weekly_request_count: int = 0

class OfficialUsage(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    session_usage: int = 0
    weekly_usage: int = 0
    # Raw fraction reported by ollama.com (``limits.{session,weekly}.usage``).
    # May be ``None`` if upstream did not emit the field.
    session_usage_fraction: float | None = None
    weekly_usage_fraction: float | None = None
    models: list[ModelUsageRow] = Field(default_factory=list)
    period_type: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    fetched_at: datetime

class LocalCumsum(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    session_prompt_tokens: int = 0
    session_completion_tokens: int = 0
    session_window_started_at: datetime | None = None
    weekly_prompt_tokens: int = 0
    weekly_completion_tokens: int = 0
    weekly_window_started_at: datetime | None = None
    last_token_at: datetime | None = None
    window_kind: Literal["rolling_24h_7d"] = "rolling_24h_7d"

class KeyUsageSnapshot(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str | None
    key_preview: str
    account: AccountInfo | None = None
    official: OfficialUsage | None = None
    local_cumsum: LocalCumsum = Field(default_factory=LocalCumsum)
    upstream_status: Literal["ok", "unreachable", "unauthorised", "pending"] = "pending"
    upstream_error: str | None = None
