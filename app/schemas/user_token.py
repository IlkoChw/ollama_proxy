from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.user_token import UserTokenStatus as _ModelUserTokenStatus

# Re-export so the API layer never imports the ORM enum directly.
UserTokenStatus = _ModelUserTokenStatus

class UserTokenCreate(BaseModel):

    label: str = Field(..., min_length=1, max_length=255)
    expires_at: datetime | None = None

    @field_validator("label")
    @classmethod
    def _strip_label(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("label must not be blank")
        return v2

    @field_validator("expires_at")
    @classmethod
    def _normalise_expires_at(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        else:
            v = v.astimezone(UTC)
        # No deeper validation here; service layer raises if value is in
        # the past at create time.
        return v

class UserTokenUpdate(BaseModel):

    label: str | None = Field(default=None, max_length=255)
    expires_at: datetime | None = None
    status: UserTokenStatus | None = None

    @field_validator("label")
    @classmethod
    def _strip_label(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v2 = v.strip()
        if not v2:
            raise ValueError("label must not be blank when provided")
        return v2

    @field_validator("expires_at")
    @classmethod
    def _normalise_expires_at(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        else:
            v = v.astimezone(UTC)
        return v

class UserTokenOut(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    key_preview: str
    status: UserTokenStatus
    expires_at: datetime | None
    last_used_at: datetime | None
    total_requests: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_orm_token(cls, token: Any) -> UserTokenOut:
        return cls(
            id=token.id,
            label=token.label,
            key_preview=token.key_preview,
            status=UserTokenStatus(token.status),
            expires_at=token.expires_at,
            last_used_at=token.last_used_at,
            total_requests=token.total_requests,
            created_at=token.created_at,
            updated_at=token.updated_at,
        )

class UserTokenCreated(UserTokenOut):

    raw_key: str
