from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import BLOB, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ApiKeyStatus(StrEnum):

    ACTIVE = "active"
    DEPLETED = "depleted"
    DISABLED = "disabled"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]

def _utcnow() -> datetime:
    return datetime.now(UTC)

class ApiKey(Base):

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(8), nullable=False)
    key_encrypted: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ApiKeyStatus.ACTIVE.value
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cooldown_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_requests: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_failures: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        nullable=False,
    )
    account_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    account_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    account_plan: Mapped[str | None] = mapped_column(String(32), nullable=True)
    account_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_usage_fetch_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_usage_session: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_usage_weekly: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_usage_models_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_usage_session_fraction: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    last_usage_weekly_fraction: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    session_prompt_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    session_completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    session_window_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    weekly_prompt_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    weekly_completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    weekly_window_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_token_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_api_keys_status", "status"),
    )

    @property
    def key_preview(self) -> str:
        if not self.key_prefix:
            return ""
        return f"{self.key_prefix}…"
