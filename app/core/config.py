from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    ollama_base_url: str = Field(default="https://ollama.com")
    db_path: str = Field(default="./data/ollama_proxy.db")
    probe_model: str | None = Field(default=None)
    timeout: float = Field(default=30.0)
    max_rotation_iterations_safety_margin: int = Field(default=1)
    model_cache_ttl_seconds: int = Field(default=300)
    usage_refresh_interval_seconds: int = Field(default=300)
    usage_request_timeout_seconds: float = Field(default=15.0)
    # Admin auth: shared bearer token. ``None`` or empty → admin endpoints
    # are refused with 503 (fail-closed). Set via ``ADMIN_TOKEN`` env var.
    admin_token: str | None = Field(default=None)
    cors_allow_origins: str = Field(default="")
    enable_docs: bool = Field(default=False)

    enable_dashboard: bool = Field(default=False)
    dashboard_password: str | None = Field(default=None)
    # HMAC secret used to sign dashboard session and CSRF cookies.
    # Must be set when the dashboard is enabled.
    dashboard_session_secret: str | None = Field(default=None)
    dashboard_session_ttl_seconds: int = Field(default=3600)
    dashboard_cookie_secure: bool = Field(default=True)
    # Templates directory (overridable so dev iterations on the host
    # can bind-mount ``./templates`` and skip an image rebuild).
    dashboard_templates_dir: str = Field(default="./templates")

    @field_validator("probe_model", mode="before")
    @classmethod
    def _empty_probe_model_to_none(cls, value: object) -> object:
        if value in ("", None):
            return None
        return value

    @field_validator("admin_token", mode="before")
    @classmethod
    def _empty_admin_token_to_none(cls, value: object) -> object:
        if value in ("", None):
            return None
        return value

    @field_validator("enable_docs", mode="before")
    @classmethod
    def _empty_enable_docs_to_false(cls, value: object) -> object:
        if value in ("", None, "0", "false", "False", "no", "off"):
            return False
        if isinstance(value, str):
            return value.lower() not in ("0", "false", "no", "off")
        return value

    @field_validator("enable_dashboard", mode="before")
    @classmethod
    def _empty_enable_dashboard_to_false(cls, value: object) -> object:
        if value in ("", None, "0", "false", "False", "no", "off"):
            return False
        if isinstance(value, str):
            return value.lower() not in ("0", "false", "no", "off")
        return value

    @field_validator("dashboard_password", mode="before")
    @classmethod
    def _empty_dashboard_password_to_none(cls, value: object) -> object:
        if value in ("", None):
            return None
        return value

    @field_validator("dashboard_session_secret", mode="before")
    @classmethod
    def _empty_dashboard_session_secret_to_none(cls, value: object) -> object:
        if value in ("", None):
            return None
        return value

    @field_validator("dashboard_cookie_secure", mode="before")
    @classmethod
    def _empty_dashboard_cookie_secure_to_false(cls, value: object) -> object:
        if value in ("", None, "0", "false", "False", "no", "off"):
            return False
        if isinstance(value, str):
            return value.lower() not in ("0", "false", "no", "off")
        return value

    @field_validator("dashboard_session_ttl_seconds")
    @classmethod
    def _validate_dashboard_ttl(cls, value: int) -> int:
        if value < 60:
            raise ValueError(
                "DASHBOARD_SESSION_TTL_SECONDS must be at least 60 seconds"
            )
        if value > 24 * 3600:
            raise ValueError(
                "DASHBOARD_SESSION_TTL_SECONDS must not exceed 24h"
            )
        return value

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
