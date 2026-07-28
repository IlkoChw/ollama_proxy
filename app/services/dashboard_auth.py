from __future__ import annotations

import json
import secrets
from typing import Any, Literal

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.core.logging import logger

# Cookie names. Centralised so test code and route handlers refer to
# the same strings.
SESSION_COOKIE = "opdash_sid"
CSRF_COOKIE = "opdash_csrf"
FLASH_COOKIE = "opdash_flash"

# Flash levels the templates know how to render.
FlashLevel = Literal["ok", "error"]

# Default max-age for the CSRF cookie (matches the session TTL by
# default; rotated on every login so it can be shorter if desired).
_CSRF_COOKIE_MAX_AGE = 24 * 3600

# Single-use flash TTL: enough to survive a 303 redirect, no more.
_FLASH_MAX_AGE = 60

class DashboardAuth:

    def __init__(
        self,
        *,
        secret: str,
        session_ttl_seconds: int,
        cookie_secure: bool,
    ) -> None:
        if not secret:
            raise ValueError("DashboardAuth.secret must not be empty")
        if session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be positive")
        self._secret = secret
        self._session_ttl_seconds = session_ttl_seconds
        self._cookie_secure = cookie_secure
        # ``salt`` here scopes the signatures so the same secret
        # reused for CSRF / flash can't be cross-traded.
        self._session_serializer = URLSafeTimedSerializer(
            secret, salt="dashboard-session-v1"
        )
        self._csrf_serializer = URLSafeTimedSerializer(
            secret, salt="dashboard-csrf-v1"
        )
        self._flash_serializer = URLSafeTimedSerializer(
            secret, salt="dashboard-flash-v1"
        )

    # ------------------------------------------------------------ session

    def issue_session(self) -> str:
        raw = secrets.token_urlsafe(32)
        return self._session_serializer.dumps(raw)

    def verify_session(self, value: str) -> str | None:
        if not value:
            return None
        try:
            return self._session_serializer.loads(
                value, max_age=self._session_ttl_seconds
            )
        except SignatureExpired:
            logger.info("verify_session: expired session")
            return None
        except BadSignature:
            logger.info("verify_session: bad signature")
            return None

    def reissue_session(self, session_id: str) -> str:
        if not session_id:
            return ""
        return self._session_serializer.dumps(session_id)

    # ------------------------------------------------------------ CSRF

    def issue_csrf(self, session_id: str) -> str:
        if not session_id:
            raise ValueError("session_id must not be empty")
        # 16 bytes of entropy are mixed into the payload so two
        # consecutive issuances for the same session differ.
        token = secrets.token_urlsafe(16)
        return self._csrf_serializer.dumps({"sid": session_id, "t": token})

    def verify_csrf(self, session_id: str, token: str | None) -> bool:
        if not session_id or not token:
            return False
        try:
            payload = self._csrf_serializer.loads(
                token, max_age=_CSRF_COOKIE_MAX_AGE
            )
        except (SignatureExpired, BadSignature):
            return False
        if not isinstance(payload, dict):
            return False
        return payload.get("sid") == session_id

    def csrf_cookie_max_age(self) -> int:
        return _CSRF_COOKIE_MAX_AGE

    # ------------------------------------------------------------ flash

    def flash_put(self, level: FlashLevel, msg: str) -> str:
        if level not in ("ok", "error"):
            raise ValueError(f"invalid flash level: {level!r}")
        payload = {"level": level, "msg": msg}
        return self._flash_serializer.dumps(json.dumps(payload))

    def flash_read(self, value: str | None) -> dict[str, Any] | None:
        if not value:
            return None
        try:
            raw = self._flash_serializer.loads(value, max_age=_FLASH_MAX_AGE)
        except (SignatureExpired, BadSignature):
            return None
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        level = data.get("level")
        msg = data.get("msg")
        if level not in ("ok", "error"):
            return None
        if not isinstance(msg, str) or not msg.strip():
            return None
        return {"level": level, "msg": msg}

    # ------------------------------------------------------------ flags

    @property
    def cookie_secure(self) -> bool:
        return self._cookie_secure

    @property
    def session_ttl_seconds(self) -> int:
        return self._session_ttl_seconds

# -------------------------------------------------------------- singleton

_auth: DashboardAuth | None = None

def get_dashboard_auth() -> DashboardAuth:
    if _auth is None:
        raise RuntimeError(
            "dashboard auth not initialised; call set_dashboard_auth() "
            "from the FastAPI lifespan"
        )
    return _auth

def set_dashboard_auth(auth: DashboardAuth | None) -> None:
    global _auth  # noqa: PLW0603
    _auth = auth
