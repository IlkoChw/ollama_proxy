"""Unit tests for :class:`DashboardAuth` (cookie primitives).

These tests do not touch FastAPI — they verify the serializer,
session/CSRF/flash lifecycle in isolation so the integration tests
can rely on the contract.
"""

from __future__ import annotations

import time

import pytest

from app.services.dashboard_auth import (
    CSRF_COOKIE,
    FLASH_COOKIE,
    SESSION_COOKIE,
    DashboardAuth,
)


@pytest.fixture
def auth() -> DashboardAuth:
    """A :class:`DashboardAuth` configured for tests.

    Sliding TTL is intentionally short so we can exercise the
    expiry path without sleeping the suite.
    """
    return DashboardAuth(
        secret="a" * 64,
        session_ttl_seconds=3600,
        cookie_secure=False,
    )


def test_init_rejects_empty_secret() -> None:
    with pytest.raises(ValueError):
        DashboardAuth(secret="", session_ttl_seconds=60, cookie_secure=False)


def test_init_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError):
        DashboardAuth(secret="x", session_ttl_seconds=0, cookie_secure=False)


# --------------------------------------------------------------- session


def test_session_issue_and_verify(auth: DashboardAuth) -> None:
    cookie = auth.issue_session()
    assert cookie  # non-empty
    sid = auth.verify_session(cookie)
    assert sid is not None
    assert len(sid) >= 16


def test_session_verify_rejects_empty(auth: DashboardAuth) -> None:
    assert auth.verify_session("") is None


def test_session_verify_rejects_garbage(auth: DashboardAuth) -> None:
    assert auth.verify_session("not-a-signed-value") is None


def test_session_verify_rejects_wrong_secret() -> None:
    a1 = DashboardAuth(secret="a" * 32, session_ttl_seconds=60, cookie_secure=False)
    a2 = DashboardAuth(secret="b" * 32, session_ttl_seconds=60, cookie_secure=False)
    cookie = a1.issue_session()
    # Same salt but a different secret must not verify.
    assert a2.verify_session(cookie) is None


def test_session_verify_expired() -> None:
    # Use a 1-second TTL and sleep past it. itsdangerous' max_age
    # enforcement rounds to seconds, so we need at least 2s.
    a = DashboardAuth(secret="z" * 32, session_ttl_seconds=1, cookie_secure=False)
    cookie = a.issue_session()
    time.sleep(2)
    assert a.verify_session(cookie) is None


# --------------------------------------------------------------- CSRF


def test_csrf_issue_and_verify(auth: DashboardAuth) -> None:
    sid = "session-xyz"
    token = auth.issue_csrf(sid)
    assert auth.verify_csrf(sid, token) is True


def test_csrf_rejects_other_session(auth: DashboardAuth) -> None:
    token = auth.issue_csrf("session-A")
    # A token minted for one session must not verify under another.
    assert auth.verify_csrf("session-B", token) is False


def test_csrf_rejects_empty_inputs(auth: DashboardAuth) -> None:
    assert auth.verify_csrf("", "abc") is False
    assert auth.verify_csrf("session", "") is False
    assert auth.verify_csrf("session", None) is False  # type: ignore[arg-type]


def test_csrf_rejects_garbage(auth: DashboardAuth) -> None:
    assert auth.verify_csrf("session", "not-signed") is False


def test_csrf_issue_requires_session(auth: DashboardAuth) -> None:
    with pytest.raises(ValueError):
        auth.issue_csrf("")


# --------------------------------------------------------------- flash


def test_flash_put_and_read_roundtrip(auth: DashboardAuth) -> None:
    cookie = auth.flash_put("ok", "key created")
    data = auth.flash_read(cookie)
    assert data == {"level": "ok", "msg": "key created"}


def test_flash_rejects_bad_level(auth: DashboardAuth) -> None:
    with pytest.raises(ValueError):
        auth.flash_put("warning", "x")  # type: ignore[arg-type]


def test_flash_read_rejects_garbage(auth: DashboardAuth) -> None:
    assert auth.flash_read("") is None
    assert auth.flash_read("garbage") is None


def test_flash_read_rejects_non_dict_payload(auth: DashboardAuth) -> None:
    # Manually craft a signed value whose JSON body is a string.
    # The easiest way is to bypass flash_put and call the
    # serializer directly with a string payload.
    raw = auth._flash_serializer.dumps('"a string"')  # type: ignore[attr-defined]
    assert auth.flash_read(raw) is None


def test_flash_read_rejects_blank_message(auth: DashboardAuth) -> None:
    cookie = auth.flash_put("ok", "   ")
    assert auth.flash_read(cookie) is None


# --------------------------------------------------------------- cookies


def test_cookie_names_match_router_expectations() -> None:
    """The dashboard router relies on these constants — keep them stable."""
    assert SESSION_COOKIE == "opdash_sid"
    assert CSRF_COOKIE == "opdash_csrf"
    assert FLASH_COOKIE == "opdash_flash"


# --------------------------------------------------------------- flags


def test_cookie_secure_flag(auth: DashboardAuth) -> None:
    assert auth.cookie_secure is False


def test_session_ttl_property(auth: DashboardAuth) -> None:
    assert auth.session_ttl_seconds == 3600
