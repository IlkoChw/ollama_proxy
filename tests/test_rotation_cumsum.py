"""Tests for the per-key token cumsum maintained by ``Rotation``.

Covers:
    * Non-stream chat response with a ``usage`` block → cumsum
      incremented and windows tracked.
    * Non-stream chat response WITHOUT a ``usage`` block → cumsum
      unchanged (e.g. older ollama versions).
    * Stream chat response → cumsum NOT updated (only non-stream).
    * 24h window reset: when ``now - session_window_started_at >= 24h``,
      the session counters reset before the increment.
    * 7d window reset: same logic for weekly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.api_key import ApiKey
from app.services import key_manager as key_manager_module
from app.services import vault as vault_module
from app.services.rotation import _parse_usage_from_response

# ----------------------------------------------------------- pure parser


def test_parse_usage_openai_shape() -> None:
    response = httpx.Response(
        200,
        content=json.dumps(
            {"usage": {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26}}
        ).encode(),
        headers={"content-type": "application/json"},
    )
    pt, ct = _parse_usage_from_response(response)
    assert pt == 17
    assert ct == 9


def test_parse_usage_ollama_native_shape() -> None:
    response = httpx.Response(
        200,
        content=json.dumps(
            {"prompt_eval_count": 12, "eval_count": 7, "done": True}
        ).encode(),
        headers={"content-type": "application/json"},
    )
    pt, ct = _parse_usage_from_response(response)
    assert pt == 12
    assert ct == 7


def test_parse_usage_missing_returns_none() -> None:
    response = httpx.Response(
        200,
        content=json.dumps({"id": "chatcmpl-x", "choices": []}).encode(),
        headers={"content-type": "application/json"},
    )
    pt, ct = _parse_usage_from_response(response)
    assert pt is None
    assert ct is None


def test_parse_usage_garbage_returns_none() -> None:
    response = httpx.Response(
        200,
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    pt, ct = _parse_usage_from_response(response)
    assert pt is None
    assert ct is None


# ----------------------------------------------------------- KeyManager integration


@pytest.mark.asyncio
async def test_record_success_with_usage_increments_cumsum(
    db_session: AsyncSession,
) -> None:
    raw = "sk-test-1234567890"
    api_key = ApiKey(
        key_hash="h" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault_module.get_vault().encrypt(raw),
        label="acc-1",
        status="active",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    km = key_manager_module.get_key_manager()
    updated = await km.record_success_with_usage(
        db_session,
        api_key.id,
        200,
        prompt_tokens=10,
        completion_tokens=4,
    )
    # Reload from the DB so we observe the post-commit state. The
    # ORM instance returned by ``record_success_with_usage`` may still
    # hold pre-commit attribute values when the underlying statement
    # uses ``synchronize_session=False`` for atomic-increment safety
    # under burst load.
    await db_session.refresh(updated)
    assert updated.session_prompt_tokens == 10
    assert updated.session_completion_tokens == 4
    assert updated.weekly_prompt_tokens == 10
    assert updated.weekly_completion_tokens == 4
    assert updated.last_token_at is not None
    assert updated.total_requests == 1


@pytest.mark.asyncio
async def test_record_success_with_usage_resets_session_window(
    db_session: AsyncSession,
) -> None:
    """A record older than 24h must reset the session counter."""
    raw = "sk-test-1234567890"
    api_key = ApiKey(
        key_hash="h" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault_module.get_vault().encrypt(raw),
        label="acc-1",
        status="active",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    km = key_manager_module.get_key_manager()
    # First call seeds the windows.
    await km.record_success_with_usage(
        db_session, api_key.id, 200, prompt_tokens=5, completion_tokens=2
    )
    # Backdate session_window_started_at to >24h ago; weekly stays fresh.
    api_key_backdated = await db_session.get(ApiKey, api_key.id)
    assert api_key_backdated is not None
    api_key_backdated.session_window_started_at = datetime.now(UTC) - timedelta(hours=25)
    api_key_backdated.session_prompt_tokens = 999
    api_key_backdated.session_completion_tokens = 999
    await db_session.commit()
    await db_session.refresh(api_key_backdated)

    # New call must reset session counters and bump weekly (still in window).
    updated = await km.record_success_with_usage(
        db_session, api_key.id, 200, prompt_tokens=7, completion_tokens=3
    )
    assert updated.session_prompt_tokens == 7
    assert updated.session_completion_tokens == 3
    # Weekly accumulated 5 + 7 = 12 (window not expired).
    assert updated.weekly_prompt_tokens == 12


@pytest.mark.asyncio
async def test_record_success_with_usage_resets_weekly_window(
    db_session: AsyncSession,
) -> None:
    raw = "sk-test-1234567890"
    api_key = ApiKey(
        key_hash="h" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault_module.get_vault().encrypt(raw),
        label="acc-1",
        status="active",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    km = key_manager_module.get_key_manager()
    await km.record_success_with_usage(
        db_session, api_key.id, 200, prompt_tokens=5, completion_tokens=2
    )
    api_key_backdated = await db_session.get(ApiKey, api_key.id)
    assert api_key_backdated is not None
    api_key_backdated.weekly_window_started_at = datetime.now(UTC) - timedelta(days=8)
    api_key_backdated.weekly_prompt_tokens = 999
    api_key_backdated.weekly_completion_tokens = 999
    await db_session.commit()
    await db_session.refresh(api_key_backdated)

    updated = await km.record_success_with_usage(
        db_session, api_key.id, 200, prompt_tokens=4, completion_tokens=1
    )
    # Both windows were either fresh or reset; only the fresh weekly window holds.
    # Session accumulated the first 5 plus the new 4 → 9.
    assert updated.weekly_prompt_tokens == 4
    assert updated.weekly_completion_tokens == 1
    assert updated.session_prompt_tokens == 9
    assert updated.session_completion_tokens == 3


@pytest.mark.asyncio
async def test_record_success_with_usage_no_tokens_no_op(
    db_session: AsyncSession,
) -> None:
    """``prompt_tokens=None`` or 0 must NOT touch the cumsum at all."""
    raw = "sk-test-1234567890"
    api_key = ApiKey(
        key_hash="h" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault_module.get_vault().encrypt(raw),
        label="acc-1",
        status="active",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    km = key_manager_module.get_key_manager()
    updated = await km.record_success_with_usage(
        db_session, api_key.id, 200, prompt_tokens=None, completion_tokens=None
    )
    await db_session.refresh(updated)
    assert updated.session_prompt_tokens == 0
    assert updated.session_window_started_at is None
    assert updated.last_token_at is None
    assert updated.total_requests == 1


@pytest.mark.asyncio
async def test_rotation_non_stream_increments_cumsum(
    client: Any,
    mock_router: Any,
    db_session: AsyncSession,
) -> None:
    """End-to-end: a successful non-stream chat request updates cumsum."""
    raw = "sk-test-1234567890"
    api_key = ApiKey(
        key_hash="h" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault_module.get_vault().encrypt(raw),
        label="acc-1",
        status="active",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    mock_router.add(
        "POST",
        "/v1/chat/completions",
        json_body={
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17},
        },
    )

    # Mint a user token to authorise the proxy path.
    resp = await client.post("/admin/user-tokens", json={"label": "rotation-test"})
    user_token = resp.json()["raw_key"]

    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "minimax-m3", "messages": [{"role": "user", "content": "ping"}]},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 200, (resp.status_code, resp.text, mock_router.calls)

    # The ORM session holds a cached copy of the row from the test
    # setup; force a refresh so the post-rotation state is visible.
    await db_session.refresh(api_key)
    assert api_key.session_prompt_tokens == 11
    assert api_key.session_completion_tokens == 6


@pytest.mark.asyncio
async def test_rotation_stream_does_not_increment_cumsum(
    client: Any,
    mock_router: Any,
    db_session: AsyncSession,
) -> None:
    """Stream responses don't carry a single ``usage`` block; cumsum stays."""
    raw = "sk-test-1234567890"
    api_key = ApiKey(
        key_hash="h" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault_module.get_vault().encrypt(raw),
        label="acc-1",
        status="active",
    )
    db_session.add(api_key)
    await db_session.commit()
    await db_session.refresh(api_key)

    body = (
        "data: {\"id\":\"chatcmpl-x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"
        "data: [DONE]\n\n"
    )
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        text=body,
        headers={"content-type": "text/event-stream"},
    )

    resp = await client.post("/admin/user-tokens", json={"label": "rotation-stream-test"})
    user_token = resp.json()["raw_key"]

    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "minimax-m3",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 200

    await db_session.refresh(api_key)
    assert api_key.session_prompt_tokens == 0
    assert api_key.last_token_at is None
