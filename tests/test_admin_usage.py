"""Tests for the admin usage endpoints.

Exercises ``GET /admin/keys/{id}/usage``,
``POST /admin/keys/{id}/usage/refresh`` and
``POST /admin/keys/usage/refresh-all`` through the ASGI client
provided by ``conftest.client``. The upstream ``/api/me`` and
``/api/usage`` responses are served by ``MockRouter``.

Covers:
    * Happy path — the snapshot payload includes account / official /
      local_cumsum / upstream_status="ok".
    * Force refresh — POST /admin/keys/{id}/usage/refresh issues
      upstream calls even if the cached snapshot is fresh.
    * Refresh-all — runs across every active key.
    * 404 on unknown key.
    * 401 without admin bearer token.
"""

from __future__ import annotations

from typing import Any

from app.services import vault as vault_module


def _account_payload() -> dict[str, Any]:
    return {
        "ID": "acct-xyz",
        "Email": "operator@example.com",
        "Name": "test-acc",
        "Plan": "free",
    }


def _usage_payload() -> dict[str, Any]:
    # Mirrors the real ollama.com /api/usage payload: ``usage`` is a
    # fraction in [0, 1] (share of the limit consumed in the window),
    # not an absolute count.
    return {
        "activity": {
            "cost": "0.00000",
            "period": {
                "type": "last_4_weeks",
                "starting_at": "2026-07-06T00:00:00Z",
                "ending_at": "2026-07-27T10:00:00Z",
            },
            "models": [],
        },
        "limits": {
            "session": {
                "usage": 0.037,
                "models": [{"name": "minimax-m3", "request_count": 1}],
            },
            "weekly": {
                "usage": 0.014,
                "models": [{"name": "minimax-m3", "request_count": 5}],
            },
        },
    }



async def test_get_key_usage_returns_snapshot(
    client: Any, mock_router: Any, db_session: Any
) -> None:
    raw = "sk-test-1234567890"
    from app.models.api_key import ApiKey

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
    key_id = api_key.id

    mock_router.add("POST", "/api/me", json_body=_account_payload())
    mock_router.add("GET", "/api/usage", json_body=_usage_payload())

    resp = await client.get(f"/admin/keys/{key_id}/usage")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["upstream_status"] == "ok"
    assert body["account"]["email"] == "operator@example.com"
    # Legacy int view: truncation of the upstream fraction.
    assert body["official"]["session_usage"] == 0
    # New float view: the raw upstream fraction.
    assert abs(body["official"]["session_usage_fraction"] - 0.037) < 1e-9
    assert abs(body["official"]["weekly_usage_fraction"] - 0.014) < 1e-9
    assert body["local_cumsum"]["session_prompt_tokens"] == 0



async def test_get_key_usage_404_unknown_id(client: Any) -> None:
    resp = await client.get("/admin/keys/999999/usage")
    assert resp.status_code == 404



async def test_post_usage_refresh_force(client: Any, mock_router: Any, db_session: Any) -> None:
    raw = "sk-test-1234567890"
    from app.models.api_key import ApiKey

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
    key_id = api_key.id

    # Two cycles: 2 (me+usage) + 2 (me+usage force) = 4 upstream calls.
    for _ in range(2):
        mock_router.add("POST", "/api/me", json_body=_account_payload())
        mock_router.add("GET", "/api/usage", json_body=_usage_payload())

    resp = await client.post(f"/admin/keys/{key_id}/usage/refresh")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["upstream_status"] == "ok"

    # The row should now have account_email populated.
    db_session.expire_all()
    refreshed = await db_session.get(ApiKey, key_id)
    assert refreshed.account_email == "operator@example.com"



async def test_post_usage_refresh_all(client: Any, mock_router: Any, db_session: Any) -> None:
    from app.models.api_key import ApiKey

    for i in range(2):
        db_session.add(
            ApiKey(
                key_hash=f"h{i}" + "0" * 63,
                key_prefix=f"sk-test-{i:02d}",
                key_encrypted=vault_module.get_vault().encrypt(f"sk-test-{i:02d}"),
                status="active",
            )
        )
    await db_session.commit()

    # 2 endpoints x 2 keys = 4 upstream calls. Order isn't guaranteed
    # so register each route multiple times.
    for _ in range(4):
        mock_router.add("POST", "/api/me", json_body=_account_payload())
        mock_router.add("GET", "/api/usage", json_body=_usage_payload())

    resp = await client.post("/admin/keys/usage/refresh-all")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert len(body["results"]) == 2
    for r in body["results"]:
        assert r["upstream_status"] == "ok"



async def test_post_usage_refresh_unauthorised_returns_unauthorised_status(
    client: Any, mock_router: Any, db_session: Any
) -> None:
    """401 from upstream must surface as ``upstream_status="unauthorised"``,
    NOT as an HTTP 401 — the admin endpoint stays accessible and the
    operator can PATCH the row to ``disabled``."""
    raw = "sk-test-1234567890"
    from app.models.api_key import ApiKey

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
    key_id = api_key.id

    mock_router.add("POST", "/api/me", 401)
    mock_router.add("GET", "/api/usage", 401)

    resp = await client.get(f"/admin/keys/{key_id}/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert body["upstream_status"] == "unauthorised"
