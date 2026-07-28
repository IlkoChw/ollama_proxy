"""Tests for the /admin/keys CRUD, probe, and test-all endpoints."""

from __future__ import annotations

import httpx

# ----------------------------------------------------------------- CRUD


async def test_create_key_returns_raw_key_once(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/admin/keys",
        json={"label": "acc1", "key": "sk-test-abcdefghij"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["raw_key"] == "sk-test-abcdefghij"
    assert body["label"] == "acc1"
    assert body["key_preview"] == "sk-test-…"
    assert body["status"] == "active"
    # No hash leaked.
    assert "key_hash" not in body


async def test_list_keys_masks_raw(client: httpx.AsyncClient) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-11111111aaaa"})
    await client.post("/admin/keys", json={"label": "acc2", "key": "sk-22222222bbbb"})

    resp = await client.get("/admin/keys")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    for item in items:
        assert "raw_key" not in item
        assert "key_hash" not in item
        assert item["key_preview"].endswith("…")
        assert len(item["key_preview"]) == 9  # 8 chars + "…"


async def test_duplicate_key_returns_409(client: httpx.AsyncClient) -> None:
    await client.post("/admin/keys", json={"label": "acc1", "key": "sk-dup-key1234"})
    resp = await client.post(
        "/admin/keys", json={"label": "acc1-dup", "key": "sk-dup-key1234"}
    )
    assert resp.status_code == 409


async def test_get_key_by_id(client: httpx.AsyncClient) -> None:
    create = await client.post(
        "/admin/keys", json={"label": "acc1", "key": "sk-get-by-id-12"}
    )
    key_id = create.json()["id"]

    resp = await client.get(f"/admin/keys/{key_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == key_id
    assert "raw_key" not in body
    assert body["key_preview"].endswith("…")


async def test_get_missing_key_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/admin/keys/9999")
    assert resp.status_code == 404


async def test_patch_key_updates_label_and_status(
    client: httpx.AsyncClient,
) -> None:
    create = await client.post(
        "/admin/keys", json={"label": "acc1", "key": "sk-patch-target"}
    )
    key_id = create.json()["id"]

    resp = await client.patch(
        f"/admin/keys/{key_id}", json={"label": "renamed", "status": "disabled"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "renamed"
    assert body["status"] == "disabled"


async def test_delete_key_returns_204(client: httpx.AsyncClient) -> None:
    create = await client.post(
        "/admin/keys", json={"label": "acc1", "key": "sk-delete-target"}
    )
    key_id = create.json()["id"]

    resp = await client.delete(f"/admin/keys/{key_id}")
    assert resp.status_code == 204

    # Subsequent GET → 404.
    follow = await client.get(f"/admin/keys/{key_id}")
    assert follow.status_code == 404


# ----------------------------------------------------------------- probes


async def test_test_one_key_200_classifies_success(
    client: httpx.AsyncClient, mock_router
) -> None:
    create = await client.post(
        "/admin/keys", json={"label": "acc1", "key": "sk-probe-ok-1"}
    )
    key_id = create.json()["id"]

    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "pong"}}],
        },
    )

    resp = await client.post(f"/admin/keys/{key_id}/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["status_code"] == 200
    assert body["latency_ms"] is not None and body["latency_ms"] >= 0
    assert body["id"] == key_id
    assert body["key_preview"].endswith("…")


async def test_test_one_key_401_disables_key(
    client: httpx.AsyncClient, mock_router
) -> None:
    create = await client.post(
        "/admin/keys", json={"label": "acc1", "key": "sk-probe-bad-2"}
    )
    key_id = create.json()["id"]

    mock_router.add("POST", "/v1/chat/completions", status=401, text="unauthorized")

    resp = await client.post(f"/admin/keys/{key_id}/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["status_code"] == 401

    # And the key is now disabled.
    follow = await client.get(f"/admin/keys/{key_id}")
    assert follow.json()["status"] == "disabled"


async def test_test_all_keys(client: httpx.AsyncClient, mock_router) -> None:
    await client.post("/admin/keys", json={"label": "a1", "key": "sk-a1111111aaa"})
    await client.post("/admin/keys", json={"label": "a2", "key": "sk-a2222222bbb"})

    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={"id": "1", "object": "chat.completion", "choices": []},
    )
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=429,
        text="rate limited",
        headers={"Retry-After": "120"},
    )

    resp = await client.post("/admin/keys/test-all")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    statuses = sorted(item["status_code"] for item in body["results"])
    assert statuses == [200, 429]


# ----------------------------------------------------------------- health


async def test_health_no_keys(client: httpx.AsyncClient) -> None:
    resp = await client.get("/admin/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "down"
    assert body["active_keys"] == 0


async def test_health_with_active_keys(client: httpx.AsyncClient) -> None:
    await client.post("/admin/keys", json={"label": "a1", "key": "sk-a1-aaaaaaa"})
    resp = await client.get("/admin/health")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["active_keys"] == 1


# -------------------------------------------------------------- reset-states


async def test_reset_states_reactivates_all_keys(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """Disabled/depleted keys must come back to 'active' after reset."""
    # Create three keys.
    await client.post("/admin/keys", json={"label": "a", "key": "sk-reset-aaa111"})
    await client.post("/admin/keys", json={"label": "b", "key": "sk-reset-bbb222"})
    await client.post("/admin/keys", json={"label": "c", "key": "sk-reset-ccc333"})

    # Push them into 401 → disabled.
    mock_router.add("POST", "/v1/chat/completions", status=401, text="nope")
    mock_router.add("POST", "/v1/chat/completions", status=401, text="nope")
    mock_router.add("POST", "/v1/chat/completions", status=401, text="nope")
    await user_client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )

    listing = await client.get("/admin/keys")
    statuses = {k["label"]: k["status"] for k in listing.json()}
    assert statuses == {"a": "disabled", "b": "disabled", "c": "disabled"}

    # Reset.
    resp = await client.post("/admin/keys/reset-states")
    assert resp.status_code == 200
    body = resp.json()
    # Post-reset snapshot.
    assert body["active_keys"] == 3
    assert body["disabled_keys"] == 0
    assert body["depleted_keys"] == 0
    assert body["status"] == "ok"

    # Verify individual key states.
    listing = await client.get("/admin/keys")
    for k in listing.json():
        assert k["status"] == "active"
        assert k["cooldown_until"] is None
        assert k["last_status_code"] is None
        assert k["total_failures"] == 0


async def test_reset_states_clears_depleted_cooldown(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """A 429-depleted key with a future cooldown must come back too."""
    await client.post("/admin/keys", json={"label": "k1", "key": "sk-rl-aa-1111"})

    # Force 429.
    mock_router.add("POST", "/v1/chat/completions", status=429, text="rl")
    await user_client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )

    pre = (await client.get("/admin/keys")).json()[0]
    assert pre["status"] == "depleted"
    assert pre["cooldown_until"] is not None
    assert pre["total_failures"] == 1

    resp = await client.post("/admin/keys/reset-states")
    assert resp.status_code == 200
    assert resp.json()["active_keys"] == 1

    post = (await client.get("/admin/keys")).json()[0]
    assert post["status"] == "active"
    assert post["cooldown_until"] is None
    assert post["total_failures"] == 0


async def test_reset_states_preserves_total_requests(
    client: httpx.AsyncClient, user_client: httpx.AsyncClient, mock_router
) -> None:
    """The reset is a recovery action, not a wipe. total_requests
    and created_at must survive."""
    await client.post("/admin/keys", json={"label": "k1", "key": "sk-rp-aa-1111"})

    # One successful request to bump total_requests.
    mock_router.add(
        "POST",
        "/v1/chat/completions",
        status=200,
        json_body={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
    )
    await user_client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
    )

    pre = (await client.get("/admin/keys")).json()[0]
    assert pre["total_requests"] == 1
    pre_created = pre["created_at"]

    # Now reset.
    await client.post("/admin/keys/reset-states")
    post = (await client.get("/admin/keys")).json()[0]
    assert post["total_requests"] == 1  # preserved
    assert post["created_at"] == pre_created  # preserved


async def test_reset_states_with_no_keys(
    client: httpx.AsyncClient,
) -> None:
    """Calling reset on an empty pool is a no-op that returns a
    'down' health snapshot."""
    resp = await client.post("/admin/keys/reset-states")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_keys"] == 0
    assert body["status"] == "down"
