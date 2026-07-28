from __future__ import annotations

from typing import Any

import httpx

from app.services._keys_ops.errors import DashboardClientError  # noqa: F401


class DashboardClient:

    def __init__(
        self,
        base_url: str,
        admin_token: str,
        *,
        timeout: float = 15.0,
        _client: httpx.AsyncClient | None = None,
    ) -> None:
        if not admin_token:
            # Fail-fast: an empty token would silently bypass admin auth
            # in the proxy. Make the mistake loud.
            raise ValueError("admin_token must not be empty")
        self._base_url = base_url.rstrip("/")
        self._admin_token = admin_token
        self._timeout = timeout
        self._client = _client

    async def __aenter__(self) -> DashboardClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._admin_token}",
                    "Accept": "application/json",
                },
            )
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----------------------------------------------------------- helpers

    def _require(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "DashboardClient must be used as an async context manager"
            )
        return self._client

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        client = self._require()
        endpoint = f"{method} {path}"
        resp = await client.request(method, path, json=json, params=params)
        if resp.status_code == 204:
            return None
        body: Any
        try:
            body = resp.json()
        except ValueError:
            body = resp.text
        if not (200 <= resp.status_code < 300):
            raise DashboardClientError(resp.status_code, body, endpoint)
        if not isinstance(body, dict | list):
            raise DashboardClientError(resp.status_code, body, endpoint)
        return body

    # ----------------------------------------------------------- API

    async def list_keys(self) -> list[dict[str, Any]]:
        body = await self._request_json("GET", "/admin/keys")
        if not isinstance(body, list):
            # The proxy contract is a JSON array; if the dashboard
            # ever sees a different shape, surface it as an error.
            raise DashboardClientError(200, body, "GET /admin/keys")
        return [x for x in body if isinstance(x, dict)]

    async def get_key(self, key_id: int) -> dict[str, Any]:
        return await self._request_json("GET", f"/admin/keys/{key_id}")  # type: ignore[return-value]

    async def create_key(
        self,
        label: str | None,
        key: str,
    ) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "POST",
            "/admin/keys",
            json={"label": label, "key": key},
        )

    async def update_key(
        self,
        key_id: int,
        *,
        label: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if label is not None:
            payload["label"] = label
        if status is not None:
            payload["status"] = status
        return await self._request_json(  # type: ignore[return-value]
            "PATCH",
            f"/admin/keys/{key_id}",
            json=payload,
        )

    async def delete_key(self, key_id: int) -> None:
        await self._request_json("DELETE", f"/admin/keys/{key_id}")

    async def test_key(self, key_id: int) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "POST", f"/admin/keys/{key_id}/test"
        )

    async def test_all_keys(self) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "POST", "/admin/keys/test-all"
        )

    async def reset_states(self) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "POST", "/admin/keys/reset-states"
        )

    async def health(self) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "GET", "/admin/health"
        )

    async def get_key_usage(self, key_id: int) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "GET", f"/admin/keys/{key_id}/usage"
        )

    async def refresh_key_usage(self, key_id: int) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "POST", f"/admin/keys/{key_id}/usage/refresh"
        )

    async def refresh_all_usage(self) -> dict[str, Any]:
        return await self._request_json(  # type: ignore[return-value]
            "POST", "/admin/keys/usage/refresh-all"
        )
