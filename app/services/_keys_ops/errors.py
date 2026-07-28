from __future__ import annotations

from typing import Any, Literal

_DashboardSource = Literal["http", "in-process"]

class KeysOpsError(Exception):
    ...

class VaultDecryptError(KeysOpsError):

    def __init__(self, key_id: int, inner: BaseException) -> None:
        self.key_id = key_id
        self.inner = inner
        super().__init__(f"cannot decrypt stored key id={key_id}: {inner}")

class ProbeModelMissingError(KeysOpsError):
    ...

class ApiKeyNotFoundError(KeysOpsError, LookupError):

    def __init__(self, key_id: int) -> None:
        self.key_id = key_id
        super().__init__(f"API key id={key_id} not found")

class DashboardClientError(RuntimeError):

    def __init__(
        self,
        status_code: int,
        body: Any,
        endpoint: str,
        *,
        source: _DashboardSource = "http",
    ) -> None:
        self.status_code = status_code
        self.body = body
        self.endpoint = endpoint
        self.source = source
        super().__init__(self._format_message(status_code, body, endpoint, source))

    @staticmethod
    def _format_message(
        status_code: int, body: Any, endpoint: str, source: _DashboardSource
    ) -> str:
        prefix = "in-process error" if source == "in-process" else "proxy returned"
        if isinstance(body, dict) and "detail" in body:
            return f"{prefix} {status_code}: {body['detail']}"
        if isinstance(body, str) and body:
            return f"{prefix} {status_code}: {body[:200]}"
        return f"{prefix} {status_code} for {endpoint}"

    @property
    def short(self) -> str:
        prefix = "in-process error" if self.source == "in-process" else "proxy returned"
        if isinstance(self.body, dict) and "detail" in self.body:
            return f"{prefix} {self.status_code}: {self.body['detail']}"
        return f"{prefix} {self.status_code}"

__all__ = [
    "ApiKeyNotFoundError",
    "DashboardClientError",
    "KeysOpsError",
    "ProbeModelMissingError",
    "VaultDecryptError",
]
