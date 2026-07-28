from __future__ import annotations

import ssl
import time
from typing import Any

import httpx

from app.core.config import Settings
from app.core.logging import logger
from app.schemas.api_key import ApiKeyTestResult

# Header names we surface from a test response. Lower-cased because
# httpx.Headers is case-insensitive.
_RATELIMIT_HEADER_NAMES: tuple[str, ...] = (
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-policy",
    "x-ratelimit-account",
    "ratelimit-limit",
    "ratelimit-remaining",
    "ratelimit-reset",
    "retry-after",
)

def _extract_ratelimit_headers(headers: httpx.Headers | None) -> dict[str, str]:
    if headers is None:
        return {}
    out: dict[str, str] = {}
    for name in _RATELIMIT_HEADER_NAMES:
        value = headers.get(name)
        if value is not None:
            out[name] = str(value)
    return out

def _wrap_low_level_oserror(exc: BaseException) -> httpx.RequestError:
    return httpx.RequestError(f"transport error: {exc!r}", request=httpx.Request("POST", "/"))

class OllamaClient:

    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http_client
        self._settings = settings

    # ----------------------------------------------------------------- probe

    async def probe(
        self, *, key: str, key_preview: str, model: str
    ) -> ApiKeyTestResult:
        if not model:
            raise ValueError("probe requires an explicit model name")
        url = f"{self._settings.ollama_base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        start = time.perf_counter()
        try:
            response = await self._http.post(url, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("probe error: key={} err={}", key_preview, exc)
            return ApiKeyTestResult(
                ok=False,
                status_code=None,
                latency_ms=round(latency_ms, 2),
                error=str(exc),
            )
        except (ssl.SSLError, OSError) as exc:
            # httpcore 1.x lets raw ssl.SSLError / ConnectionResetError
            # bubble up. Treat as transient transport error.
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.warning("probe error: key={} err={}", key_preview, exc)
            return ApiKeyTestResult(
                ok=False,
                status_code=None,
                latency_ms=round(latency_ms, 2),
                error=str(exc),
            )
        latency_ms = (time.perf_counter() - start) * 1000.0
        ratelimit = _extract_ratelimit_headers(response.headers)
        ok = 200 <= response.status_code < 300
        logger.info(
            "probe done: key={} status={} latency_ms={:.1f} ok={}",
            key_preview,
            response.status_code,
            latency_ms,
            ok,
        )
        return ApiKeyTestResult(
            ok=ok,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            ratelimit_headers=ratelimit,
        )

    # ----------------------------------------------------- upstream /v1/models

    async def list_models_v1(self, *, key: str, key_preview: str) -> list[str]:
        url = f"{self._settings.ollama_base_url}/v1/models"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }
        logger.debug("list_models_v1: key={}", key_preview)
        try:
            response = await self._http.get(url, headers=headers)
        except (ssl.SSLError, OSError) as exc:
            raise _wrap_low_level_oserror(exc) from exc
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return []
        data = payload.get("data") or []
        out: list[str] = []
        for entry in data:
            if isinstance(entry, dict):
                model_id = entry.get("id")
                if isinstance(model_id, str) and model_id:
                    out.append(model_id)
            elif isinstance(entry, str) and entry:
                out.append(entry)
        return out

    # --------------------------------------------------------- chat proxying

    async def chat_completion(
        self,
        *,
        key: str,
        key_preview: str,
        body: dict[str, Any],
    ) -> httpx.Response:
        url = f"{self._settings.ollama_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not body.get("stream") else "text/event-stream",
        }
        logger.debug(
            "chat_completion: key={} stream={}", key_preview, bool(body.get("stream"))
        )
        try:
            return await self._http.post(
                url, json=body, headers=headers, timeout=None
            )
        except (ssl.SSLError, OSError) as exc:
            raise _wrap_low_level_oserror(exc) from exc

    async def list_tags(
        self,
        *,
        key: str,
        key_preview: str,
    ) -> httpx.Response:
        url = f"{self._settings.ollama_base_url}/api/tags"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }
        logger.debug("list_tags: key={}", key_preview)
        try:
            return await self._http.get(url, headers=headers)
        except (ssl.SSLError, OSError) as exc:
            raise _wrap_low_level_oserror(exc) from exc

    # ----------------------------------------------------------- account

    async def get_account(
        self,
        *,
        key: str,
        key_preview: str,
    ) -> dict[str, Any]:
        url = f"{self._settings.ollama_base_url}/api/me"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        logger.debug("get_account: key={}", key_preview)
        try:
            response = await self._http.post(url, headers=headers, content=b"{}")
        except (ssl.SSLError, OSError) as exc:
            raise _wrap_low_level_oserror(exc) from exc
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        return payload

    async def get_usage(
        self,
        *,
        key: str,
        key_preview: str,
    ) -> dict[str, Any]:
        url = f"{self._settings.ollama_base_url}/api/usage"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }
        logger.debug("get_usage: key={}", key_preview)
        try:
            response = await self._http.get(url, headers=headers)
        except (ssl.SSLError, OSError) as exc:
            raise _wrap_low_level_oserror(exc) from exc
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        return payload
