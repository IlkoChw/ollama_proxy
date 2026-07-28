from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import logger
from app.services._keys_ops import VaultDecryptError, safe_decrypt
from app.services.key_manager import KeyManager
from app.services.ollama_client import OllamaClient
from app.services.vault import Vault

_DEAD_KEY_CODES: frozenset[int] = frozenset({401, 403})

_SATURATION_WAIT_TOTAL: float = 10.0
_SATURATION_BACKOFF: float = 0.01

_BLOCKED_UPSTREAM_HEADERS: frozenset[str] = frozenset(
    {
        "server",
        "x-powered-by",
        "x-aspnet-version",
        "x-aspnetmvc-version",
    }
)

_TRANSPORT_HEADERS: frozenset[str] = frozenset(
    {
        "content-length",
        "content-encoding",
        "transfer-encoding",
    }
)

@dataclass(slots=True)
class _DispatchOutcome:

    response: httpx.Response | None = None
    terminal: Response | None = None
    rotate: bool = False
    all_404_hint: bool = False

def _safe_headers(upstream_headers: httpx.Headers | Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in upstream_headers.items():
        lower = name.lower()
        if lower in _BLOCKED_UPSTREAM_HEADERS or lower in _TRANSPORT_HEADERS:
            continue
        out[name] = value
    return out

class Rotation:

    def __init__(
        self,
        key_manager: KeyManager,
        vault: Vault,
        ollama_client: OllamaClient,
        settings: Settings,
    ) -> None:
        self._keys = key_manager
        self._vault = vault
        self._client = ollama_client
        self._settings = settings

    async def execute_with_rotation(
        self,
        session: AsyncSession,
        body: dict[str, Any],
    ) -> Response:
        initial_count = await self._count_active(session)
        max_iter = max(1, initial_count) + self._settings.max_rotation_iterations_safety_margin

        stream = bool(body.get("stream"))
        requested_model = body.get("model")

        all_404 = True
        all_disabled = True
        any_attempt = False
        started = time.monotonic()

        for _attempt in range(max_iter):
            key = await self._keys.pick_next_key(session)
            if key is None:
                if initial_count == 0:
                    logger.warning("execute_with_rotation: no active keys configured")
                    break
                if not await self._wait_for_free_slot():
                    logger.warning(
                        "execute_with_rotation: all {} active keys in-flight (anti-stacking) after {:.2f}s",
                        initial_count,
                        time.monotonic() - started,
                    )
                    break
                continue

            any_attempt = True

            outcome: _DispatchOutcome | None = None
            try:
                outcome = await self._dispatch_one(
                    session, key, body, requested_model, stream
                )
            finally:
                await self._keys.release(key.id)

            # No more keys to try — the dispatcher decided to rotate.
            if outcome is None or outcome.rotate:
                # ``all_404`` stays ``True`` only if every single
                # attempt so far was a 404 "model unavailable" hint.
                if outcome is not None and not outcome.all_404_hint:
                    all_404 = False
                continue

            if outcome.terminal is not None:
                return outcome.terminal

            # Successful path (2xx) — pass the upstream response through.
            assert outcome.response is not None  # for type-checkers
            all_404 = False
            return self._build_success_response(outcome.response, stream)

        if not any_attempt:
            if initial_count == 0:
                logger.error("execute_with_rotation: no keys attempted (empty pool)")
                return self._json_error(503, "no active keys configured")
            logger.error(
                "execute_with_rotation: no keys attempted ({} active, all in-flight)",
                initial_count,
            )
            return self._json_error(
                503,
                "all backends busy, retry shortly",
            )

        if all_404 and requested_model:
            logger.warning(
                "execute_with_rotation: model '{}' is not available for any key",
                requested_model,
            )
            return self._json_error(
                404,
                f"model '{requested_model}' is not available for any active key",
            )

        if all_disabled:
            logger.error("execute_with_rotation: all keys disabled")
            return self._json_error(503, "all backends failed")

        logger.error("execute_with_rotation: exhausted all keys after {} attempts", max_iter)
        return self._json_error(503, "all backends failed")

    async def _count_active(self, session: AsyncSession) -> int:
        counts = await self._keys.count_by_status(session)
        return counts.get("active", 0)

    async def _wait_for_free_slot(self) -> bool:
        deadline = time.monotonic() + _SATURATION_WAIT_TOTAL
        slept = False
        while time.monotonic() < deadline:
            await asyncio.sleep(_SATURATION_BACKOFF)
            slept = True
            # Optimisation: bail early if the pool actually emptied.
            if self._keys.in_flight_count() == 0:
                return True
        return slept

    async def _dispatch_one(
        self,
        session: AsyncSession,
        key: Any,  # ApiKey — typed loosely to avoid an extra import here
        body: dict[str, Any],
        requested_model: str | None,
        stream: bool,
    ) -> _DispatchOutcome | None:
        try:
            raw_key = safe_decrypt(self._vault, key)
            response = await self._client.chat_completion(
                key=raw_key,
                key_preview=key.key_preview,
                body=body,
            )
        except VaultDecryptError as exc:
            logger.error(
                "execute_with_rotation: cannot decrypt key id={} err={}",
                exc.key_id,
                exc.inner,
            )
            return _DispatchOutcome(rotate=True, all_404_hint=False)
        except (httpx.TimeoutException, httpx.RequestError, OSError) as exc:
            logger.warning(
                "execute_with_rotation: network error key={} err={}",
                key.key_preview,
                exc,
            )
            await self._keys.record_server_error(session, key.id, 0)
            return _DispatchOutcome(rotate=True, all_404_hint=False)

        code = response.status_code

        # 2xx — success.
        if 200 <= code < 300:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            if not stream:
                prompt_tokens, completion_tokens = _parse_usage_from_response(
                    response
                )
            if prompt_tokens is not None or completion_tokens is not None:
                await self._keys.record_success_with_usage(
                    session,
                    key.id,
                    code,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            else:
                await self._keys.record_success(session, key.id, code)
            return _DispatchOutcome(response=response, rotate=False)

        # 401/403 — key is dead.
        if code in _DEAD_KEY_CODES:
            await self._keys.record_unauthorized(session, key.id, code)
            logger.info(
                "execute_with_rotation: key disabled key={} code={}",
                key.key_preview,
                code,
            )
            return _DispatchOutcome(rotate=True, all_404_hint=False)

        # 404 — model unavailable for this account; key stays active.
        if code == 404:
            logger.info(
                "execute_with_rotation: model unavailable for key={} model={}",
                key.key_preview,
                requested_model,
            )
            # Intentionally NOT calling any record_* method: the key
            # is fine, the model just doesn't exist for this account.
            return _DispatchOutcome(rotate=True, all_404_hint=True)

        # 429 — rate-limited; put the key in cooldown.
        if code == 429:
            retry_after = KeyManager.extract_retry_after(response.headers)
            await self._keys.record_rate_limited(
                session, key.id, code, retry_after
            )
            logger.info(
                "execute_with_rotation: rate limited key={} code={} cooldown={}s",
                key.key_preview,
                code,
                retry_after,
            )
            return _DispatchOutcome(rotate=True, all_404_hint=False)

        # 5xx — temporary, keep active, increment failures.
        if code >= 500:
            await self._keys.record_server_error(session, key.id, code)
            logger.warning(
                "execute_with_rotation: 5xx key={} code={}",
                key.key_preview,
                code,
            )
            return _DispatchOutcome(rotate=True, all_404_hint=False)

        return _DispatchOutcome(
            terminal=self._build_success_response(response, stream),
            rotate=False,
        )

    @staticmethod
    def _json_error(status_code: int, detail: str) -> Response:
        return Response(
            content=json.dumps({"detail": detail}).encode(),
            status_code=status_code,
            media_type="application/json",
        )

    @staticmethod
    def _build_success_response(response: httpx.Response, stream: bool) -> Response:
        headers = _safe_headers(response.headers)
        if stream:
            media_type = response.headers.get("content-type", "text/event-stream")
            return StreamingResponse(
                _iter_aiter(response),
                status_code=response.status_code,
                media_type=media_type,
                headers=headers,
            )
        # Non-streaming: read body fully and forward.
        content = response.content
        return Response(
            content=content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
            headers=headers,
        )

async def _iter_aiter(response: httpx.Response):
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    finally:
        await response.aclose()

def _parse_usage_from_response(response: httpx.Response) -> tuple[int | None, int | None]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        return (None, None)
    if not isinstance(payload, dict):
        return (None, None)

    usage = payload.get("usage")
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        prompt_tokens = int(pt) if isinstance(pt, int) else None
        completion_tokens = int(ct) if isinstance(ct, int) else None
        if prompt_tokens is not None or completion_tokens is not None:
            return (prompt_tokens, completion_tokens)

    # Ollama-native shape (kept for completeness; the proxy normally
    # receives the OpenAI-compatible response via /v1/chat/completions).
    pt = payload.get("prompt_eval_count")
    ct = payload.get("eval_count")
    prompt_tokens = int(pt) if isinstance(pt, int) else None
    completion_tokens = int(ct) if isinstance(ct, int) else None
    return (prompt_tokens, completion_tokens)
