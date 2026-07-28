from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.api_key import ApiKey
from app.schemas.api_key import (
    AccountInfo,
    KeyUsageSnapshot,
    LocalCumsum,
    ModelUsageRow,
    OfficialUsage,
)
from app.services._keys_ops import VaultDecryptError, safe_decrypt
from app.services.ollama_client import OllamaClient
from app.services.vault import Vault

# --- public dataclasses ------------------------------------------------------

@dataclass(slots=True)
class UsageSnapshot:

    account: AccountInfo
    official: OfficialUsage

    def to_response(
        self,
        *,
        api_key: ApiKey,
        upstream_status: str = "ok",
        upstream_error: str | None = None,
    ) -> KeyUsageSnapshot:
        return KeyUsageSnapshot(
            id=api_key.id,
            label=api_key.label,
            key_preview=api_key.key_preview,
            account=self.account,
            official=self.official,
            local_cumsum=LocalCumsum(
                session_prompt_tokens=api_key.session_prompt_tokens,
                session_completion_tokens=api_key.session_completion_tokens,
                session_window_started_at=api_key.session_window_started_at,
                weekly_prompt_tokens=api_key.weekly_prompt_tokens,
                weekly_completion_tokens=api_key.weekly_completion_tokens,
                weekly_window_started_at=api_key.weekly_window_started_at,
                last_token_at=api_key.last_token_at,
            ),
            upstream_status=upstream_status,  # type: ignore[arg-type]
            upstream_error=upstream_error,
        )

# --- parser ------------------------------------------------------------------

def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # ollama.com emits RFC 3339 with a trailing Z.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

def _parse_account(payload: dict[str, Any], fetched_at: datetime) -> AccountInfo:
    return AccountInfo(
        account_id=payload.get("ID") or payload.get("id"),
        email=payload.get("Email") or payload.get("email"),
        name=payload.get("Name") or payload.get("name"),
        plan=payload.get("Plan") or payload.get("plan"),
        fetched_at=fetched_at,
    )

def _parse_official(payload: dict[str, Any], fetched_at: datetime) -> OfficialUsage:
    activity = payload.get("activity") if isinstance(payload, dict) else None
    limits = payload.get("limits") if isinstance(payload, dict) else None

    period_type: str | None = None
    period_start: datetime | None = None
    period_end: datetime | None = None
    if isinstance(activity, dict):
        period = activity.get("period")
        if isinstance(period, dict):
            period_type = period.get("type") if isinstance(period.get("type"), str) else None
            period_start = _parse_iso(period.get("starting_at"))
            period_end = _parse_iso(period.get("ending_at"))

    session_usage = 0
    weekly_usage = 0
    session_fraction: float | None = None
    weekly_fraction: float | None = None
    models_by_name: dict[str, ModelUsageRow] = {}

    if isinstance(limits, dict):
        for window_name, target in (("session", "session_usage"), ("weekly", "weekly_usage")):
            window = limits.get(window_name)
            if isinstance(window, dict):
                if "usage" in window:
                    usage_val = window["usage"]
                    if isinstance(usage_val, int | float):
                        usage_f = float(usage_val)
                        if target == "session_usage":
                            session_usage = int(usage_f)
                            session_fraction = usage_f
                        else:
                            weekly_usage = int(usage_f)
                            weekly_fraction = usage_f
                for entry in window.get("models", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name")
                    count = entry.get("request_count", 0)
                    if not isinstance(name, str) or not name:
                        continue
                    if not isinstance(count, int | float):
                        count = 0
                    row = models_by_name.get(name) or ModelUsageRow(name=name)
                    if window_name == "session":
                        row.session_request_count = int(count)
                    else:
                        row.weekly_request_count = int(count)
                    models_by_name[name] = row

    return OfficialUsage(
        session_usage=session_usage,
        weekly_usage=weekly_usage,
        session_usage_fraction=session_fraction,
        weekly_usage_fraction=weekly_fraction,
        models=list(models_by_name.values()),
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        fetched_at=fetched_at,
    )

# --- service -----------------------------------------------------------------

class UsageService:

    def __init__(
        self,
        *,
        ollama: OllamaClient,
        vault: Vault,
        ttl_seconds: int = 300,
    ) -> None:
        self._ollama = ollama
        self._vault = vault
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._cache: dict[int, tuple[float, UsageSnapshot]] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    # ----------------------------------------------------------- cache mgmt

    def _get_lock(self, key_id: int) -> asyncio.Lock:
        lock = self._locks.get(key_id)
        if lock is not None:
            return lock
        self._locks[key_id] = asyncio.Lock()
        return self._locks[key_id]

    def _cache_fresh(self, key_id: int) -> bool:
        entry = self._cache.get(key_id)
        if not entry:
            return False
        import time as _time

        ts, _snap = entry
        return (_time.monotonic() - ts) < self._ttl_seconds

    def _cache_put(self, key_id: int, snap: UsageSnapshot) -> None:
        import time as _time

        self._cache[key_id] = (_time.monotonic(), snap)

    def _cache_drop(self, key_id: int) -> None:
        self._cache.pop(key_id, None)

    # ----------------------------------------------------------- public API

    async def fetch_snapshot(
        self,
        *,
        api_key: ApiKey,
        force: bool = False,
    ) -> KeyUsageSnapshot:
        if not force and self._cache_fresh(api_key.id):
            cached = self._cache[api_key.id][1]
            return cached.to_response(api_key=api_key, upstream_status="ok")

        lock = self._get_lock(api_key.id)
        async with lock:
            # Re-check inside the lock — another coroutine may have
            # refreshed while we were waiting.
            if not force and self._cache_fresh(api_key.id):
                cached = self._cache[api_key.id][1]
                return cached.to_response(api_key=api_key, upstream_status="ok")

            try:
                raw_key = safe_decrypt(self._vault, api_key)
            except VaultDecryptError as exc:
                logger.warning(
                    "usage_service: cannot decrypt key id={} err={}",
                    api_key.id,
                    exc.inner,
                )
                return KeyUsageSnapshot(
                    id=api_key.id,
                    label=api_key.label,
                    key_preview=api_key.key_preview,
                    local_cumsum=_cumsum_from_row(api_key),
                    upstream_status="unreachable",
                    upstream_error=f"vault: {exc.inner}",
                )

            try:
                snap = await self._fetch_locked(api_key=api_key, raw_key=raw_key)
            except _AuthRejected:
                logger.warning(
                    "usage_service: upstream rejected key id={}",
                    api_key.id,
                )
                return KeyUsageSnapshot(
                    id=api_key.id,
                    label=api_key.label,
                    key_preview=api_key.key_preview,
                    local_cumsum=_cumsum_from_row(api_key),
                    upstream_status="unauthorised",
                    upstream_error="upstream returned 401/403",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "usage_service: fetch failed for key id={} err={}",
                    api_key.id,
                    exc,
                )
                stale = self._cache.get(api_key.id)
                if stale is not None:
                    return stale[1].to_response(
                        api_key=api_key,
                        upstream_status="unreachable",
                        upstream_error=str(exc),
                    )
                return KeyUsageSnapshot(
                    id=api_key.id,
                    label=api_key.label,
                    key_preview=api_key.key_preview,
                    local_cumsum=_cumsum_from_row(api_key),
                    upstream_status="unreachable",
                    upstream_error=str(exc),
                )
            else:
                self._cache_put(api_key.id, snap)
                return snap.to_response(api_key=api_key, upstream_status="ok")

    async def refresh_all_active(
        self,
        session: AsyncSession,
    ) -> list[KeyUsageSnapshot]:
        stmt = (
            select(ApiKey)
            .where(ApiKey.status == "active")
            .order_by(ApiKey.id)
        )
        result = await session.execute(stmt)
        keys = list(result.scalars().all())
        if not keys:
            return []

        snapshots = await asyncio.gather(
            *(self.fetch_snapshot(api_key=k, force=True) for k in keys),
            return_exceptions=False,
        )

        for k, snap in zip(keys, snapshots, strict=True):
            if snap.upstream_status != "ok":
                continue
            await self._persist(session=session, api_key=k, snapshot=snap)
        await session.commit()

        logger.info("usage_service: refresh_all_active: count={}", len(snapshots))
        return snapshots

    # ----------------------------------------------------------- internals

    async def persist_snapshot(
        self,
        *,
        session: AsyncSession,
        api_key: ApiKey,
        snapshot: KeyUsageSnapshot,
    ) -> None:
        if snapshot.upstream_status != "ok":
            return
        await self._persist(session=session, api_key=api_key, snapshot=snapshot)
        await session.commit()

    async def _fetch_locked(self, *, api_key: ApiKey, raw_key: str) -> UsageSnapshot:
        key_preview = api_key.key_preview
        try:
            account_payload = await self._ollama.get_account(
                key=raw_key, key_preview=key_preview
            )
            usage_payload = await self._ollama.get_usage(
                key=raw_key, key_preview=key_preview
            )
        except Exception as exc:
            # httpx.HTTPStatusError carries .response.status_code.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise _AuthRejected(status) from exc
            raise

        fetched_at = datetime.now(UTC)
        return UsageSnapshot(
            account=_parse_account(account_payload, fetched_at),
            official=_parse_official(usage_payload, fetched_at),
        )

    async def _persist(
        self,
        *,
        session: AsyncSession,
        api_key: ApiKey,
        snapshot: KeyUsageSnapshot,
    ) -> None:
        if snapshot.account is not None:
            api_key.account_email = snapshot.account.email
            api_key.account_name = snapshot.account.name
            api_key.account_plan = snapshot.account.plan
            api_key.account_id = snapshot.account.account_id
        if snapshot.official is not None:
            api_key.last_usage_fetch_at = snapshot.official.fetched_at
            api_key.last_usage_session = snapshot.official.session_usage
            api_key.last_usage_weekly = snapshot.official.weekly_usage
            if snapshot.official.session_usage_fraction is not None:
                api_key.last_usage_session_fraction = (
                    snapshot.official.session_usage_fraction
                )
            if snapshot.official.weekly_usage_fraction is not None:
                api_key.last_usage_weekly_fraction = (
                    snapshot.official.weekly_usage_fraction
                )
            api_key.last_usage_models_json = json.dumps(
                [m.model_dump() for m in snapshot.official.models],
                ensure_ascii=False,
            )

# --- helpers -----------------------------------------------------------------

def _cumsum_from_row(api_key: ApiKey) -> LocalCumsum:
    return LocalCumsum(
        session_prompt_tokens=api_key.session_prompt_tokens,
        session_completion_tokens=api_key.session_completion_tokens,
        session_window_started_at=api_key.session_window_started_at,
        weekly_prompt_tokens=api_key.weekly_prompt_tokens,
        weekly_completion_tokens=api_key.weekly_completion_tokens,
        weekly_window_started_at=api_key.weekly_window_started_at,
        last_token_at=api_key.last_token_at,
    )

class _AuthRejected(Exception):

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"upstream returned {status_code}")
