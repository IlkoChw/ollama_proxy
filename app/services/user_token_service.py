from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.models.user_token import UserToken, UserTokenStatus

# Public prefix advertised to clients and used in Swagger examples.
TOKEN_PREFIX: str = "opk_"

_RANDOM_BYTES: int = 32

_PREFIX_LEN: int = 12

# ----------------------------------------------------------- token format

def generate_raw_key() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(_RANDOM_BYTES)

def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

def prefix_of(raw_key: str) -> str:
    return raw_key[:_PREFIX_LEN]

# ----------------------------------------------------------- CRUD

async def create_token(
    session: AsyncSession,
    label: str,
    expires_at: datetime | None = None,
) -> tuple[UserToken, str]:
    if expires_at is not None and expires_at <= datetime.now(UTC):
        raise ValueError("expires_at must be in the future")

    raw = generate_raw_key()
    orm_token = UserToken(
        key_hash=hash_key(raw),
        key_prefix=prefix_of(raw),
        label=label,
        status=UserTokenStatus.ACTIVE.value,
        expires_at=expires_at,
    )
    session.add(orm_token)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        logger.error("create_token: integrity error label={!r}", label)
        raise ValueError("could not create user token (duplicate hash)") from exc
    await session.commit()
    await session.refresh(orm_token)
    logger.info("create_token: id={} prefix={}", orm_token.id, orm_token.key_preview)
    return orm_token, raw

async def list_tokens(session: AsyncSession) -> list[UserToken]:
    stmt = select(UserToken).order_by(UserToken.id.desc())
    result = await session.execute(stmt)
    return list(result.scalars().all())

async def get_token(session: AsyncSession, token_id: int) -> UserToken | None:
    return await session.get(UserToken, token_id)

async def update_token(
    session: AsyncSession,
    token: UserToken,
    *,
    label: str | None = None,
    expires_at: datetime | None = None,
    clear_expires_at: bool = False,
    status: UserTokenStatus | None = None,
) -> UserToken:
    if label is not None:
        token.label = label
    if clear_expires_at:
        token.expires_at = None
    elif expires_at is not None:
        token.expires_at = expires_at
    if status is not None:
        token.status = status.value
    await session.commit()
    await session.refresh(token)
    logger.info("update_token: id={} preview={}", token.id, token.key_preview)
    return token

async def revoke_token(session: AsyncSession, token: UserToken) -> UserToken:
    token.status = UserTokenStatus.REVOKED.value
    await session.commit()
    await session.refresh(token)
    logger.info("revoke_token: id={} preview={}", token.id, token.key_preview)
    return token

# ----------------------------------------------------- authentication path

def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)

async def has_any_active_token(session: AsyncSession) -> bool:
    stmt = select(func.count()).select_from(UserToken).where(
        UserToken.status == UserTokenStatus.ACTIVE.value
    )
    result = await session.execute(stmt)
    count = result.scalar_one()
    return bool(count)

async def authenticate(session: AsyncSession, raw_key: str) -> UserToken | None:
    if not isinstance(raw_key, str) or not raw_key:
        raise ValueError("raw_key must be a non-empty string")

    h = hash_key(raw_key)
    stmt = select(UserToken).where(UserToken.key_hash == h)
    result = await session.execute(stmt)
    token: UserToken | None = result.scalar_one_or_none()
    if token is None:
        return None
    if token.status != UserTokenStatus.ACTIVE.value:
        return None
    if token.expires_at is not None and _as_utc(token.expires_at) <= datetime.now(UTC):
        return None

    now = datetime.now(UTC)
    bump = (
        update(UserToken)
        .where(UserToken.id == token.id)
        .values(total_requests=UserToken.total_requests + 1, last_used_at=now)
        .returning(UserToken.total_requests, UserToken.last_used_at)
    )
    bumped = await session.execute(bump)
    row = bumped.mappings().first()
    if row is not None:
        token.total_requests = row["total_requests"]
        token.last_used_at = _as_utc(row["last_used_at"])  # type: ignore[arg-type]
    await session.commit()
    return token

__all__: list[str] = [
    "TOKEN_PREFIX",
    "authenticate",
    "create_token",
    "generate_raw_key",
    "get_token",
    "has_any_active_token",
    "hash_key",
    "list_tokens",
    "prefix_of",
    "revoke_token",
    "update_token",
]  # noqa: WPS410

