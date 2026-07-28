from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin_token
from app.db.session import get_session
from app.models.user_token import UserTokenStatus
from app.schemas.user_token import (
    UserTokenCreate,
    UserTokenCreated,
    UserTokenOut,
    UserTokenUpdate,
)
from app.services.user_token_service import (
    create_token,
    get_token,
    list_tokens,
    revoke_token,
    update_token,
)

router = APIRouter(
    prefix="/admin/user-tokens",
    tags=["user-tokens"],
    dependencies=[Depends(require_admin_token)],
)

# --------------------------------------------------------------------- create

@router.post(
    "",
    response_model=UserTokenCreated,
    status_code=status.HTTP_201_CREATED,
    summary="Mint a new end-user token (raw key returned once)",
)
async def create_user_token(
    payload: UserTokenCreate,
    session: AsyncSession = Depends(get_session),
) -> UserTokenCreated:
    try:
        orm_token, raw_key = await create_token(
            session=session,
            label=payload.label,
            expires_at=payload.expires_at,
        )
    except ValueError as exc:
        # ``create_token`` raises ``ValueError`` for expires_at-in-the-past
        # or for an IntegrityError on hash collision. Map both to 422.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return UserTokenCreated(
        id=orm_token.id,
        label=orm_token.label,
        key_preview=orm_token.key_preview,
        status=UserTokenStatus(orm_token.status),
        expires_at=orm_token.expires_at,
        last_used_at=orm_token.last_used_at,
        total_requests=orm_token.total_requests,
        created_at=orm_token.created_at,
        updated_at=orm_token.updated_at,
        raw_key=raw_key,
    )

# ----------------------------------------------------------------------- list

@router.get(
    "",
    response_model=list[UserTokenOut],
    summary="List user tokens (masked)",
)
async def list_user_tokens(
    session: AsyncSession = Depends(get_session),
) -> list[UserTokenOut]:
    tokens = await list_tokens(session)
    return [UserTokenOut.from_orm_token(t) for t in tokens]

# --------------------------------------------------------------------- detail

@router.get(
    "/{token_id}",
    response_model=UserTokenOut,
    summary="Show one user token (masked)",
)
async def get_user_token(
    token_id: int = Path(..., ge=1),
    session: AsyncSession = Depends(get_session),
) -> UserTokenOut:
    orm_token = await get_token(session, token_id)
    if orm_token is None:
        raise HTTPException(status_code=404, detail="user token not found")
    return UserTokenOut.from_orm_token(orm_token)

# --------------------------------------------------------------------- patch

@router.patch(
    "/{token_id}",
    response_model=UserTokenOut,
    summary="Update user token fields (label / expires_at / status)",
)
async def patch_user_token(
    payload: UserTokenUpdate,
    token_id: int = Path(..., ge=1),
    session: AsyncSession = Depends(get_session),
) -> UserTokenOut:
    orm_token = await get_token(session, token_id)
    if orm_token is None:
        raise HTTPException(status_code=404, detail="user token not found")
    if not payload.model_fields_set:
        # No fields provided → return the current state. Avoids a no-op
        # write that would bump ``updated_at`` for nothing.
        return UserTokenOut.from_orm_token(orm_token)
    clear_expiry = "expires_at" in payload.model_fields_set and payload.expires_at is None
    new_expiry = None if clear_expiry else payload.expires_at
    try:
        updated = await update_token(
            session,
            orm_token,
            label=payload.label if "label" in payload.model_fields_set else None,
            expires_at=new_expiry,
            clear_expires_at=clear_expiry,
            status=payload.status if "status" in payload.model_fields_set else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return UserTokenOut.from_orm_token(updated)

# -------------------------------------------------------------------- revoke

@router.delete(
    "/{token_id}",
    response_model=UserTokenOut,
    summary="Revoke a user token (soft-delete)",
)
async def revoke_user_token(
    token_id: int = Path(..., ge=1),
    session: AsyncSession = Depends(get_session),
) -> UserTokenOut:
    orm_token = await get_token(session, token_id)
    if orm_token is None:
        raise HTTPException(status_code=404, detail="user token not found")
    revoked = await revoke_token(session, orm_token)
    return UserTokenOut.from_orm_token(revoked)
