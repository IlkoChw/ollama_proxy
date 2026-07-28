"""Tests for ApiKeyStatus (StrEnum) semantics.

Cover the migration from ``class ApiKeyStatus(str, Enum)`` to native
``class ApiKeyStatus(StrEnum)``. The migration must preserve:

* Equality with the underlying ``str`` (``ApiKeyStatus.ACTIVE == "active"``).
* ``str(member)`` returns the string value.
* ``.value`` still works (callers use it as SQLAlchemy default).
* ``ApiKeyStatus(string)`` lookups still work.
* Pydantic v2 schemas accept both ``str`` and ``ApiKeyStatus`` members.
* SQLAlchemy column default ``ApiKeyStatus.ACTIVE.value`` still produces a
  string in the row.
"""

from __future__ import annotations

from enum import StrEnum

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from app.db.base import Base
from app.models.api_key import ApiKey, ApiKeyStatus
from app.schemas.api_key import ApiKeyOut, ApiKeyUpdate
from app.services.vault import Vault

# ---------------------------------------------------------- enum semantics


def test_is_str_enum_subclass() -> None:
    """ApiKeyStatus must subclass ``StrEnum`` (not just ``(str, Enum)``)."""
    assert issubclass(ApiKeyStatus, StrEnum)


def test_equality_with_str() -> None:
    """Members must compare equal to their underlying string value."""
    assert ApiKeyStatus.ACTIVE == "active"
    assert ApiKeyStatus.DEPLETED == "depleted"
    assert ApiKeyStatus.DISABLED == "disabled"


def test_str_returns_value() -> None:
    """``str(member)`` returns the literal value (not ``ApiKeyStatus.ACTIVE``)."""
    assert str(ApiKeyStatus.ACTIVE) == "active"
    assert str(ApiKeyStatus.DEPLETED) == "depleted"


def test_value_attribute_preserved() -> None:
    """``Member.value`` is the source of truth and still equals the str."""
    assert ApiKeyStatus.ACTIVE.value == "active"
    assert ApiKeyStatus.DISABLED.value == "disabled"


def test_lookup_by_string() -> None:
    """``ApiKeyStatus('active')`` returns the corresponding member."""
    assert ApiKeyStatus("active") is ApiKeyStatus.ACTIVE
    assert ApiKeyStatus("depleted") is ApiKeyStatus.DEPLETED
    assert ApiKeyStatus("disabled") is ApiKeyStatus.DISABLED


def test_invalid_lookup_raises_value_error() -> None:
    """Unknown values raise ``ValueError`` (default Enum behaviour)."""
    with pytest.raises(ValueError):
        ApiKeyStatus("bogus")


def test_values_helper() -> None:
    """``ApiKeyStatus.values()`` returns the 3 expected strings."""
    members = ApiKeyStatus.values()
    assert members == ["active", "depleted", "disabled"]
    assert set(members) == {s.value for s in ApiKeyStatus}


# ------------------------------------------------------------- pydantic


def test_pydantic_accepts_string_status() -> None:
    """Pydantic schemas must accept both raw strings and members."""
    m = ApiKeyUpdate(status="active")
    assert m.status == ApiKeyStatus.ACTIVE
    # The resolved value is the enum member, not a string copy.
    assert isinstance(m.status, ApiKeyStatus)


def test_pydantic_accepts_enum_member() -> None:
    """Pydantic schemas accept an ApiKeyStatus member directly."""
    m = ApiKeyUpdate(status=ApiKeyStatus.DEPLETED)
    assert m.status == ApiKeyStatus.DEPLETED


def test_pydantic_accepts_none() -> None:
    """Optional updates still allow ``status=None`` (no change)."""
    m = ApiKeyUpdate(label="relabel", status=None)
    assert m.status is None


def test_pydantic_rejects_unknown_string() -> None:
    """Schema validation surfaces an unknown status as ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ApiKeyUpdate(status="bogus")


def test_pydantic_output_model_from_orm_normalises_status() -> None:
    """``ApiKeyOut.from_orm_key`` converts a DB ``str`` to the StrEnum member."""
    from datetime import UTC, datetime

    vault = Vault(fernet=Fernet(Fernet.generate_key()))
    raw = "sk-test-12345678"
    k = ApiKey(
        id=42,
        key_hash="a" * 64,
        key_prefix=raw[:8],
        key_encrypted=vault.encrypt(raw),
        label="acc42",
        status="depleted",  # raw string from DB row
        last_used_at=None,
        last_status_code=429,
        cooldown_until=datetime.now(UTC),
        total_requests=7,
        total_failures=1,
    )
    # Override ``created_at``/``updated_at`` since they're non-nullable.
    k.created_at = datetime.now(UTC)
    k.updated_at = datetime.now(UTC)

    out = ApiKeyOut.from_orm_key(k)
    assert out.status == ApiKeyStatus.DEPLETED
    assert isinstance(out.status, ApiKeyStatus)


# ----------------------------------------------------------- SQLAlchemy


async def test_sqlalchemy_default_uses_value() -> None:
    """The ``status`` column default produces the raw string in the DB."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    vault = Vault(fernet=Fernet(Fernet.generate_key()))
    raw = "sk-test-12345678"

    async with sm() as session:  # type: AsyncSession
        k = ApiKey(
            key_hash="a" * 64,
            key_prefix=raw[:8],
            key_encrypted=vault.encrypt(raw),
        )
        session.add(k)
        await session.commit()
        assert k.status == ApiKeyStatus.ACTIVE.value
        # Compare against the StrEnum member too — equality is symmetric.
        assert k.status == ApiKeyStatus.ACTIVE

    await engine.dispose()


async def test_sqlalchemy_accepts_enum_member_on_set() -> None:
    """Setting ``status`` to a StrEnum member persists as the str value."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    vault = Vault(fernet=Fernet(Fernet.generate_key()))
    raw = "sk-test-87654321"

    async with sm() as session:  # type: AsyncSession
        k = ApiKey(
            key_hash="b" * 64,
            key_prefix=raw[:8],
            key_encrypted=vault.encrypt(raw),
            status=ApiKeyStatus.DISABLED,
        )
        session.add(k)
        await session.commit()
        assert k.status == "disabled"
        assert k.status == ApiKeyStatus.DISABLED

    await engine.dispose()
