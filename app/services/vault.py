from __future__ import annotations

import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.core.logging import logger


class Vault:

    def __init__(self, fernet: Fernet | None) -> None:
        self._fernet = fernet

    @property
    def is_persistent(self) -> bool:
        return self._fernet is not None

    def encrypt(self, raw_key: str) -> bytes:
        if self._fernet is None:
            raise RuntimeError("Vault is not initialised (no master key)")
        return self._fernet.encrypt(raw_key.encode())

    def decrypt(self, blob: bytes) -> str:
        if self._fernet is None:
            raise RuntimeError("Vault is not initialised (no master key)")
        try:
            return self._fernet.decrypt(blob).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError(
                "encrypted blob cannot be decrypted (wrong master key?)"
            ) from exc

# ----------------------------------------------------------------- resolution

def _read_master_key_from_env() -> bytes | None:
    raw = os.environ.get("OLLAMA_PROXY_MASTER_KEY", "").strip()
    if not raw:
        return None
    try:
        Fernet(raw.encode("ascii"))  # validates format
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "OLLAMA_PROXY_MASTER_KEY env var is set but not a valid Fernet key. "
            "Generate one with: python -c 'from app.services.vault import "
            "generate_master_key; print(generate_master_key())'"
        ) from exc
    return raw.encode("ascii")

def _read_master_key_from_file(path: Path) -> bytes | None:
    if not path.exists():
        return None
    _check_key_file_perms(path)
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    return raw.encode("ascii")

def _write_master_key_to_file(path: Path, key: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key.decode("ascii"), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows or non-POSIX fs — best effort.
        pass
    _check_key_file_perms(path)
    logger.info("vault: master key written to {}", path)

def _check_key_file_perms(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError as exc:
        logger.warning("vault: cannot stat master key file {}: {}", path, exc)
        return
    leak_bits = mode & 0o077
    if leak_bits:
        # Human-readable mode string (e.g. "0o644").
        logger.warning(
            "vault: master key file {} is readable by group/other (mode={:#o}, "
            "leak_bits={:#o}). Restrict with `chmod 600` or move "
            "OLLAMA_PROXY_MASTER_KEY into the environment.",
            path,
            mode,
            leak_bits,
        )

def resolve_master_key(db_path: str | None = None) -> bytes:
    env_key = _read_master_key_from_env()
    if env_key is not None:
        logger.info("vault: using OLLAMA_PROXY_MASTER_KEY from env")
        return env_key

    if db_path:
        key_path = Path(db_path).expanduser().resolve().parent / ".master_key"
        file_key = _read_master_key_from_file(key_path)
        if file_key is not None:
            logger.info("vault: using master key from {}", key_path)
            return file_key
        new_key = Fernet.generate_key()
        _write_master_key_to_file(key_path, new_key)
        return new_key

    logger.warning(
        "vault: no master key resolved (set OLLAMA_PROXY_MASTER_KEY or "
        "ensure data/.master_key is writable). Keys will be LOST on restart."
    )
    return Fernet.generate_key()

def build_vault(db_path: str | None = None) -> Vault:
    key = resolve_master_key(db_path)
    return Vault(fernet=Fernet(key))

# ----------------------------------------------------------------- singleton

_vault_override: Vault | None = None

def get_vault() -> Vault:
    global _vault_override  # noqa: PLW0603
    if _vault_override is None:
        _vault_override = build_vault()
    return _vault_override

def set_vault(vault: Vault) -> None:
    global _vault_override  # noqa: PLW0603
    _vault_override = vault

def is_vault_initialised() -> bool:
    return _vault_override is not None

def reset_vault_for_tests() -> None:
    global _vault_override  # noqa: PLW0603
    _vault_override = None

def generate_master_key() -> str:
    return Fernet.generate_key().decode("ascii")
