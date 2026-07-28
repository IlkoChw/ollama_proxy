from __future__ import annotations

from app.core.logging import logger
from app.models.api_key import ApiKey
from app.services.vault import Vault

from .errors import VaultDecryptError

__all__ = ["safe_decrypt"]

def safe_decrypt(vault: Vault, key: ApiKey) -> str:
    try:
        return vault.decrypt(key.key_encrypted)
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "_keys_ops.safe_decrypt: cannot decrypt key preview={} err={}",
            key.key_preview,
            exc,
        )
        raise VaultDecryptError(key_id=key.id, inner=exc) from exc
