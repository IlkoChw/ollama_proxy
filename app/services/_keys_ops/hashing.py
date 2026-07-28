from __future__ import annotations

import hashlib

# Length of the masked ``key_prefix`` kept in the ``api_keys`` table.
# Must match the column width in ``app/models/api_key.py``.
KEY_PREFIX_LEN: int = 8

def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

def prefix_of(raw_key: str, n: int = KEY_PREFIX_LEN) -> str:
    return raw_key[:n]

__all__ = ["KEY_PREFIX_LEN", "hash_key", "prefix_of"]
