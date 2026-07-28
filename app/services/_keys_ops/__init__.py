from __future__ import annotations

from .errors import (
    ApiKeyNotFoundError,
    DashboardClientError,
    KeysOpsError,
    ProbeModelMissingError,
    VaultDecryptError,
)
from .hashing import KEY_PREFIX_LEN, hash_key, prefix_of
from .health import build_health_snapshot
from .probe import ProbeClassifier, require_probe_model
from .probe_batch import probe_all_active
from .repository import load_or_404
from .vault import safe_decrypt

__all__ = [
    # errors
    "ApiKeyNotFoundError",
    "DashboardClientError",
    "KeysOpsError",
    "ProbeModelMissingError",
    "VaultDecryptError",
    # hashing
    "KEY_PREFIX_LEN",
    "hash_key",
    "prefix_of",
    # repository
    "load_or_404",
    # vault
    "safe_decrypt",
    # probe
    "ProbeClassifier",
    "require_probe_model",
    # health
    "build_health_snapshot",
    # probe_batch
    "probe_all_active",
]
