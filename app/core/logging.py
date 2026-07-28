from __future__ import annotations

import re
import sys
from typing import Any

from loguru import logger as _loguru_logger

_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "key_hash",
        "raw_key",
        "api_key",
        "authorization",
        "secret",
        "password",
        "bearer",
    }
)

# Display length for masked key previews: prefix (8) + ellipsis.
_PREVIEW_LEN = 8

_MAX_RECURSION_DEPTH = 3

_REDACT_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(k) for k in _FORBIDDEN_KEYS)
    + r")(\s*)([=:])(\s*)(\S+)"
)

def _preview(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    if not s:
        return ""
    if len(s) <= _PREVIEW_LEN:
        # Short values still get the ellipsis so the reader can tell at a
        # glance that the line has been sanitised.
        return s + "…"
    return s[:_PREVIEW_LEN] + "…"

def _mask_record(record: Any) -> None:
    extra = record.get("extra")
    if isinstance(extra, dict):
        _mask_dict(extra, depth=0)

    # Defensive: also scrub the raw message text in case a key was
    # interpolated without going through bind().
    message = record.get("message")
    if isinstance(message, str):
        record["message"] = _redact_message(message)

def _mask_dict(d: dict[str, Any], *, depth: int) -> None:
    if depth >= _MAX_RECURSION_DEPTH:
        return
    for k in list(d.keys()):
        v = d[k]
        if isinstance(k, str) and k.lower() in _FORBIDDEN_KEYS:
            d[k] = _preview(v)
            continue
        if isinstance(v, dict):
            _mask_dict(v, depth=depth + 1)
        elif isinstance(v, list):
            _mask_list(v, depth=depth)

def _mask_list(items: list[Any], *, depth: int) -> None:
    if depth >= _MAX_RECURSION_DEPTH:
        return
    for _i, v in enumerate(items):
        if isinstance(v, dict):
            _mask_dict(v, depth=depth + 1)
        elif isinstance(v, list):
            _mask_list(v, depth=depth)

def _redact_message(message: str) -> str:

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        ws_left = match.group(2)
        sep = match.group(3)
        ws_right = match.group(4)
        value = match.group(5)
        return f"{key}{ws_left}{sep}{ws_right}{_preview(value)}"

    return _REDACT_PATTERN.sub(_sub, message)

def setup_logging(level: str = "INFO") -> None:
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stdout,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    _loguru_logger.configure(patcher=_mask_record)

# Install a sane default sink at import time so early-logged events
# (e.g. during app startup before lifespan runs) are also masked.
setup_logging()

# Re-export the configured logger as the public name.
logger = _loguru_logger
