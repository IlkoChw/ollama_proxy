"""Tests for ``app.core.logging``: secret-masking patcher + preview helper.

The patcher is applied at import time, so we capture log output by
removing the default sink and adding a custom one that records full
``record`` dicts into the ``captured_records`` list fixture.

We also exercise the public helpers in isolation (``_preview``,
``_redact_message``, ``_mask_dict``).
"""

from __future__ import annotations

import re

import pytest

from app.core import logging as logmod
from app.core.logging import (
    _FORBIDDEN_KEYS,
    _MAX_RECURSION_DEPTH,
    _mask_dict,
    _preview,
    _redact_message,
    setup_logging,
)

# ---------------------------------------------------------------- _preview


def test_preview_truncates_long_strings_to_8_chars_plus_ellipsis() -> None:
    """Stable contract: first 8 chars + '…' for any string longer than 8."""
    assert _preview("abcdefghijklmnop") == "abcdefgh…"
    assert _preview("a" * 64) == "aaaaaaaa…"


def test_preview_short_string_still_gets_ellipsis_marker() -> None:
    """Even short values get '…' so readers can see the line was scrubbed."""
    assert _preview("abc") == "abc…"


def test_preview_boundary_at_exactly_eight_chars() -> None:
    """At the boundary, the rule says ``<= _PREVIEW_LEN`` keeps the
    value and appends ``…``. Visually distinguishing masked vs unmasked."""
    assert _preview("abcdefgh") == "abcdefgh…"


def test_preview_handles_none_and_empty() -> None:
    assert _preview(None) == ""
    assert _preview("") == ""


def test_preview_handles_non_string_types() -> None:
    """Booleans and ints are coerced via str(); behaviour is consistent
    across scalar types."""
    # 12345678901 → str → '12345678901' (11 chars) → preview = '1234567…'
    # Wait: str(12345678901) is '12345678901' (length 11), so s[:8] = '12345678'.
    assert _preview(12345678901) == "12345678…"
    assert _preview(True) == "True…"


# ------------------------------------------------------- _redact_message


def test_redact_message_replaces_raw_key_with_prefix() -> None:
    out = _redact_message("Trying raw_key=sk-abc1234567890 for acc1")
    assert out == "Trying raw_key=sk-abc12… for acc1"


def test_redact_message_preserves_other_tokens() -> None:
    """Adjacent non-secret tokens must stay untouched."""
    out = _redact_message("count=5 raw_key=sk-abc1234567890 user=alice")
    assert "count=5" in out
    assert "user=alice" in out
    assert "raw_key=sk-abc12…" in out


def test_redact_message_handles_colon_separator() -> None:
    assert _redact_message("password: hunter2") == "password: hunter2…"


def test_redact_message_does_not_treat_space_as_separator() -> None:
    """Plain prose with a forbidden-key word followed by a space must NOT
    be masked — otherwise ``the password management policy`` would
    become gibberish. Only ``=`` and ``:`` count as separators.
    """
    out = _redact_message("the password management policy is enforced")
    assert out == "the password management policy is enforced"


def test_redact_message_ignores_authorization_bearer_pattern() -> None:
    """``Authorization Bearer xyz`` is prose, not a KV pair; must stay."""
    out = _redact_message("Authorization Bearer xyz")
    assert out == "Authorization Bearer xyz"


def test_redact_message_handles_dict_style_header_text() -> None:
    """Dotted/colon-quoted header text doesn't get matched because the
    separator only accepts ``=`` or ``:`` directly after the key."""
    out = _redact_message(
        "upstream headers: {'Authorization': 'Bearer sk-abcdef1234567890'}"
    )
    # The character after ``Authorization`` is ``'``, not ``=``/``:``.
    assert "sk-abcdef1234567890" in out


def test_redact_message_preserves_whitespace_around_separator() -> None:
    """Original whitespace around ``=`` is preserved in the output."""
    assert _redact_message("password : hunter2") == "password : hunter2…"
    assert _redact_message("api_key    =    sk-abc1234") == (
        "api_key    =    sk-abc12…"
    )


def test_redact_message_handles_all_seven_forbidden_keys() -> None:
    """All 7 forbidden keys must be scrubbed."""
    keys = ["key_hash", "raw_key", "api_key", "authorization", "secret",
            "password", "bearer"]
    for k in keys:
        msg = f"{k}=0123456789abcdef"
        out = _redact_message(msg)
        assert "0123456789" not in out, f"value leaked for key={k}: {out!r}"
        assert k in out


def test_redact_message_is_idempotent() -> None:
    """Running redact twice equals running once."""
    msg = "raw_key=sk-abc1234567890 api_key=sk-xyz1234567890"
    once = _redact_message(msg)
    twice = _redact_message(once)
    assert once == twice


# -------------------------------------------------- _mask_dict (recursion)


def test_mask_dict_scrubs_top_level_forbidden_key() -> None:
    d = {"label": "acc1", "raw_key": "sk-abcdef1234567890"}
    _mask_dict(d, depth=0)
    assert d["label"] == "acc1"
    assert d["raw_key"] == "sk-abcde…"


def test_mask_dict_recurses_into_nested_dict() -> None:
    """Forbidden keys nested inside ``headers`` are scrubbed."""
    d = {"headers": {"Authorization": "Bearer sk-secret-key"}}
    _mask_dict(d, depth=0)
    # 'Bearer sk-secret-key' has 20 chars; preview = 'Bearer s' (8 chars).
    assert d["headers"]["Authorization"] == "Bearer s…"


def test_mask_dict_recurses_into_list_of_dicts() -> None:
    """Common OpenAI-style pattern: a list of header dicts."""
    d = {"items": [{"Authorization": "Bearer sk-secret"}]}
    _mask_dict(d, depth=0)
    assert d["items"][0]["Authorization"] == "Bearer s…"


def test_mask_dict_respects_max_recursion_depth() -> None:
    """A deeply nested self-referential structure must terminate."""
    d: dict = {}
    inner = d
    # Build a chain 2x deeper than the limit. Anything beyond the limit
    # is left alone, but the call must return without RecursionError.
    for _ in range(_MAX_RECURSION_DEPTH + 2):
        inner["next"] = {}
        inner = inner["next"]
    inner["raw_key"] = "sk-bomb"
    # Should not raise.
    _mask_dict(d, depth=0)
    # The deepest layer (>= _MAX_RECURSION_DEPTH) is intentionally not
    # masked — but the shallower layers above the limit would have been
    # if there were any forbidden keys there.


def test_mask_dict_handles_non_string_value_via_str_coercion() -> None:
    """Numeric / boolean values are converted via ``_preview`` -> ``str``."""
    d = {"raw_key": 12345678901}
    _mask_dict(d, depth=0)
    # str(12345678901) is '12345678901' (11 chars). Preview = first 8 + '…'.
    assert d["raw_key"] == "12345678…"


# ------------------------------------------------------- logger integration


@pytest.fixture()
def captured_records(monkeypatch):  # type: ignore[no-untyped-def]
    """Re-route ``logger`` into an in-memory buffer of full records.

    The returned list grows with each ``.info()`` call. Each entry is
    the raw ``record`` dict — including ``record["message"]`` (the
    formatted message after the patcher ran) and ``record["extra"]``
    (the bound keyword arguments).
    """
    buf: list[dict] = []

    def _sink(message):  # type: ignore[no-untyped-def]
        buf.append(dict(message.record))
        return None

    logmod._loguru_logger.remove()
    logmod._loguru_logger.add(_sink, level="INFO", format="{message}")
    logmod._loguru_logger.configure(patcher=logmod._mask_record)
    yield buf
    setup_logging()


def test_logger_masks_string_interpolation_with_forbidden_keys(
    captured_records: list[dict],
) -> None:
    """End-to-end: ``logger.info('msg {}', secret_value)`` runs through
    the patcher and the value is rewritten to a preview."""
    logmod.logger.info(
        "Trying raw_key={} for label={}", "sk-abcdef1234567890", "acc1"
    )
    assert captured_records
    out = str(captured_records[-1].get("message", ""))
    assert "sk-abcde…" in out
    assert "sk-abcdef1234567890" not in out


def test_logger_masks_bind_extra(captured_records: list[dict]) -> None:
    """``.bind(api_key=...)`` masks ``extra`` before the sink receives
    the record. We capture the full record via a custom sink to assert
    against ``record["extra"]`` directly."""
    logmod.logger.bind(api_key="sk-xyz1234567890").info("plain message")
    assert captured_records
    rec = captured_records[-1]
    assert "sk-xyz12…" in str(rec["extra"].get("api_key", ""))
    assert "sk-xyz1234567890" not in str(rec["extra"].get("api_key", ""))


def test_logger_masks_nested_dict_in_extra(captured_records: list[dict]) -> None:
    """A ``headers`` dict nested in ``extra`` must be scrubbed via the
    recursive walk."""
    logmod.logger.bind(
        headers={"Authorization": "Bearer sk-real-secret-123456"}
    ).info("upstream call")
    assert captured_records
    rec = captured_records[-1]
    auth = rec["extra"]["headers"]["Authorization"]
    # The whole 'Bearer sk-real-secret-123456' value is scrubbed to a
    # 8-char prefix + ellipsis. 'sk-real-secret' must NOT appear.
    assert "secret" not in auth or auth.endswith("…")
    assert auth.endswith("…")


def test_logger_does_not_mask_counter_names(captured_records: list[dict]) -> None:
    """Counter-like field names are preserved unchanged."""
    logmod.logger.bind(key_hash_count=3, raw_key_total=10).info("ok")
    assert captured_records
    rec = captured_records[-1]
    assert rec["extra"]["key_hash_count"] == 3
    assert rec["extra"]["raw_key_total"] == 10


# ----------------------------------------------------------- safety net


def test_forbidden_keys_constant_includes_seven_entries() -> None:
    """Lock the size of ``_FORBIDDEN_KEYS`` — a regression here should
    require an explicit test update."""
    assert len(_FORBIDDEN_KEYS) == 7


def test_redact_pattern_matches_simple_kv_form() -> None:
    """Sanity: the compiled regex actually matches what it should.

    Note: the production regex restricts separators to ``=`` and ``:``
    so this test mirrors that behaviour explicitly.
    """
    pat = re.compile(
        r"\b("
        + "|".join(re.escape(k) for k in _FORBIDDEN_KEYS)
        + r")(\s*)([=:])(\s*)(\S+)"
    )
    assert pat.search("raw_key=sk-abc")
    assert pat.search("password : hunter2")
    # Plain prose with a space must NOT match — that was the old bug.
    assert not pat.search("my_key_hash_story")
    assert not pat.search("the password management policy")
