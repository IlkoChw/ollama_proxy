"""Tests for the master-key file permission check in :mod:`app.services.vault`.

The check is purely diagnostic — it never raises. On POSIX hosts it
warns when the file is readable beyond the owner (mode bits in
group/other). On Windows the check is a no-op (NTFS ACLs are not
introspected via ``stat``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.services.vault import _check_key_file_perms


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only behaviour")
def test_check_key_file_perms_warns_when_world_readable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A world-readable master key file produces a single warning log line."""
    import logging

    key_path = tmp_path / ".master_key"
    key_path.write_text("dummy-key")
    os.chmod(key_path, 0o644)  # group + other readable

    with caplog.at_level(logging.WARNING, logger="app.services.vault"):
        _check_key_file_perms(key_path)

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert records, "expected at least one warning record"
    msg = records[0].getMessage()
    assert "readable by group/other" in msg
    # The mode bits must appear so the operator can act on them.
    assert "0o644" in msg


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only behaviour")
def test_check_key_file_perms_silent_when_owner_only(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A 0o600 (owner-only) master key file produces NO warnings."""
    import logging

    key_path = tmp_path / ".master_key"
    key_path.write_text("dummy-key")
    os.chmod(key_path, 0o600)

    with caplog.at_level(logging.WARNING, logger="app.services.vault"):
        _check_key_file_perms(key_path)

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not records, f"expected no warnings for 0o600, got: {[r.getMessage() for r in records]}"


def test_check_key_file_perms_no_op_on_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing file must not raise (the function is best-effort)."""
    import logging

    key_path = tmp_path / ".does_not_exist"
    with caplog.at_level(logging.WARNING, logger="app.services.vault"):
        # Must not raise.
        _check_key_file_perms(key_path)
    # We do not assert on warnings here — on Windows the function is a
    # silent no-op; on POSIX a stat() failure produces one warning. Either
    # is acceptable; the only contract is "no exception".
