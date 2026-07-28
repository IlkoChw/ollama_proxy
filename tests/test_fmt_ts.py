"""Unit tests for the ``fmt_ts`` Jinja filter on the dashboard.

The filter renders timestamps in ``DD.MM.YYYY HH:MM:SS`` for UI display
while the proxy's API contract still ships ISO-8601 strings in JSON.
The tests verify the filter's input contract: it must never raise —
unparseable input falls back to ``str(value)``, empty / ``None``
become empty strings.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.core.dashboard_filters import _fmt_ts


class TestFmtTsEmpty:
    """Empty / null inputs render as empty strings."""

    def test_none_returns_empty(self) -> None:
        assert _fmt_ts(None) == ""

    def test_empty_string_returns_empty(self) -> None:
        assert _fmt_ts("") == ""


class TestFmtTsFromIsoString:
    """ISO-8601 strings are parsed and reformatted."""

    def test_utc_with_microseconds(self) -> None:
        assert _fmt_ts("2026-07-26T21:43:12.751321+00:00") == "26.07.2026 21:43:12"

    def test_utc_z_form(self) -> None:
        # Pydantic v2 emits ``Z`` for UTC datetimes in JSON.
        assert _fmt_ts("2026-07-26T21:43:12.751321Z") == "26.07.2026 21:43:12"

    def test_naive_iso(self) -> None:
        # No tz info: rendered as-is, no UTC conversion.
        assert _fmt_ts("2026-07-26T21:43:12") == "26.07.2026 21:43:12"

    def test_positive_offset_normalised_to_utc(self) -> None:
        # 21:43 in Tokyo (+09:00) → 12:43 UTC same day.
        assert _fmt_ts("2026-07-26T21:43:12+09:00") == "26.07.2026 12:43:12"

    def test_negative_offset_normalised_to_utc(self) -> None:
        # 18:43 in NY (-04:00) → 22:43 UTC same day.
        assert _fmt_ts("2026-07-26T18:43:12-04:00") == "26.07.2026 22:43:12"


class TestFmtTsFromDatetime:
    """Datetime objects are reformatted directly."""

    def test_aware_utc(self) -> None:
        assert _fmt_ts(datetime(2026, 7, 26, 21, 43, 12, tzinfo=UTC)) == "26.07.2026 21:43:12"

    def test_aware_non_utc_normalised(self) -> None:
        tokyo = timezone(timedelta(hours=9))
        assert _fmt_ts(datetime(2026, 7, 26, 21, 43, 12, tzinfo=tokyo)) == "26.07.2026 12:43:12"

    def test_naive_datetime_renders_as_is(self) -> None:
        # Naive datetimes: assumed UTC, no conversion.
        assert _fmt_ts(datetime(2026, 7, 26, 21, 43, 12)) == "26.07.2026 21:43:12"


class TestFmtTsFallback:
    """Anything else falls back to ``str(value)`` without raising."""

    def test_unparseable_string_returns_raw(self) -> None:
        assert _fmt_ts("not-a-date") == "not-a-date"

    def test_int_returns_str(self) -> None:
        assert _fmt_ts(123) == "123"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Latest of the year.
        ("2026-12-31T23:59:59+00:00", "31.12.2026 23:59:59"),
        # Single-digit day/month: the format string zero-pads both,
        # so ``1.1.2027 00:00:01`` is the rendered form (parser is
        # strict; format string is well-defined).
        ("2027-01-01T00:00:01+00:00", "01.01.2027 00:00:01"),
    ],
)
def test_fmt_ts_format_string_is_zero_padded(value: str, expected: str) -> None:
    assert _fmt_ts(value) == expected
