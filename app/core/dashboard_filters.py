from __future__ import annotations

import datetime as _dt


def register_template_filters(templates: object) -> None:
    env = getattr(templates, "env", None)
    if env is None:
        return
    env.filters.setdefault("fmt_ts", _fmt_ts)

def _fmt_ts(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        if not value:
            return ""
        try:
            value = _dt.datetime.fromisoformat(value)
        except ValueError:
            # Unparseable — fall through and return the raw text
            # rather than blow up the whole row.
            return value
    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(_dt.UTC).replace(tzinfo=None)
        return value.strftime("%d.%m.%Y %H:%M:%S")
    return str(value)
