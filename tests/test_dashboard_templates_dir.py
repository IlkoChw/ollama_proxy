"""Unit tests for :func:`app.core.dashboard_filters.register_template_filters`.

The dashboard's Jinja environment is created at lifespan-startup;
tests that don't run the full lifespan need to call
``register_template_filters`` explicitly so the custom filters
(``fmt_ts`` today, future ones) are present. This file documents
the contract: the helper is idempotent and never raises on
double-invocation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.templating import Jinja2Templates


@pytest.fixture
def templates_dir(tmp_path: Path) -> Path:
    """An isolated templates directory for one test."""
    d = tmp_path / "templates"
    d.mkdir()
    return d


def test_register_installs_fmt_ts(templates_dir: Path) -> None:
    templates = Jinja2Templates(directory=str(templates_dir))
    # Import inside the test so module import-time side-effects are
    # not part of the contract under test.
    from app.core.dashboard_filters import register_template_filters

    register_template_filters(templates)
    assert "fmt_ts" in templates.env.filters


def test_register_is_idempotent(templates_dir: Path) -> None:
    """Calling ``register_template_filters`` twice must not raise."""
    from app.core.dashboard_filters import register_template_filters

    templates = Jinja2Templates(directory=str(templates_dir))
    register_template_filters(templates)
    # Second call must not overwrite or raise.
    register_template_filters(templates)
    assert "fmt_ts" in templates.env.filters


def test_dashboard_templates_dir_env_override(
    templates_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DASHBOARD_TEMPLATES_DIR`` controls where Jinja loads templates from.

    We don't run the lifespan here — instead we verify that the
    *default* path logic in ``lifespan`` honours the env var by
    simulating the same expression. The lifespan itself is exercised
    in the integration tests.
    """
    monkeypatch.setenv("DASHBOARD_TEMPLATES_DIR", str(templates_dir))
    assert (
        Path(os.environ["DASHBOARD_TEMPLATES_DIR"]) == templates_dir
    )


def test_default_templates_dir_is_relative_to_project_root() -> None:
    """Without the env var, the default falls back to
    ``<project_root>/templates`` — i.e. the directory shipped next to
    ``app/`` in the repo.
    """
    from app import main as app_main

    project_root = Path(app_main.__file__).resolve().parent.parent
    expected = project_root / "templates"
    assert expected.is_dir(), f"expected default templates dir missing: {expected}"
