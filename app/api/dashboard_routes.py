from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.deps_dashboard import (
    get_dashboard_auth_dep,
    get_dashboard_session,
    require_dashboard_csrf,
)
from app.core.config import get_settings
from app.core.logging import logger
from app.models.user_token import UserTokenStatus
from app.services.dashboard_auth import (
    CSRF_COOKIE,
    FLASH_COOKIE,
    SESSION_COOKIE,
    DashboardAuth,
    FlashLevel,
)
from app.services.dashboard_backend import (
    DashboardClientError,
    InProcessDashboardBackend,
)

# Status values we want to show with a colour hint in the api-keys table.
_VALID_STATUSES = {"active", "depleted", "disabled"}

_STATUS_ALL = "all"

_DEFAULT_STATUS_FILTER = "active"

# User-token status filter — narrower set than api keys (no depleted/disabled).
_VALID_USER_TOKEN_STATUSES = {s.value for s in UserTokenStatus}

_DEFAULT_USER_TOKEN_STATUS_FILTER = UserTokenStatus.ACTIVE.value

# --------------------------------------------------------------- helpers

def _client_dep(request: Request) -> InProcessDashboardBackend:
    state_client = getattr(request.app.state, "http_client", None)
    usage_service = getattr(request.app.state, "usage_service", None)
    if state_client is None or not isinstance(state_client, httpx.AsyncClient):
        raise RuntimeError(
            "dashboard backend not initialised; lifespan did not run "
            "(http_client missing on app.state)"
        )
    if usage_service is None:
        raise RuntimeError(
            "dashboard backend not initialised; usage_service missing "
            "on app.state"
        )
    return InProcessDashboardBackend(
        http_client=state_client,
        usage_service=usage_service,
    )

def _templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates

def _redirect(path: str) -> RedirectResponse:
    return RedirectResponse(url=path, status_code=status.HTTP_303_SEE_OTHER)

def _set_session_cookie(
    response: RedirectResponse | HTMLResponse,
    auth: DashboardAuth,
    value: str,
) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=value,
        max_age=auth.session_ttl_seconds,
        httponly=True,
        secure=auth.cookie_secure,
        samesite="lax",
        path="/",
    )

def _set_csrf_cookie(
    response: RedirectResponse | HTMLResponse,
    value: str,
    *,
    auth: DashboardAuth | None = None,
) -> None:
    secure = auth.cookie_secure if auth is not None else False
    response.set_cookie(
        key=CSRF_COOKIE,
        value=value,
        max_age=24 * 3600,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )

def _clear_session_cookie(
    response: RedirectResponse | HTMLResponse,
    auth: DashboardAuth,
) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        secure=auth.cookie_secure,
        samesite="lax",
        httponly=True,
    )

def _clear_csrf_cookie(
    response: RedirectResponse | HTMLResponse,
    auth: DashboardAuth | None = None,
) -> None:
    secure = auth.cookie_secure if auth is not None else False
    response.delete_cookie(key=CSRF_COOKIE, path="/", samesite="lax", secure=secure, httponly=True)

def _clear_flash_cookie(
    response: RedirectResponse | HTMLResponse,
) -> None:
    response.delete_cookie(key=FLASH_COOKIE, path="/", samesite="lax")

def _set_flash(
    response: RedirectResponse | HTMLResponse,
    auth: DashboardAuth,
    level: FlashLevel,
    msg: str,
) -> None:
    response.set_cookie(
        key=FLASH_COOKIE,
        value=auth.flash_put(level, msg),
        max_age=60,
        httponly=True,
        secure=auth.cookie_secure,
        samesite="lax",
        path="/",
    )

def _read_flash(
    request: Request,
    response: RedirectResponse | HTMLResponse,
    auth: DashboardAuth,
) -> dict[str, Any] | None:
    raw = request.cookies.get(FLASH_COOKIE, "")
    flash = auth.flash_read(raw)
    _clear_flash_cookie(response)
    return flash

def _copy_set_cookies(src: HTMLResponse, dst: HTMLResponse) -> None:
    for header_name, header_value in src.headers.items():
        if header_name.lower() == "set-cookie":
            dst.headers.append(header_name, header_value)

def _client_or_503(request: Request):
    try:
        return _client_dep(request)
    except RuntimeError as exc:
        logger.error("dashboard: backend not initialised: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard not fully initialised",
        ) from exc

# --------------------------------------------------------------- router

router = APIRouter(tags=["dashboard"])

# --------------------------------------------------------------- login

@router.get("/dashboard/login", include_in_schema=False)
async def login_form(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is not None:
        # Already logged in: skip the form.
        return _redirect("/dashboard")
    templates = _templates(request)
    flash: dict[str, Any] | None = None

    def _render() -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "dashboard_login.html",
            {
                "request": request,
                "error": None,
                "flash": flash,
            },
        )

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    for header_name, header_value in probe.headers.items():
        if header_name.lower() == "set-cookie":
            response.headers.append(header_name, header_value)
    return response

@router.post("/dashboard/login", include_in_schema=False)
async def login_submit(
    request: Request,
    password: str = Form(..., min_length=1, max_length=512),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    expected = get_settings().dashboard_password
    if not expected:
        # Misconfiguration: the lifespan should have refused to start
        # in this state. Fail closed.
        logger.error("login_submit: DASHBOARD_PASSWORD is not set")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="dashboard password is not configured",
        )
    # Constant-time comparison to avoid timing oracles.
    import hmac

    if not hmac.compare_digest(
        password.encode("utf-8"), expected.encode("utf-8")
    ):
        response = _redirect("/dashboard/login")
        _set_flash(response, auth, "error", "Invalid password.")
        return response
    # Mint a fresh session + CSRF.
    session_cookie = auth.issue_session()
    session_id = auth.verify_session(session_cookie) or ""
    csrf = auth.issue_csrf(session_id)
    response = _redirect("/dashboard")
    _set_session_cookie(response, auth, session_cookie)
    _set_csrf_cookie(response, csrf, auth=auth)
    _set_flash(response, auth, "ok", "Logged in.")
    logger.info("login_submit: ok (session rotated)")
    return response

@router.post("/dashboard/logout", include_in_schema=False)
async def logout(
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
    _csrf: None = Depends(require_dashboard_csrf),
) -> RedirectResponse:
    response = _redirect("/dashboard/login")
    _clear_session_cookie(response, auth)
    _clear_csrf_cookie(response, auth=auth)
    _set_flash(response, auth, "ok", "Logged out.")
    return response

# --------------------------------------------------------------- dashboard

@router.get("/dashboard", include_in_schema=False)
async def dashboard_home(
    request: Request,
    status: str = Query(
        default=_DEFAULT_STATUS_FILTER,
        description=(
            "Filter the keys table by status. Default: ``active``. "
            "Use ``all`` to show every key regardless of state."
        ),
    ),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    keys: list[dict[str, Any]] = []
    health: dict[str, Any] | None = None
    error: str | None = None
    try:
        async with _client_or_503(request) as client:
            all_keys = await client.list_keys()
    except DashboardClientError as exc:
        error = exc.short
        logger.warning("dashboard_home: list_keys failed: {}", exc)
        all_keys = []

    if status == _STATUS_ALL:
        current_status = _STATUS_ALL
    elif status in _VALID_STATUSES:
        current_status = status
    else:
        current_status = _DEFAULT_STATUS_FILTER

    if current_status == _STATUS_ALL:
        keys = all_keys
    else:
        keys = [k for k in all_keys if k.get("status") == current_status]
    total_keys = len(all_keys)

    flash: dict[str, Any] | None = None
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "request": request,
                "keys": keys,
                "total_keys": total_keys,
                "current_status": current_status,
                "statuses": [_STATUS_ALL] + sorted(_VALID_STATUSES),
                "health": health,
                "health_error": None,
                "error": error,
                "flash": flash,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys/refresh-health", include_in_schema=False)
async def refresh_health(
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    response = _redirect("/dashboard")
    try:
        async with _client_or_503(request) as client:
            health = await client.health()
        msg = (
            f"health: {health.get('status', '?')} "
            f"(active={health.get('active_keys', 0)}, "
            f"depleted={health.get('depleted_keys', 0)}, "
            f"disabled={health.get('disabled_keys', 0)})"
        )
        level: FlashLevel = "ok" if health.get("status") == "ok" else "error"
        _set_flash(response, auth, level, msg)
    except DashboardClientError as exc:
        _set_flash(response, auth, "error", f"health check failed: {exc.short}")
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

# --------------------------------------------------------------- keys CRUD

@router.get("/dashboard/keys/new", include_in_schema=False)
async def new_key_form(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    flash: dict[str, Any] | None = None
    # Mint the CSRF token ONCE per request so probe and response
    # agree. See ``_csrf_form_value`` for the rationale.
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_key_form.html",
            {
                "request": request,
                "key": None,
                "error": None,
                "flash": flash,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys", include_in_schema=False)
async def create_key_submit(
    request: Request,
    label: str = Form(default=""),
    key: str = Form(..., min_length=1, max_length=512),
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    label_clean = label.strip() or None
    try:
        async with _client_or_503(request) as client:
            body = await client.create_key(label_clean, key)
    except DashboardClientError as exc:
        response = _redirect("/dashboard/keys/new")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    raw_key = body.get("raw_key", "")
    if not raw_key:
        response = _redirect("/dashboard")
        _set_flash(
            response,
            auth,
            "error",
            "key created, but proxy did not return raw_key — recreate and check logs",
        )
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    # Show the raw key once on a dedicated page via flash.
    response = _redirect("/dashboard/keys/created")
    _set_flash(response, auth, "ok", f"NEW_KEY::{raw_key}")
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    logger.info(
        "create_key_submit: created key id={} label={}",
        body.get("id"),
        body.get("label"),
    )
    return response

@router.get("/dashboard/keys/created", include_in_schema=False)
async def key_created(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    flash: dict[str, Any] | None = None
    raw_key: str | None = None

    # Mint the CSRF token ONCE per request — see ``_csrf_form_value``.
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_key_created.html",
            {
                "request": request,
                "raw_key": raw_key,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    if flash and flash.get("msg", "").startswith("NEW_KEY::"):
        raw_key = flash["msg"][len("NEW_KEY::") :]
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.get("/dashboard/keys/{key_id}", include_in_schema=False)
async def edit_key_form(
    key_id: int,
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    try:
        async with _client_or_503(request) as client:
            key = await client.get_key(key_id)
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        return response
    flash: dict[str, Any] | None = None
    # Mint the CSRF token ONCE per request — see ``_csrf_form_value``.
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_key_edit.html",
            {
                "request": request,
                "key": key,
                "error": None,
                "flash": flash,
                "csrf_token": csrf_token,
                "valid_statuses": sorted(_VALID_STATUSES),
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys/test-all", include_in_schema=False)
async def test_all_submit(
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            body = await client.test_all_keys()
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    results = body.get("results") or []
    total = body.get("total", len(results))
    ok_count = sum(1 for r in results if r.get("ok"))
    fail_count = total - ok_count
    msg = f"test-all: total={total} ok={ok_count} fail={fail_count}"
    level: FlashLevel = "ok" if fail_count == 0 else "error"
    response = _redirect("/dashboard")
    _set_flash(response, auth, level, msg)
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys/reset-states", include_in_schema=False)
async def reset_states_submit(
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            body = await client.reset_states()
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    msg = (
        f"reset: status={body.get('status', '?')} "
        f"active={body.get('active_keys', 0)} "
        f"depleted={body.get('depleted_keys', 0)} "
        f"disabled={body.get('disabled_keys', 0)}"
    )
    level: FlashLevel = "ok" if body.get("status") == "ok" else "error"
    response = _redirect("/dashboard")
    _set_flash(response, auth, level, msg)
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys/{key_id}", include_in_schema=False)
async def update_key_submit(
    key_id: int,
    request: Request,
    label: str = Form(default=""),
    status: str = Form(default=""),
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    form = await request.form()
    method = form.get("_method", "patch")
    if method != "patch":
        response = _redirect(f"/dashboard/keys/{key_id}")
        _set_flash(response, auth, "error", f"unsupported method: {method}")
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    label_clean = label.strip() or None
    status_clean = status.strip().lower() or None
    if status_clean and status_clean not in _VALID_STATUSES:
        response = _redirect(f"/dashboard/keys/{key_id}")
        _set_flash(
            response,
            auth,
            "error",
            f"invalid status {status_clean!r}; expected one of {sorted(_VALID_STATUSES)}",
        )
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    try:
        async with _client_or_503(request) as client:
            await client.update_key(
                key_id, label=label_clean, status=status_clean
            )
    except DashboardClientError as exc:
        response = _redirect(f"/dashboard/keys/{key_id}")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    response = _redirect("/dashboard")
    _set_flash(response, auth, "ok", f"key {key_id} updated")
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys/{key_id}/delete", include_in_schema=False)
async def delete_key_submit(
    key_id: int,
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            await client.delete_key(key_id)
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    response = _redirect("/dashboard")
    _set_flash(response, auth, "ok", f"key {key_id} deleted")
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/keys/{key_id}/test", include_in_schema=False)
async def test_key_submit(
    key_id: int,
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            body = await client.test_key(key_id)
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    code = body.get("status_code")
    ok = body.get("ok")
    msg = f"key {key_id}: ok={ok} status_code={code}"
    if body.get("error"):
        msg += f" error={body['error']}"
    level: FlashLevel = "ok" if ok else "error"
    response = _redirect("/dashboard")
    _set_flash(response, auth, level, msg)
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

# ----------------------------------------------------------- usage (read)

@router.get("/dashboard/keys/{key_id}/usage", include_in_schema=False)
async def key_usage_view(
    key_id: int,
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    try:
        async with _client_or_503(request) as client:
            snapshot = await client.get_key_usage(key_id)
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        return response

    flash: dict[str, Any] | None = None
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_key_usage.html",
            {
                "request": request,
                "key_id": key_id,
                "snapshot": snapshot,
                "error": None,
                "flash": flash,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

# ----------------------------------------------------------- usage (refresh)

@router.post(
    "/dashboard/keys/{key_id}/usage/refresh",
    include_in_schema=False,
)
async def key_usage_refresh_submit(
    key_id: int,
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            snap = await client.refresh_key_usage(key_id)
    except DashboardClientError as exc:
        response = _redirect(f"/dashboard/keys/{key_id}/usage")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response

    upstream_status = snap.get("upstream_status") or "unknown"
    if upstream_status == "ok":
        level: FlashLevel = "ok"
        msg = f"key {key_id}: usage refreshed"
    else:
        level = "error"
        err = snap.get("upstream_error") or ""
        msg = f"key {key_id}: refresh failed ({upstream_status}) {err}".strip()

    response = _redirect(f"/dashboard/keys/{key_id}/usage")
    _set_flash(response, auth, level, msg)
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post(
    "/dashboard/keys/usage/refresh-all",
    include_in_schema=False,
)
async def usage_refresh_all_submit(
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            body = await client.refresh_all_usage()
    except DashboardClientError as exc:
        response = _redirect("/dashboard")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response

    total = int(body.get("total", 0))
    results = body.get("results") or []
    failures = sum(1 for r in results if (r.get("upstream_status") or "") != "ok")
    if total == 0:
        msg = "no active keys to refresh"
        level: FlashLevel = "ok"
    elif failures == 0:
        msg = f"refreshed {total} active keys"
        level = "ok"
    else:
        msg = f"refreshed {total - failures}/{total} active keys ({failures} failed)"
        level = "error"

    response = _redirect("/dashboard")
    _set_flash(response, auth, level, msg)
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

# ------------------------------------------------------- user tokens (CRUD)

@router.get("/dashboard/user-tokens", include_in_schema=False)
async def user_tokens_home(
    request: Request,
    status: str = Query(
        default=_DEFAULT_USER_TOKEN_STATUS_FILTER,
        description=(
            "Filter the tokens table by status. Default: ``active``. "
            "Use ``all`` to show every token regardless of state."
        ),
    ),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    all_tokens: list[dict[str, Any]] = []
    error: str | None = None
    try:
        async with _client_or_503(request) as client:
            all_tokens = await client.list_user_tokens()
    except DashboardClientError as exc:
        error = exc.short
        logger.warning("user_tokens_home: list_user_tokens failed: {}", exc)

    if status == _STATUS_ALL:
        current_status = _STATUS_ALL
    elif status in _VALID_USER_TOKEN_STATUSES:
        current_status = status
    else:
        current_status = _DEFAULT_USER_TOKEN_STATUS_FILTER

    if current_status == _STATUS_ALL:
        tokens = all_tokens
    else:
        tokens = [t for t in all_tokens if t.get("status") == current_status]
    total_tokens = len(all_tokens)

    flash: dict[str, Any] | None = None
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_user_tokens.html",
            {
                "request": request,
                "tokens": tokens,
                "total_tokens": total_tokens,
                "current_status": current_status,
                "statuses": [_STATUS_ALL] + sorted(_VALID_USER_TOKEN_STATUSES),
                "error": error,
                "flash": flash,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.get("/dashboard/user-tokens/new", include_in_schema=False)
async def new_user_token_form(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    flash: dict[str, Any] | None = None
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_user_token_form.html",
            {
                "request": request,
                "error": None,
                "flash": flash,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/user-tokens", include_in_schema=False)
async def create_user_token_submit(
    request: Request,
    label: str = Form(default=""),
    expires_at: str = Form(default=""),
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    label_clean = label.strip()
    if not label_clean:
        response = _redirect("/dashboard/user-tokens/new")
        _set_flash(response, auth, "error", "label is required")
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    try:
        expires_dt = _parse_user_token_expiry(expires_at)
    except ValueError as exc:
        response = _redirect("/dashboard/user-tokens/new")
        _set_flash(
            response,
            auth,
            "error",
            f"invalid expires_at ({exc}); expected YYYY-MM-DDTHH:MM[:SS]",
        )
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    try:
        async with _client_or_503(request) as client:
            body = await client.create_user_token(
                label=label_clean, expires_at=expires_dt
            )
    except DashboardClientError as exc:
        response = _redirect("/dashboard/user-tokens/new")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    raw_key = body.get("raw_key", "")
    if not raw_key:
        response = _redirect("/dashboard/user-tokens")
        _set_flash(
            response,
            auth,
            "error",
            "token created, but proxy did not return raw_key — recreate and check logs",
        )
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    response = _redirect("/dashboard/user-tokens/created")
    _set_flash(response, auth, "ok", f"NEW_TOKEN::{raw_key}")
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    logger.info(
        "create_user_token_submit: created token id={} label={}",
        body.get("id"),
        body.get("label"),
    )
    return response

@router.get("/dashboard/user-tokens/created", include_in_schema=False)
async def user_token_created(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    flash: dict[str, Any] | None = None
    raw_key: str | None = None

    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_user_token_created.html",
            {
                "request": request,
                "raw_key": raw_key,
                "csrf_token": csrf_token,
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    if flash and flash.get("msg", "").startswith("NEW_TOKEN::"):
        raw_key = flash["msg"][len("NEW_TOKEN::") :]
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.get("/dashboard/user-tokens/{token_id}", include_in_schema=False)
async def edit_user_token_form(
    token_id: int,
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> HTMLResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    templates = _templates(request)
    try:
        async with _client_or_503(request) as client:
            token = await client.get_user_token(token_id)
    except DashboardClientError as exc:
        response = _redirect("/dashboard/user-tokens")
        _set_flash(response, auth, "error", exc.short)
        return response
    flash: dict[str, Any] | None = None
    csrf_token = _csrf_form_value(request, auth, session_id)

    def _render() -> HTMLResponse:
        response = templates.TemplateResponse(
            request,
            "dashboard_user_token_edit.html",
            {
                "request": request,
                "token": token,
                "error": None,
                "flash": flash,
                "csrf_token": csrf_token,
                "valid_statuses": sorted(_VALID_USER_TOKEN_STATUSES),
                "expires_at_value": _format_user_token_expiry(
                    token.get("expires_at")
                ),
            },
        )
        _set_csrf_cookie(response, csrf_token, auth=auth)
        return response

    probe = _render()
    flash = _read_flash(request, probe, auth)
    response = _render()
    _copy_set_cookies(probe, response)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post("/dashboard/user-tokens/{token_id}", include_in_schema=False)
async def update_user_token_submit(
    token_id: int,
    request: Request,
    label: str = Form(default=""),
    expires_at: str = Form(default=""),
    clear_expires_at: str = Form(default=""),
    status: str = Form(default=""),
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    form = await request.form()
    method = form.get("_method", "patch")
    if method != "patch":
        response = _redirect(f"/dashboard/user-tokens/{token_id}")
        _set_flash(response, auth, "error", f"unsupported method: {method}")
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    label_clean = label.strip() or None
    status_clean = status.strip().lower() or None
    if status_clean and status_clean not in _VALID_USER_TOKEN_STATUSES:
        response = _redirect(f"/dashboard/user-tokens/{token_id}")
        _set_flash(
            response,
            auth,
            "error",
            f"invalid status {status_clean!r}; expected one of {sorted(_VALID_USER_TOKEN_STATUSES)}",
        )
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    clear_flag = clear_expires_at.strip().lower() in {"1", "true", "on", "yes"}
    try:
        expires_dt = _parse_user_token_expiry(expires_at)
    except ValueError as exc:
        response = _redirect(f"/dashboard/user-tokens/{token_id}")
        _set_flash(
            response,
            auth,
            "error",
            f"invalid expires_at ({exc}); expected YYYY-MM-DDTHH:MM[:SS]",
        )
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    # ``clear_expires_at`` overrides a present ``expires_at`` value.
    expires_for_update: datetime | None
    if clear_flag:
        expires_for_update = None
    else:
        expires_for_update = expires_dt
    try:
        async with _client_or_503(request) as client:
            await client.update_user_token(
                token_id,
                label=label_clean,
                expires_at=expires_for_update,
                clear_expires_at=clear_flag,
                status=UserTokenStatus(status_clean)
                if status_clean
                else None,
            )
    except DashboardClientError as exc:
        response = _redirect(f"/dashboard/user-tokens/{token_id}")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    response = _redirect("/dashboard/user-tokens")
    _set_flash(response, auth, "ok", f"token {token_id} updated")
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

@router.post(
    "/dashboard/user-tokens/{token_id}/delete",
    include_in_schema=False,
)
async def delete_user_token_submit(
    token_id: int,
    request: Request,
    _csrf: None = Depends(require_dashboard_csrf),
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> RedirectResponse:
    session_id = get_dashboard_session(request, auth=auth)
    if session_id is None:
        return _redirect("/dashboard/login")
    try:
        async with _client_or_503(request) as client:
            await client.hard_delete_user_token(token_id)
    except DashboardClientError as exc:
        response = _redirect("/dashboard/user-tokens")
        _set_flash(response, auth, "error", exc.short)
        _rotate_csrf(response, auth, session_id)
        _set_session_cookie(response, auth, _reissue_session(request, auth))
        return response
    response = _redirect("/dashboard/user-tokens")
    _set_flash(response, auth, "ok", f"token {token_id} deleted")
    _rotate_csrf(response, auth, session_id)
    _set_session_cookie(response, auth, _reissue_session(request, auth))
    return response

# --------------------------------------------------------------- helpers

def _reissue_session(request: Request, auth: DashboardAuth) -> str:
    raw = request.cookies.get(SESSION_COOKIE, "")
    sid = auth.verify_session(raw) or ""
    return auth.reissue_session(sid)

def _csrf_form_value(request: Request, auth: DashboardAuth, session_id: str) -> str:
    return auth.issue_csrf(session_id)

def _rotate_csrf(
    response: RedirectResponse,
    auth: DashboardAuth,
    session_id: str,
) -> None:
    new_token = auth.issue_csrf(session_id)
    _set_csrf_cookie(response, new_token, auth=auth)

def _parse_user_token_expiry(raw: str) -> datetime | None:
    """Parse a form ``expires_at`` value.

    Accepts:
    - empty / blank → ``None`` (caller must combine with ``clear_expires_at``).
    - ``YYYY-MM-DDTHH:MM[:SS]`` (HTML5 ``datetime-local`` format) → naive
      value treated as UTC.

    Raises ``ValueError`` if the value is present but not a valid ISO 8601
    string; the caller is responsible for surfacing the error via flash.
    """
    s = (raw or "").strip()
    if not s:
        return None
    # ``datetime-local`` may emit naive ISO. ``fromisoformat`` accepts that.
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt

def _format_user_token_expiry(dt: datetime | None) -> str:
    """Render a stored expiry as the HTML5 ``datetime-local`` ``value``.

    Returns an empty string if the value is missing, so the input
    renders blank in the form (matches the create form). The naive
    ISO string we emit is what ``_parse_user_token_expiry`` accepts
    on round-trip.
    """
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    # ``datetime-local`` ignores sub-minute precision in most browsers,
    # but emitting seconds doesn't hurt.
    return dt.strftime("%Y-%m-%dT%H:%M")

