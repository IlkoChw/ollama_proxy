from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.services.dashboard_auth import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    DashboardAuth,
    get_dashboard_auth,
)


def get_dashboard_auth_dep() -> DashboardAuth:
    return get_dashboard_auth()

def get_dashboard_session(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> str | None:
    raw = request.cookies.get(SESSION_COOKIE, "")
    return auth.verify_session(raw)

async def require_dashboard_csrf(
    request: Request,
    auth: DashboardAuth = Depends(get_dashboard_auth_dep),
) -> None:
    import hmac

    session_id = get_dashboard_session(request, auth=auth)
    if not session_id:
        # No session at all — the session dependency should have
        # redirected already; this branch is a defensive guard.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="no session",
        )
    form = await _read_form(request)
    form_token = form.get("csrf_token")
    if not form_token:
        form_token = request.headers.get("X-CSRF-Token")
    if not isinstance(form_token, str):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="missing csrf_token",
        )
    cookie_token = request.cookies.get(CSRF_COOKIE, "")
    if not cookie_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="missing csrf cookie",
        )
    # 1) form value must equal cookie value (double-submit).
    if not hmac.compare_digest(form_token.encode("utf-8"), cookie_token.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="csrf token mismatch",
        )
    # 2) the value must be a valid signed blob for this session.
    if not auth.verify_csrf(session_id, form_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid csrf token",
        )

async def _read_form(request: Request) -> dict[str, object]:
    form = await request.form()
    out: dict[str, object] = {}
    for k in form.keys():
        v = form.get(k)
        if isinstance(v, str):
            out[k] = v
    return out
