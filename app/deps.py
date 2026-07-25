"""Dipendenze FastAPI condivise."""

from __future__ import annotations

from fastapi import HTTPException, Request

from .context import AppContext
from .security import token_from_request


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_admin_ctx(request: Request) -> AppContext:
    """Come :func:`get_ctx` ma solo con un token di regia valido."""
    ctx: AppContext = request.app.state.ctx
    if not ctx.auth.check(token_from_request(request)):
        raise HTTPException(401, "token regia mancante o non valido")
    return ctx
