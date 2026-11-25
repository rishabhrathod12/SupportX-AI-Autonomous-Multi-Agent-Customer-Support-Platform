"""Defensive HTTP hardening: security headers, request-size limit, trusted hosts.

Limits (read before relying on it):
  * The size limit checks the Content-Length header. A client that streams a chunked body
    without it is not stopped here; the reverse proxy / server must also cap bodies.
  * HSTS is only sent when HSTS_ENABLED=true, because it is only meaningful (and only safe to
    switch on) when the site is really served over HTTPS. This code does NOT terminate TLS.
"""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from supportx.core import Settings, get_logger

logger = get_logger("supportx.security")

HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",  # API answers hold order and ticket data
}
HSTS = "max-age=31536000; includeSubDomains"


def add_security(app: FastAPI, settings: Settings) -> None:
    limit = settings.max_request_bytes

    @app.middleware("http")
    async def harden(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            if not declared.isdigit():
                response: Response = JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length."}
                )
            elif limit > 0 and int(declared) > limit:
                response = JSONResponse(status_code=413, content={"detail": "Request too large."})
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)
        for name, value in HEADERS.items():
            response.headers.setdefault(name, value)
        if settings.hsts_enabled:
            response.headers.setdefault("Strict-Transport-Security", HSTS)
        return response

    hosts = settings.allowed_host_list()
    if hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)
