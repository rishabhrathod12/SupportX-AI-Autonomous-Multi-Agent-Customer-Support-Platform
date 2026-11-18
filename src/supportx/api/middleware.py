"""Request-id and access-log middleware.

The access log line contains method, path (NO query string), status and duration only: never
request bodies, headers or query strings, which can hold emails and the staff key.
"""

import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response

from supportx.core import get_logger
from supportx.core.logging_config import request_id_var

logger = get_logger("supportx.access")

HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")  # caller-supplied ids are only reused if harmless


def add_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        supplied = request.headers.get(HEADER, "")
        request_id = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[HEADER] = request_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "%s %s -> %d (%.1f ms)", request.method, request.url.path, status, elapsed_ms
            )
            request_id_var.reset(token)
