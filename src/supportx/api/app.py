"""Application factory.

Run:  uvicorn supportx.api.app:create_app --factory --reload
Docs: http://127.0.0.1:8000/docs   (turned off when APP_ENV=production)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from supportx.api.routers import customer, health, staff
from supportx.core import (
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
    Settings,
    SupportXError,
    WorkflowError,
    get_logger,
    get_settings,
    setup_logging,
)
from supportx.db import init_db, make_engine
from supportx.services import RefundPolicy

logger = get_logger(__name__)


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    policy: RefundPolicy | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own settings and engine; production passes nothing."""
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        own_engine = engine is None
        db = engine or make_engine(settings.require_database_url())
        if own_engine and settings.app_env == "development":
            init_db(db)  # convenience for local work; production uses migrations
        app.state.session_factory = sessionmaker(db, expire_on_commit=False)
        yield
        if own_engine:
            db.dispose()

    app = FastAPI(
        title="SupportX AI",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.app_env == "production" else "/docs",
        redoc_url=None,
        openapi_url=None if settings.app_env == "production" else "/openapi.json",
    )
    app.state.settings = settings
    app.state.policy = policy or RefundPolicy()
    if engine is not None:  # tests may not run the lifespan, so wire the session factory now
        app.state.session_factory = sessionmaker(engine, expire_on_commit=False)

    app.include_router(health.router)
    app.include_router(customer.router)
    app.include_router(staff.router)

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return _error(404, str(exc))

    @app.exception_handler(WorkflowError)
    async def conflict(_: Request, exc: WorkflowError) -> JSONResponse:
        return _error(409, str(exc))

    @app.exception_handler(ConfigurationError)
    async def misconfigured(_: Request, exc: ConfigurationError) -> JSONResponse:
        logger.error("Configuration problem: %s", exc)
        return _error(503, "The service is not configured correctly.")

    @app.exception_handler(ExternalServiceError)
    async def upstream(_: Request, exc: ExternalServiceError) -> JSONResponse:
        logger.error("External service failed: %s", exc)
        return _error(502, "An upstream service failed.")

    @app.exception_handler(SupportXError)
    async def unexpected(_: Request, exc: SupportXError) -> JSONResponse:
        logger.exception("Unhandled application error")
        return _error(500, "Internal server error.")

    return app
