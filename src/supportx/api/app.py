"""Application factory.

Run:  uvicorn supportx.api.app:create_app --factory --reload
Docs: http://127.0.0.1:8000/docs   (turned off when APP_ENV=production)
"""

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from supportx.agents.tickets import TicketService, build_ticket_service
from supportx.api.middleware import add_request_logging
from supportx.api.routers import customer, health, staff
from supportx.api.routers import tickets as tickets_router
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
    tickets: TicketService | None = None,
) -> FastAPI:
    """Build the app. Tests pass their own settings, engine and ticket service; production
    passes nothing and everything is built from the settings at start-up."""
    settings = settings or get_settings()
    setup_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        own_engine = engine is None
        db = engine or make_engine(settings.require_database_url())
        if own_engine and settings.app_env == "development":
            init_db(db)  # convenience for local work; production uses migrations
        app.state.session_factory = sessionmaker(db, expire_on_commit=False)

        checkpoint_connection: sqlite3.Connection | None = None
        if tickets is None:
            try:
                app.state.tickets, checkpoint_connection = build_ticket_service(
                    settings, db, app.state.policy
                )
            except ConfigurationError as exc:  # the rest of the API still works
                logger.warning("Ticket workflow disabled: %s", exc)
        yield
        if checkpoint_connection is not None:
            checkpoint_connection.close()
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
    app.state.tickets = tickets  # None until the lifespan builds it (or stays None if disabled)
    if engine is not None:  # tests may not run the lifespan, so wire the session factory now
        app.state.session_factory = sessionmaker(engine, expire_on_commit=False)

    add_request_logging(app)
    app.include_router(health.router)
    app.include_router(customer.router)
    app.include_router(tickets_router.router)
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
