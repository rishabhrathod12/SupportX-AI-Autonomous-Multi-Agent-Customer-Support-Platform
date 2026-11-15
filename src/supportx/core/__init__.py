"""Core utilities: settings, logging, exceptions. Imports nothing else from supportx."""

from supportx.core.config import Settings, get_settings
from supportx.core.exceptions import (
    ConfigurationError,
    DataError,
    ExternalServiceError,
    ModelError,
    SupportXError,
)
from supportx.core.logging_config import get_logger, setup_logging

__all__ = [
    "ConfigurationError",
    "DataError",
    "ExternalServiceError",
    "ModelError",
    "Settings",
    "SupportXError",
    "get_logger",
    "get_settings",
    "setup_logging",
]
