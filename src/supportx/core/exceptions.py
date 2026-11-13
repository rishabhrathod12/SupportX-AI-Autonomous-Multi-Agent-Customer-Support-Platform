"""Project-wide exception hierarchy.

Every error we raise on purpose inherits from SupportXError, so callers (for example the
API layer) can catch one base class and still tell specific problems apart.
"""


class SupportXError(Exception):
    """Base class for all SupportX AI errors."""


class ConfigurationError(SupportXError):
    """A required setting is missing or invalid."""


class ExternalServiceError(SupportXError):
    """A call to an external service (OpenAI, database, ...) failed."""
