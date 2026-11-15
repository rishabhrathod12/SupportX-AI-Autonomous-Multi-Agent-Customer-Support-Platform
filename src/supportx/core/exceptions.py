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


class ModelError(SupportXError):
    """A model is not trained, cannot be loaded, or received unusable input."""


class NotFoundError(SupportXError):
    """A record does not exist, or the caller is not allowed to see it."""


class WorkflowError(SupportXError):
    """An action is not allowed in the current state (for example, deciding a refund twice)."""


class DataError(SupportXError):
    """A dataset is unknown, unavailable, or its license was not acknowledged."""
