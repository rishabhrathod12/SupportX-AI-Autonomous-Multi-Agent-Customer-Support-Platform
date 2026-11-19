"""Interfaces to the outside world: payments, inventory, customer notifications.

NO REAL PROVIDER IS IMPLEMENTED. Only labeled mock adapters exist (`supportx.integrations.mocks`).
A real provider must implement these protocols and honour the idempotency key: calling again with
the same key must not repeat the effect (no double refund, no double stock release, no duplicate
message), because the outbox delivers at least once.

Errors tell the dispatcher what to do:
  TransientProviderError  worth retrying later (timeout, 5xx, rate limit)
  PermanentProviderError  retrying cannot help (rejected, invalid data); a person must look
"""

from dataclasses import dataclass
from typing import Protocol

from supportx.core import SupportXError


class TransientProviderError(SupportXError):
    """The provider could not do it right now; try again later."""


class PermanentProviderError(SupportXError):
    """The provider refused; do not retry."""


@dataclass(frozen=True)
class ProviderResult:
    reference: str  # the provider's id for what it did (a mock's id starts with "mock-")
    is_mock: bool = False


@dataclass(frozen=True)
class StockLine:
    sku: str
    quantity: int


class PaymentProvider(Protocol):
    def refund(self, idempotency_key: str, order_number: str, amount_cents: int) -> ProviderResult:
        """Return `amount_cents` of the payment for `order_number` to the customer."""
        ...


class InventoryService(Protocol):
    def release(
        self, idempotency_key: str, order_number: str, lines: list[StockLine]
    ) -> ProviderResult:
        """Put reserved stock for a cancelled order back on sale."""
        ...


class Notifier(Protocol):
    def send(
        self, idempotency_key: str, customer_id: int, template: str, params: dict[str, str]
    ) -> ProviderResult:
        """Tell the customer something. The notifier looks up the contact details itself, so
        outbox rows never carry an email address."""
        ...
