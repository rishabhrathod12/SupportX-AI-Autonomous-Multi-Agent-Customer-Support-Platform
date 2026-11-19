"""MOCK adapters for local demos and tests. They do NOT move money, change stock or contact anyone.

Every result has `is_mock=True` and a reference starting with "mock-". Each mock remembers the
idempotency keys it has seen, so repeating a call returns the first result and changes nothing
(the same promise a real provider must keep). `fail_next` / `fail_permanently` inject failures so
retry and dead-letter paths can be exercised.
"""

from supportx.integrations.ports import (
    PermanentProviderError,
    ProviderResult,
    StockLine,
    TransientProviderError,
)


class _MockBase:
    kind = "mock"

    def __init__(self) -> None:
        self._seen: dict[str, ProviderResult] = {}
        self.calls = 0  # real executions (not idempotent replays)
        self._transient_failures = 0
        self._permanent_failure = False

    def fail_next(self, times: int = 1) -> None:
        """The next `times` calls raise TransientProviderError."""
        self._transient_failures = times

    def fail_permanently(self) -> None:
        self._permanent_failure = True

    def _execute(self, key: str, prefix: str) -> ProviderResult:
        if key in self._seen:
            return self._seen[key]  # idempotent replay: no new effect
        if self._permanent_failure:
            raise PermanentProviderError("mock provider rejected the request")
        if self._transient_failures > 0:
            self._transient_failures -= 1
            raise TransientProviderError("mock provider is temporarily unavailable")
        self.calls += 1
        result = ProviderResult(reference=f"mock-{prefix}-{self.calls}", is_mock=True)
        self._seen[key] = result
        return result


class MockPaymentProvider(_MockBase):
    def __init__(self) -> None:
        super().__init__()
        self.refunded: list[
            tuple[str, int]
        ] = []  # (order number, cents): what a real one would pay

    def refund(self, idempotency_key: str, order_number: str, amount_cents: int) -> ProviderResult:
        before = self.calls
        result = self._execute(idempotency_key, "pay")
        if self.calls > before:
            self.refunded.append((order_number, amount_cents))
        return result


class MockInventoryService(_MockBase):
    def __init__(self) -> None:
        super().__init__()
        self.released: list[tuple[str, list[StockLine]]] = []

    def release(
        self, idempotency_key: str, order_number: str, lines: list[StockLine]
    ) -> ProviderResult:
        before = self.calls
        result = self._execute(idempotency_key, "stock")
        if self.calls > before:
            self.released.append((order_number, lines))
        return result


class MockNotifier(_MockBase):
    def __init__(self) -> None:
        super().__init__()
        self.sent: list[tuple[int, str, dict[str, str]]] = []

    def send(
        self, idempotency_key: str, customer_id: int, template: str, params: dict[str, str]
    ) -> ProviderResult:
        before = self.calls
        result = self._execute(idempotency_key, "msg")
        if self.calls > before:
            self.sent.append((customer_id, template, params))
        return result
