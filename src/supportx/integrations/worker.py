"""Deliver outbox events. Run it as its own process, next to (not inside) the API:

    python -m supportx.integrations.worker --once                 # deliver what is due, then exit
    python -m supportx.integrations.worker --loop --interval 10   # keep going

INTEGRATIONS_MODE decides which adapters it uses:
    disabled  (default)  refuses to run: events stay PENDING in the outbox until a provider exists
    mock                 the MOCK adapters: nothing real happens (money, stock and messages are
                         NOT touched); every result is labeled as a mock

There is no real provider implementation yet; add one by implementing the protocols in
`supportx.integrations.ports` and returning it from `build_providers`.
"""

import argparse
import sys
import time
from collections.abc import Sequence

from supportx.core import ConfigurationError, Settings, get_logger, get_settings, setup_logging
from supportx.db import make_engine
from supportx.integrations.mocks import MockInventoryService, MockNotifier, MockPaymentProvider
from supportx.integrations.outbox import OutboxDispatcher, Providers

logger = get_logger(__name__)


def build_providers(settings: Settings) -> Providers:
    if settings.integrations_mode == "mock":
        return Providers(MockPaymentProvider(), MockInventoryService(), MockNotifier())
    raise ConfigurationError(
        "INTEGRATIONS_MODE is 'disabled': no payment, inventory or notification provider is "
        "configured, so outbox events stay pending. Set INTEGRATIONS_MODE=mock for a demo with "
        "mock adapters (they do not move money or contact anyone)."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deliver outbox events.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0, help="seconds between passes")
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, settings.log_format)
    try:
        providers = build_providers(settings)
    except ConfigurationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if settings.integrations_mode == "mock":
        print("MOCK ADAPTERS: nothing real is paid, released or sent.", file=sys.stderr)

    engine = make_engine()
    dispatcher = OutboxDispatcher(engine, providers, max_attempts=settings.outbox_max_attempts)
    try:
        while True:
            summary = dispatcher.run_once()
            print(
                f"sent={summary.sent} retried={summary.retried} dead={summary.dead} "
                f"skipped={summary.skipped}"
            )
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
