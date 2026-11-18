"""Create the tables and a few DEMO orders for local trials (safe to run twice).

    set DATABASE_URL=sqlite:///demo.db        (PowerShell: $env:DATABASE_URL = "sqlite:///demo.db")
    python scripts/seed_demo.py

Customer for every order: demo@example.com
  ORD-1001  delivered 5 days ago   -> refund-eligible (within the demo 30-day window)
  ORD-2002  delivered 40 days ago  -> refund denied (window passed)
  ORD-3003  placed, not shipped    -> can be cancelled
  ORD-4004  shipped 2 days ago     -> tracking works, cannot be cancelled
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import select  # noqa: E402

from supportx.db import (  # noqa: E402
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    Shipment,
    init_db,
    make_engine,
    session_scope,
)

EMAIL = "demo@example.com"


def main() -> None:
    engine = make_engine()
    init_db(engine)
    now = datetime.now(UTC)
    with session_scope(engine) as session:
        if session.scalars(select(Order)).first() is not None:
            print("Orders already exist; nothing to do.")
            return
        customer = Customer(email=EMAIL, name="Demo Customer")

        def item(name: str, cents: int) -> OrderItem:
            return OrderItem(sku=name[:3].upper(), name=name, unit_price_cents=cents, quantity=1)

        def ship(number: str, shipped_days: int, delivered_days: int | None) -> Shipment:
            return Shipment(
                carrier="DemoPost",
                tracking_number=f"TRK-{number}",
                shipped_at=now - timedelta(days=shipped_days),
                delivered_at=None if delivered_days is None else now - timedelta(days=delivered_days),
            )

        session.add_all(
            [
                Order(order_number="ORD-1001", customer=customer, status=OrderStatus.DELIVERED,
                      items=[item("Kettle", 4999)], shipment=ship("ORD-1001", 8, 5)),
                Order(order_number="ORD-2002", customer=customer, status=OrderStatus.DELIVERED,
                      items=[item("Lamp", 3500)], shipment=ship("ORD-2002", 43, 40)),
                Order(order_number="ORD-3003", customer=customer, status=OrderStatus.PLACED,
                      items=[item("Mug", 1200)]),
                Order(order_number="ORD-4004", customer=customer, status=OrderStatus.SHIPPED,
                      items=[item("Backpack", 6500)], shipment=ship("ORD-4004", 2, None)),
            ]
        )
    print(f"Seeded 4 demo orders for {EMAIL}.")


if __name__ == "__main__":
    main()
