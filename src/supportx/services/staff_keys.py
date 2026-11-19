"""Staff API keys stored in the database: create, rotate and revoke WITHOUT restarting the server.

Only a SHA-256 hash of each key is stored (keys are long random strings, so a fast hash is enough;
a leaked database does not reveal usable keys). The key itself is shown once, when it is created.
Rotation = create a new key for the person, hand it over, revoke the old one.

Run from the command line (the database comes from DATABASE_URL):

    python -m supportx.services.staff_keys create jane [--days 90]
    python -m supportx.services.staff_keys list
    python -m supportx.services.staff_keys revoke jane          # every active key of that name
    python -m supportx.services.staff_keys revoke --id 3        # one key
"""

import argparse
import hashlib
import re
import secrets
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from supportx.core import DataError
from supportx.db.models import StaffKey
from supportx.db.repositories import to_utc
from supportx.services.audit import record_audit

_NAME = re.compile(r"^[A-Za-z0-9._-]{1,60}$")
KEY_PREFIX = "sx_"


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _is_active(key: StaffKey, now: datetime) -> bool:
    expires = to_utc(key.expires_at)
    return key.revoked_at is None and (expires is None or expires > now)


class StaffKeyService:
    def __init__(self, session: Session, actor: str = "cli"):
        self._session = session
        self._actor = actor

    def create(
        self, name: str, expires_in: timedelta | None = None, now: datetime | None = None
    ) -> str:
        """Make a new key for `name` and return it. This is the ONLY time it can be read."""
        if not _NAME.match(name):
            raise DataError("A staff name may use only letters, digits, '.', '_' and '-'.")
        now = now or datetime.now(UTC)
        key = KEY_PREFIX + secrets.token_urlsafe(32)
        row = StaffKey(
            name=name,
            key_hash=hash_key(key),
            created_at=now,
            expires_at=now + expires_in if expires_in else None,
        )
        self._session.add(row)
        self._session.flush()
        record_audit(
            self._session,
            self._actor,
            "staff_key.created",
            "staff_key",
            row.id,
            {"name": name, "expires_at": row.expires_at.isoformat() if row.expires_at else None},
            now,
        )
        return key

    def revoke(
        self, name: str | None = None, key_id: int | None = None, now: datetime | None = None
    ) -> int:
        """Revoke every active key of `name`, or one key by id. Returns how many were revoked."""
        if (name is None) == (key_id is None):
            raise DataError("Give either a name or a key id.")
        now = now or datetime.now(UTC)
        stmt = select(StaffKey).where(StaffKey.revoked_at.is_(None))
        stmt = stmt.where(StaffKey.name == name) if name else stmt.where(StaffKey.id == key_id)
        rows = list(self._session.scalars(stmt))
        for row in rows:
            row.revoked_at = now
            record_audit(
                self._session,
                self._actor,
                "staff_key.revoked",
                "staff_key",
                row.id,
                {"name": row.name},
                now,
            )
        self._session.flush()
        return len(rows)

    def authenticate(self, presented: str, now: datetime | None = None) -> str | None:
        """The staff name for a valid, unexpired, unrevoked key; otherwise None."""
        if not presented:
            return None
        row = self._session.scalars(
            select(StaffKey).where(StaffKey.key_hash == hash_key(presented))
        ).one_or_none()
        if row is None or not _is_active(row, now or datetime.now(UTC)):
            return None
        return row.name

    def has_active_keys(self, now: datetime | None = None) -> bool:
        moment = now or datetime.now(UTC)
        return any(_is_active(k, moment) for k in self._session.scalars(select(StaffKey)))

    def all_keys(self) -> list[StaffKey]:
        return list(self._session.scalars(select(StaffKey).order_by(StaffKey.id)))


def main(argv: Sequence[str] | None = None) -> int:
    from supportx.db import make_engine, session_scope

    parser = argparse.ArgumentParser(description="Manage staff API keys in the database.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create a key (shown once)")
    create.add_argument("name")
    create.add_argument("--days", type=int, default=None, help="expire after this many days")
    sub.add_parser("list", help="list keys (never shows the keys themselves)")
    revoke = sub.add_parser("revoke", help="revoke by name or id")
    revoke.add_argument("name", nargs="?")
    revoke.add_argument("--id", type=int, dest="key_id")
    args = parser.parse_args(argv)

    engine = make_engine()
    try:
        with session_scope(engine) as session:
            service = StaffKeyService(session)
            if args.command == "create":
                days = timedelta(days=args.days) if args.days else None
                key = service.create(args.name, days)
                print(f"Key for {args.name} (shown once, store it in a secret manager):\n{key}")
            elif args.command == "revoke":
                count = service.revoke(name=args.name, key_id=args.key_id)
                print(f"Revoked {count} key(s).")
            else:
                now = datetime.now(UTC)
                for k in service.all_keys():
                    state = "active" if _is_active(k, now) else "inactive"
                    print(f"{k.id}\t{k.name}\t{state}\tcreated {k.created_at:%Y-%m-%d}")
    except DataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    finally:
        engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
