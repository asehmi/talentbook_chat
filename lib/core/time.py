"""UTC time helpers.

`datetime.utcnow()` is deprecated (removal planned) and was always a trap: it
returns a *naive* datetime holding UTC, so the timezone has to be remembered
rather than carried. Comparing one against an aware datetime raises TypeError.

The app's rule is simple: **every datetime in memory is timezone-aware UTC.**
Use `utcnow()` to make one and `to_utc()` to normalise anything arriving from
outside.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """The current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc(value: datetime | None) -> datetime | None:
    """Normalise any datetime to timezone-aware UTC.

    A naive datetime is *assumed* to already be UTC rather than local time.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
