"""Small shared utility helpers used across backend modules."""

from datetime import datetime, timezone


def utc_iso_now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""

    return datetime.now(timezone.utc).isoformat()
