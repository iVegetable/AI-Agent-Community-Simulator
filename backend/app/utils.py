from datetime import datetime, timezone


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
