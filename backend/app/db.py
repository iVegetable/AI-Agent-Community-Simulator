"""Database bootstrap for SQLModel/SQLite storage.

This module exposes the shared SQLAlchemy engine and table initialization
utility used by FastAPI lifespan and tests.
"""

from pathlib import Path

from sqlmodel import SQLModel, create_engine

from .config import settings


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    """Create local SQLite parent directory when file-based URL is used."""

    if not database_url.startswith("sqlite:///"):
        return
    raw_path = database_url[len("sqlite:///") :].split("?", 1)[0].strip()
    if not raw_path or raw_path == ":memory:" or raw_path.startswith("file:"):
        return
    db_path = Path(raw_path).expanduser()
    parent = db_path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent_dir(settings.database_url)
engine = create_engine(settings.database_url, echo=False)


def init_db() -> None:
    """Create all registered SQLModel tables."""

    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)
