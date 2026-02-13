"""Database bootstrap for SQLModel/SQLite storage.

This module exposes the shared SQLAlchemy engine and table initialization
utility used by FastAPI lifespan and tests.
"""

from sqlmodel import SQLModel, create_engine

from .config import settings

engine = create_engine(settings.database_url, echo=False)


def init_db() -> None:
    """Create all registered SQLModel tables."""

    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)
