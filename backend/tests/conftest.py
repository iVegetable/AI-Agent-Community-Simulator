"""Shared pytest fixture: reset database and force mock AI per test."""

import pytest
from sqlmodel import SQLModel

from app import models  # noqa: F401
from app.config import settings
from app.db import engine


@pytest.fixture(autouse=True)
def reset_database():
    original_mode = settings.ai_mode
    settings.ai_mode = "mock"
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
    yield
    settings.ai_mode = original_mode
