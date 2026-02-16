"""Database bootstrap safety checks."""

from pathlib import Path

from app.db import _ensure_sqlite_parent_dir


def test_ensure_sqlite_parent_dir_creates_nested_parent(tmp_path):
    db_file = tmp_path / "nested" / "db" / "test.db"
    assert not db_file.parent.exists()

    _ensure_sqlite_parent_dir(f"sqlite:///{db_file.as_posix()}")

    assert db_file.parent.exists()
