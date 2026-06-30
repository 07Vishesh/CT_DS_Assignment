"""
SQLAlchemy engine/session setup for the local SQLite database.

SQLite is intentionally used here instead of Postgres/MySQL: this is a
single-user local study tool, and SQLite keeps setup to zero external
services while still demonstrating proper ORM modeling, relationships,
and migrations-ready structure.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI's threadpool
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and guarantees closure."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called once on application startup."""
    from app import models  # noqa: F401  (ensures models are registered on Base)
    Base.metadata.create_all(bind=engine)
