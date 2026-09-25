#"""
#Database engine and session setup.

#Uses SQLite as specified in the Data Layer (Chapter 3, 3.5.4).
#create_all() is used instead of Alembic migrations since the schema is
#fixed for the scope of this thesis prototype.


import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.models import Base

# Resolved relative to this file's own folder (app/), not the current
# working directory -- so the database is always found at app/data/
# regardless of where a script that imports this module is run from.
_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(_DB_DIR, exist_ok=True)
DATABASE_URL = f"sqlite:///{os.path.join(_DB_DIR, 'academic_repository.db')}"

# check_same_thread=False is needed because frameworks like FastAPI/Flask
# may handle a single SQLite connection across different threads.
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """
    Dependency-style generator for use with FastAPI's Depends(),
    or call next(get_session()) manually in scripts.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
