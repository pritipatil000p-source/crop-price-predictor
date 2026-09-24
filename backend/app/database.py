"""
database.py
-----------
SQLite database setup using SQLAlchemy.
Stores prediction history so users can review past predictions.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# ---------------------------------------------------------------------------
# Database location — stored inside the backend folder
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
DB_PATH = os.path.join(BASE_DIR, "predictions.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# SQLAlchemy engine + session factory
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite with FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM model: PredictionLog
# ---------------------------------------------------------------------------
class PredictionLog(Base):
    """Stores every prediction request and its result."""
    __tablename__ = "prediction_log"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    state = Column(String)
    district = Column(String)
    market = Column(String)
    commodity = Column(String)
    variety = Column(String)
    grade = Column(String)
    prediction_date = Column(String)
    predicted_price = Column(Float)


def create_tables():
    """Create all database tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency: yields a database session and ensures it is closed
    after the request is complete.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
