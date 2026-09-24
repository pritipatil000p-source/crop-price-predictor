"""
prediction.py
-------------
SQLAlchemy ORM model for storing prediction logs.
This is a simple wrapper that re-exports PredictionLog from database.py.
"""

# The actual model is defined in database.py to avoid circular imports.
# This file exists to keep the models/ directory consistent with project structure.

from backend.app.database import PredictionLog  # noqa: F401

__all__ = ["PredictionLog"]
