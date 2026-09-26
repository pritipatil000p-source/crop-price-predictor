"""
main.py
-------
FastAPI application entry point for the Crop Price Predictor backend.

Start the server with:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
"""

from backend.app.api.routes import router
from backend.app.database import create_tables
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager
import sys
import os

# ---------------------------------------------------------------------------
# Make the project root importable so that `ml/` and `backend/` can be found
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Lifespan: runs once on startup (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    print("[INFO] Database tables created/verified.")
    yield  # application runs here


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Crop Price Prediction API",
    description=(
        "Predicts Indian mandi commodity prices using a RandomForest ML model "
        "trained on historical daily wholesale price data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Streamlit frontend (localhost) to call the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Fine for local development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Root and health routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
def root():
    """API  — confirms the service is running."""
    return {"message": "Crop Price Prediction API"}


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Include all domain routes
# ---------------------------------------------------------------------------
app.include_router(router, tags=["Prediction"])
