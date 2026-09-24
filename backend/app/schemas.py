"""
schemas.py
----------
Pydantic models (request/response schemas) for the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input data required to make a price prediction."""
    state: str = Field(..., example="Maharashtra")
    district: str = Field(..., example="Pune")
    market: str = Field(..., example="Pune")
    commodity: str = Field(..., example="Tomato")
    variety: str = Field(default="Local", example="Local")
    grade: str = Field(default="FAQ", example="FAQ")
    prediction_date: date = Field(..., example="2026-10-15")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    """Prediction result returned by POST /predict."""
    commodity: str
    market: str
    prediction_date: str
    predicted_price: float
    unit: str = "INR per quintal"
    disclaimer: str = (
        "This is an ML model estimate based on historical data. "
        "It is NOT a guaranteed market price."
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str


class ModelMetricsResponse(BaseModel):
    """Model evaluation metrics returned by GET /model-metrics."""
    MAE: float
    RMSE: float
    R2: float
    train_size: Optional[int] = None
    test_size: Optional[int] = None


class HistoricalPriceRecord(BaseModel):
    """A single historical price record."""
    date: str
    commodity: str
    market: str
    state: str
    min_price: float
    max_price: float
    modal_price: float


class HistoricalPricesResponse(BaseModel):
    """Response for GET /historical-prices."""
    records: List[HistoricalPriceRecord]
    total: int
