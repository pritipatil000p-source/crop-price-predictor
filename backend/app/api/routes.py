"""
routes.py
---------
All FastAPI route handlers for the Crop Price Predictor API.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from backend.app.schemas import (
    PredictionRequest,
    PredictionResponse,
    ModelMetricsResponse,
    HistoricalPricesResponse,
    HistoricalPriceRecord,
)
from backend.app.database import get_db, PredictionLog
from backend.app.services import data_service, prediction_service

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /commodities
# ---------------------------------------------------------------------------
@router.get("/commodities", response_model=List[str], summary="List all commodities")
def list_commodities():
    """Return all unique commodity names available in the dataset."""
    try:
        return data_service.get_commodities()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# GET /states
# ---------------------------------------------------------------------------
@router.get("/states", response_model=List[str], summary="List all states")
def list_states():
    """Return all unique state names available in the dataset."""
    try:
        return data_service.get_states()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# GET /districts
# ---------------------------------------------------------------------------
@router.get("/districts", response_model=List[str], summary="List districts for a state")
def list_districts(state: Optional[str] = Query(None, description="Filter by state name")):
    """Return districts, optionally filtered by state."""
    try:
        return data_service.get_districts(state=state)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# GET /markets
# ---------------------------------------------------------------------------
@router.get("/markets", response_model=List[str], summary="List markets")
def list_markets(
    state: Optional[str] = Query(None, description="Filter by state"),
    district: Optional[str] = Query(None, description="Filter by district"),
):
    """Return markets, optionally filtered by state and/or district."""
    try:
        return data_service.get_markets(state=state, district=district)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# GET /varieties
# ---------------------------------------------------------------------------
@router.get("/varieties", response_model=List[str], summary="List varieties")
def list_varieties():
    """Return all unique variety names."""
    try:
        return data_service.get_varieties()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# GET /grades
# ---------------------------------------------------------------------------
@router.get("/grades", response_model=List[str], summary="List grades")
def list_grades():
    """Return all unique grade names."""
    try:
        return data_service.get_grades()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# GET /historical-prices
# ---------------------------------------------------------------------------
@router.get(
    "/historical-prices",
    response_model=HistoricalPricesResponse,
    summary="Get historical prices",
)
def get_historical_prices(
    commodity: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(200, ge=1, le=2000),
):
    """
    Return historical modal prices filtered by commodity, market, and date range.
    """
    try:
        df = data_service.get_historical_prices(
            commodity=commodity,
            market=market,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if df.empty:
        return HistoricalPricesResponse(records=[], total=0)

    records = []
    for _, row in df.iterrows():
        records.append(
            HistoricalPriceRecord(
                date=str(row["date"].date()),
                commodity=row["commodity"],
                market=row["market"],
                state=row["state"],
                min_price=float(row["min_price"]),
                max_price=float(row["max_price"]),
                modal_price=float(row["modal_price"]),
            )
        )

    return HistoricalPricesResponse(records=records, total=len(records))


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@router.post("/predict", response_model=PredictionResponse, summary="Predict crop price")
def predict_price(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Predict the modal price for the given commodity, market, and future date.
    Logs the prediction to the SQLite database.
    """
    try:
        predicted_price = prediction_service.predict(
            state=request.state,
            district=request.district,
            market=request.market,
            commodity=request.commodity,
            variety=request.variety,
            grade=request.grade,
            prediction_date=request.prediction_date,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Log the prediction to the database
    log = PredictionLog(
        state=request.state,
        district=request.district,
        market=request.market,
        commodity=request.commodity,
        variety=request.variety,
        grade=request.grade,
        prediction_date=str(request.prediction_date),
        predicted_price=predicted_price,
    )
    db.add(log)
    db.commit()

    return PredictionResponse(
        commodity=request.commodity,
        market=request.market,
        prediction_date=str(request.prediction_date),
        predicted_price=predicted_price,
    )


# ---------------------------------------------------------------------------
# GET /model-metrics
# ---------------------------------------------------------------------------
@router.get(
    "/model-metrics",
    response_model=ModelMetricsResponse,
    summary="Get model evaluation metrics",
)
def get_model_metrics():
    """Return MAE, RMSE, and R² for the trained model."""
    try:
        metrics = prediction_service.get_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ModelMetricsResponse(
        MAE=metrics["MAE"],
        RMSE=metrics["RMSE"],
        R2=metrics["R2"],
        train_size=metrics.get("train_size"),
        test_size=metrics.get("test_size"),
    )


# ---------------------------------------------------------------------------
# GET /prediction-history
# ---------------------------------------------------------------------------
@router.get("/prediction-history", summary="Get past predictions")
def get_prediction_history(limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    """Return the most recent predictions stored in the database."""
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "created_at": str(log.created_at),
            "commodity": log.commodity,
            "market": log.market,
            "state": log.state,
            "prediction_date": log.prediction_date,
            "predicted_price": log.predicted_price,
        }
        for log in logs
    ]
