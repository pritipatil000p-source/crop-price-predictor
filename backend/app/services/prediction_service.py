"""
prediction_service.py
---------------------
Loads the trained ML pipeline and provides the predict() function.
Also loads model evaluation metrics from ml/metrics.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "saved_model.pkl")
METRICS_PATH = os.path.join(PROJECT_ROOT, "ml", "metrics.json")

# ---------------------------------------------------------------------------
# Lazy-load the model once
# ---------------------------------------------------------------------------
_pipeline = None


def get_pipeline():
    """Load and cache the trained sklearn Pipeline."""
    global _pipeline
    if _pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Please train the model first: python ml/train.py"
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def get_metrics() -> Dict[str, Any]:
    """Load evaluation metrics from ml/metrics.json."""
    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(
            f"Metrics file not found at {METRICS_PATH}. "
            "Please train the model first: python ml/train.py"
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


def predict(
    state: str,
    district: str,
    market: str,
    commodity: str,
    variety: str,
    grade: str,
    prediction_date,  # datetime.date or str
) -> float:
    """
    Build a feature row for the given inputs and return the predicted modal price.

    For future dates (where min_price and max_price are not known), we estimate
    them using the recent average of that commodity+market combination from the
    historical dataset. If no historical data exists for the combination, we fall
    back to the global average for the commodity.

    Parameters
    ----------
    prediction_date : datetime.date | str
        The target date for prediction.

    Returns
    -------
    float
        Predicted modal price in INR/quintal.
    """
    pipeline = get_pipeline()

    # Load metrics to know which features were used during training
    metrics = get_metrics()
    categorical_features = metrics.get(
        "features_categorical",
        ["commodity", "state", "district", "market", "variety", "grade"],
    )
    numerical_features = metrics.get(
        "features_numerical",
        ["min_price", "max_price", "year", "month", "day", "day_of_year"],
    )

    # Parse prediction date
    pred_date = pd.to_datetime(prediction_date)
    year = pred_date.year
    month = pred_date.month
    day = pred_date.day
    day_of_year = pred_date.day_of_year

    # Estimate min/max price from historical data
    min_price_est, max_price_est = _estimate_prices(commodity, market, state)

    # Build the feature row as a DataFrame (same schema as training)
    row_data = {
        "commodity": commodity,
        "state": state,
        "district": district,
        "market": market,
        "variety": variety,
        "grade": grade,
        "min_price": min_price_est,
        "max_price": max_price_est,
        "year": year,
        "month": month,
        "day": day,
        "day_of_year": day_of_year,
    }

    # Keep only columns the pipeline expects
    all_features = categorical_features + numerical_features
    row = {k: v for k, v in row_data.items() if k in all_features}
    X = pd.DataFrame([row])

    predicted_price = float(pipeline.predict(X)[0])

    # Ensure non-negative prediction
    predicted_price = max(0.0, predicted_price)
    return round(predicted_price, 2)


def _estimate_prices(commodity: str, market: str, state: str):
    """
    Estimate min/max prices for a commodity+market combination by looking up
    recent historical data. Falls back to commodity-wide averages if no data found.
    """
    try:
        from backend.app.services.data_service import get_dataframe
        df = get_dataframe()

        # Try commodity + market first
        mask = (
            (df["commodity"].str.lower() == commodity.lower()) &
            (df["market"].str.lower() == market.lower())
        )
        subset = df[mask].tail(30)

        if len(subset) == 0:
            # Fall back to commodity + state
            mask = (
                (df["commodity"].str.lower() == commodity.lower()) &
                (df["state"].str.lower() == state.lower())
            )
            subset = df[mask].tail(30)

        if len(subset) == 0:
            # Fall back to just commodity
            subset = df[df["commodity"].str.lower() == commodity.lower()].tail(30)

        if len(subset) > 0:
            min_price = float(subset["min_price"].mean())
            max_price = float(subset["max_price"].mean())
            return min_price, max_price

    except Exception:
        pass

    # Hard fallback: use 0 (the pipeline handles unknown combos via OHE)
    return 0.0, 0.0
