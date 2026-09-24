"""
api_client.py
-------------
Helper functions for the Streamlit frontend to call the FastAPI backend.
All HTTP calls are isolated here so app.py stays clean.
"""

import requests
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# Backend base URL — change this if you run the backend on a different port
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"

# Timeout in seconds for all requests
TIMEOUT = 15


def _get(endpoint: str, params: Optional[Dict] = None) -> Any:
    """
    Internal helper for GET requests. Returns parsed JSON or raises
    a RuntimeError with a user-friendly message.
    """
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to the backend API.\n"
            "Please make sure the FastAPI server is running:\n"
            "  uvicorn backend.app.main:app --reload"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("The backend request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"Backend error {e.response.status_code}: {detail or str(e)}")


def _post(endpoint: str, data: Dict) -> Any:
    """Internal helper for POST requests."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.post(url, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to the backend API.\n"
            "Please make sure the FastAPI server is running:\n"
            "  uvicorn backend.app.main:app --reload"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("The backend request timed out.")
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        raise RuntimeError(f"Backend error {e.response.status_code}: {detail or str(e)}")


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def check_health() -> bool:
    """Returns True if backend is healthy, False otherwise."""
    try:
        result = _get("/health")
        return result.get("status") == "healthy"
    except Exception:
        return False


def get_states() -> List[str]:
    return _get("/states")


def get_districts(state: Optional[str] = None) -> List[str]:
    params = {}
    if state:
        params["state"] = state
    return _get("/districts", params=params)


def get_markets(state: Optional[str] = None, district: Optional[str] = None) -> List[str]:
    params = {}
    if state:
        params["state"] = state
    if district:
        params["district"] = district
    return _get("/markets", params=params)


def get_commodities() -> List[str]:
    return _get("/commodities")


def get_varieties() -> List[str]:
    return _get("/varieties")


def get_grades() -> List[str]:
    return _get("/grades")


def get_historical_prices(
    commodity: Optional[str] = None,
    market: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> Dict:
    """Returns dict with 'records' list and 'total' count."""
    params = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    if market:
        params["market"] = market
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return _get("/historical-prices", params=params)


def predict_price(
    state: str,
    district: str,
    market: str,
    commodity: str,
    variety: str,
    grade: str,
    prediction_date: str,
) -> Dict:
    """
    Call POST /predict and return the prediction response dict.
    Keys: commodity, market, prediction_date, predicted_price, unit, disclaimer
    """
    payload = {
        "state": state,
        "district": district,
        "market": market,
        "commodity": commodity,
        "variety": variety,
        "grade": grade,
        "prediction_date": prediction_date,
    }
    return _post("/predict", payload)


def get_model_metrics() -> Dict:
    """Returns dict with MAE, RMSE, R2."""
    return _get("/model-metrics")
