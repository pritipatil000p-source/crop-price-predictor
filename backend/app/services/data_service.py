"""
data_service.py
---------------
Provides data access functions: loading the CSV dataset into memory,
and querying historical prices, commodities, states, districts, markets.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# Paths — relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "daily_price.csv")
METADATA_PATH = os.path.join(PROJECT_ROOT, "ml", "metadata.json")


# ---------------------------------------------------------------------------
# Load the cleaned dataset once into memory (cached)
# ---------------------------------------------------------------------------
_cached_df: Optional[pd.DataFrame] = None


def get_dataframe() -> pd.DataFrame:
    """
    Load and cache the cleaned dataset. Returns cached copy on subsequent calls.
    Raises a clear error if the CSV is missing or metadata is missing.
    """
    global _cached_df
    if _cached_df is not None:
        return _cached_df

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Please download the Kaggle dataset and place it at data/daily_price.csv. "
            "See data/README.md for instructions."
        )

    # Use the same preprocessing pipeline as training
    import sys
    ml_dir = os.path.join(PROJECT_ROOT, "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)

    from preprocess import load_clean_data
    _cached_df = load_clean_data(DATA_PATH)
    return _cached_df


def reload_dataframe():
    """Force reload of the cached dataframe (useful after new data uploads)."""
    global _cached_df
    _cached_df = None
    return get_dataframe()


# ---------------------------------------------------------------------------
# Metadata (loaded from ml/metadata.json after training)
# ---------------------------------------------------------------------------
def get_metadata() -> Dict:
    """Load metadata JSON generated during model training."""
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata not found at {METADATA_PATH}. "
            "Please train the model first: python ml/train.py"
        )
    with open(METADATA_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_commodities() -> List[str]:
    """Return sorted list of unique commodity names."""
    meta = get_metadata()
    return sorted(meta.get("commodities", []))


def get_states() -> List[str]:
    """Return sorted list of unique state names."""
    meta = get_metadata()
    return sorted(meta.get("states", []))


def get_districts(state: Optional[str] = None) -> List[str]:
    """Return districts, optionally filtered by state."""
    meta = get_metadata()
    district_state_map = meta.get("district_state_map", {})
    if state:
        return sorted(district_state_map.get(state, []))
    # Return all districts
    all_districts = []
    for dists in district_state_map.values():
        all_districts.extend(dists)
    return sorted(set(all_districts))


def get_markets(state: Optional[str] = None, district: Optional[str] = None) -> List[str]:
    """Return markets, optionally filtered by state and/or district."""
    meta = get_metadata()
    market_map = meta.get("market_map", {})

    if state and district:
        key = f"{state}||{district}"
        return sorted(market_map.get(key, []))

    # Return all markets matching partial filters
    all_markets = []
    for key, markets in market_map.items():
        parts = key.split("||")
        key_state = parts[0] if len(parts) > 0 else ""
        key_district = parts[1] if len(parts) > 1 else ""
        if state and key_state != state:
            continue
        all_markets.extend(markets)

    return sorted(set(all_markets))


def get_varieties() -> List[str]:
    """Return sorted list of unique variety names."""
    meta = get_metadata()
    return sorted(meta.get("varieties", []))


def get_grades() -> List[str]:
    """Return sorted list of unique grade names."""
    meta = get_metadata()
    return sorted(meta.get("grades", []))


def get_historical_prices(
    commodity: Optional[str] = None,
    market: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """
    Query historical prices with optional filters.
    Returns a DataFrame sorted by date ascending.
    """
    df = get_dataframe()

    if commodity:
        df = df[df["commodity"].str.lower() == commodity.lower()]
    if market:
        df = df[df["market"].str.lower() == market.lower()]
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    # Return the most recent `limit` records
    df = df.sort_values("date").tail(limit)
    return df
