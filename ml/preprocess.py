"""
preprocess.py
-------------
Loads the raw Kaggle CSV from data/daily_price.csv, inspects it,
normalizes column names, cleans the data, and returns a clean DataFrame
ready for model training.
"""

import os
import pandas as pd
import numpy as np

# Path to the raw dataset (relative to project root)
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "daily_price.csv")


# ---------------------------------------------------------------------------
# Column name mapping: map various possible Kaggle column names → normalized
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    # Date column variations
    "arrival_date": "date",
    "date": "date",
    "arrivaldate": "date",
    "price_date": "date",

    # Location
    "state_name": "state",
    "state": "state",

    "district_name": "district",
    "district": "district",

    "market_name": "market",
    "market": "market",

    # Commodity
    "commodity_name": "commodity",
    "commodity": "commodity",

    "variety": "variety",
    "variety_name": "variety",

    "grade": "grade",

    # Prices
    "min_price": "min_price",
    "minimum_price": "min_price",
    "minprice": "min_price",

    "max_price": "max_price",
    "maximum_price": "max_price",
    "maxprice": "max_price",

    "modal_price": "modal_price",
    "modalprice": "modal_price",
    "modal price": "modal_price",
}


def load_and_inspect(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV and print a quick inspection report.
    Returns the raw DataFrame before any cleaning.
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset from: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {path}\n"
            "Please download the Kaggle dataset and place it at data/daily_price.csv\n"
            "See data/README.md for instructions."
        )

    df = pd.read_csv(path, low_memory=False)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn names detected:\n  {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to lowercase snake_case and apply the COLUMN_MAP.
    """
    # Lowercase + strip whitespace + replace spaces with underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    # Apply the mapping
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

    # Verify essential columns exist
    required = {"date", "state", "district", "market", "commodity", "modal_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[ERROR] After normalization, required columns are still missing: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            "Please check the COLUMN_MAP in ml/preprocess.py and update it to match your CSV."
        )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the normalized DataFrame:
    - Parse dates
    - Drop rows with missing critical values
    - Remove invalid/negative prices
    - Remove duplicates
    - Add time-based feature columns
    """
    print(f"\n{'='*60}")
    print("Cleaning data...")
    print(f"{'='*60}")

    original_size = len(df)

    # --- Parse date ---
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    bad_dates = df["date"].isnull().sum()
    if bad_dates > 0:
        print(f"  Dropping {bad_dates:,} rows with unparseable dates.")
    df = df.dropna(subset=["date"])

    # --- Fill optional columns ---
    for col in ["variety", "grade"]:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # --- Fill min/max price from modal price if missing ---
    if "min_price" not in df.columns:
        df["min_price"] = df["modal_price"]
    if "max_price" not in df.columns:
        df["max_price"] = df["modal_price"]

    df["min_price"] = pd.to_numeric(df["min_price"], errors="coerce")
    df["max_price"] = pd.to_numeric(df["max_price"], errors="coerce")
    df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")

    # --- Drop rows with missing prices ---
    df = df.dropna(subset=["modal_price", "min_price", "max_price"])

    # --- Drop rows with zero or negative prices ---
    df = df[(df["modal_price"] > 0) & (df["min_price"] >= 0) & (df["max_price"] >= 0)]

    # --- Drop duplicates ---
    df = df.drop_duplicates()

    # --- Strip string columns ---
    for col in ["state", "district", "market", "commodity", "variety", "grade"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # --- Add time-based features ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear

    # --- Sort by date for chronological splitting ---
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  Rows before cleaning : {original_size:,}")
    print(f"  Rows after  cleaning : {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Unique commodities   : {df['commodity'].nunique()}")
    print(f"  Unique states        : {df['state'].nunique()}")
    print(f"  Unique markets       : {df['market'].nunique()}")

    return df


def load_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Full pipeline: load → normalize columns → clean → return clean DataFrame.
    """
    df_raw = load_and_inspect(path)
    df_norm = normalize_columns(df_raw)
    df_clean = clean_data(df_norm)
    return df_clean


if __name__ == "__main__":
    df = load_clean_data()
    print("\nSample cleaned data:")
    print(df.head())
    print(f"\nFinal shape: {df.shape}")
