"""
train.py
--------
Trains a RandomForestRegressor pipeline on the cleaned agricultural price data.
Uses chronological (time-based) train/test splitting — NOT random shuffling.
Saves the trained pipeline to ml/saved_model.pkl using joblib.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import our preprocessing module
from preprocess import load_clean_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = ["commodity", "state", "district", "market", "variety", "grade"]
NUMERICAL_FEATURES = ["min_price", "max_price", "year", "month", "day", "day_of_year"]
TARGET = "modal_price"

# Train/test split ratio (chronological)
TRAIN_RATIO = 0.80


def chronological_split(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO):
    """
    Split the dataframe chronologically (oldest records → train, newest → test).
    This preserves temporal order and avoids data leakage.
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\nChronological split:")
    print(f"  Train: {len(train_df):,} rows  ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
    print(f"  Test : {len(test_df):,} rows  ({test_df['date'].min().date()} → {test_df['date'].max().date()})")
    return train_df, test_df


def train():
    """
    Main training function:
    1. Load and clean data
    2. Split chronologically
    3. Build and train pipeline
    4. Evaluate on test set
    5. Save model and metrics
    """
    print("\n" + "="*60)
    print("  CROP PRICE PREDICTOR — Model Training")
    print("="*60)

    # --- Load data ---
    df = load_clean_data()

    # Keep only required columns
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [c for c in NUMERICAL_FEATURES if c in df.columns]
    all_features = available_cat + available_num

    missing_features = set(CATEGORICAL_FEATURES + NUMERICAL_FEATURES) - set(df.columns)
    if missing_features:
        print(f"\n[WARNING] These features are not in the dataset and will be skipped: {missing_features}")

    # Update feature lists to only available columns
    global CATEGORICAL_FEATURES, NUMERICAL_FEATURES
    CATEGORICAL_FEATURES = available_cat
    NUMERICAL_FEATURES = available_num

    print(f"\nFeatures used:")
    print(f"  Categorical: {CATEGORICAL_FEATURES}")
    print(f"  Numerical  : {NUMERICAL_FEATURES}")
    print(f"  Target     : {TARGET}")

    # --- Chronological split ---
    train_df, test_df = chronological_split(df, TRAIN_RATIO)

    X_train = train_df[all_features]
    y_train = train_df[TARGET]
    X_test = test_df[all_features]
    y_test = test_df[TARGET]

    # --- Build pipeline with the actual available feature lists ---
    print("\nTraining RandomForestRegressor pipeline...")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # --- Evaluate ---
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "features_categorical": CATEGORICAL_FEATURES,
        "features_numerical": NUMERICAL_FEATURES,
    }

    print(f"\n{'='*40}")
    print(f"  Model Evaluation Metrics")
    print(f"{'='*40}")
    print(f"  MAE  : {mae:,.2f} INR/quintal")
    print(f"  RMSE : {rmse:,.2f} INR/quintal")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*40}")

    # --- Save model ---
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

    # --- Save metrics ---
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to: {METRICS_PATH}")

    # --- Also save metadata for the backend (unique values for dropdowns) ---
    metadata = {
        "commodities": sorted(df["commodity"].unique().tolist()),
        "states": sorted(df["state"].unique().tolist()),
        "districts": sorted(df["district"].unique().tolist()),
        "markets": sorted(df["market"].unique().tolist()),
        "varieties": sorted(df["variety"].unique().tolist()),
        "grades": sorted(df["grade"].unique().tolist()),
        # district → state mapping
        "district_state_map": (
            df[["district", "state"]]
            .drop_duplicates()
            .groupby("state")["district"]
            .apply(list)
            .to_dict()
        ),
        # market → district+state mapping
        "market_map": (
            df[["market", "district", "state"]]
            .drop_duplicates()
            .groupby(["state", "district"])["market"]
            .apply(list)
            .to_dict()
        ),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
    }

    metadata_path = os.path.join(MODEL_DIR, "metadata.json")

    # Convert tuple keys to strings for JSON serialization
    metadata["market_map"] = {
        f"{k[0]}||{k[1]}": v for k, v in metadata["market_map"].items()
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_path}")

    return pipeline, metrics


if __name__ == "__main__":
    train()
