"""
evaluate.py
-----------
Loads the saved model and re-evaluates it on the test portion of the dataset.
Useful for re-checking performance without retraining.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocess import load_clean_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(MODEL_DIR, "saved_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TRAIN_RATIO = 0.80


def evaluate():
    """
    Load the saved model and evaluate on the held-out test split.
    Prints metrics and saves a simple prediction vs actual plot.
    """
    print("\n" + "="*60)
    print("  CROP PRICE PREDICTOR — Model Evaluation")
    print("="*60)

    # --- Load model ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Model not found at {MODEL_PATH}\n"
            "Please run: python ml/train.py"
        )
    pipeline = joblib.load(MODEL_PATH)
    print(f"\n✅ Model loaded from: {MODEL_PATH}")

    # --- Load metrics metadata ---
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            saved_metrics = json.load(f)
        CATEGORICAL_FEATURES = saved_metrics.get("features_categorical", [])
        NUMERICAL_FEATURES = saved_metrics.get("features_numerical", [])
    else:
        # Fallback defaults
        CATEGORICAL_FEATURES = ["commodity", "state", "district", "market", "variety", "grade"]
        NUMERICAL_FEATURES = ["min_price", "max_price", "year", "month", "day", "day_of_year"]

    all_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    TARGET = "modal_price"

    # --- Load and split data ---
    df = load_clean_data()
    split_idx = int(len(df) * TRAIN_RATIO)
    test_df = df.iloc[split_idx:]

    X_test = test_df[all_features]
    y_test = test_df[TARGET]

    print(f"\nTest set: {len(test_df):,} rows")
    print(f"Date range: {test_df['date'].min().date()} → {test_df['date'].max().date()}")

    # --- Predict ---
    y_pred = pipeline.predict(X_test)

    # --- Metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"  Evaluation Results")
    print(f"{'='*40}")
    print(f"  MAE  : {mae:,.2f}  INR/quintal")
    print(f"  RMSE : {rmse:,.2f}  INR/quintal")
    print(f"  R²   : {r2:.4f}")
    print(f"{'='*40}")

    # --- Update saved metrics ---
    updated_metrics = {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
        "train_size": split_idx,
        "test_size": len(test_df),
        "features_categorical": CATEGORICAL_FEATURES,
        "features_numerical": NUMERICAL_FEATURES,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(updated_metrics, f, indent=2)
    print(f"\n✅ Updated metrics saved to: {METRICS_PATH}")

    # --- Plot: Actual vs Predicted (sample of 500 points) ---
    sample_size = min(500, len(y_test))
    idx = np.random.choice(len(y_test), sample_size, replace=False)
    y_test_sample = np.array(y_test)[idx]
    y_pred_sample = y_pred[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test_sample, y_pred_sample, alpha=0.4, s=15, color="#3b82d4")
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Modal Price (INR/quintal)")
    ax.set_ylabel("Predicted Modal Price (INR/quintal)")
    ax.set_title(f"Actual vs Predicted — R² = {r2:.4f}")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(MODEL_DIR, "evaluation_plot.png")
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    print(f"✅ Evaluation plot saved to: {plot_path}")

    return updated_metrics


if __name__ == "__main__":
    evaluate()
