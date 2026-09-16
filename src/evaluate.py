from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_train.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_forecaster.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load the feature-engineered dataset.
    """

    dataframe = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    print(
        f"Loaded {len(dataframe):,} rows."
    )

    return dataframe


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the saved XGBoost model and feature list.
    """

    model_package = joblib.load(
        MODEL_PATH
    )

    model = model_package["model"]
    feature_columns = model_package["features"]

    print(
        "XGBoost model loaded successfully."
    )

    return model, feature_columns


# ============================================================
# TIME-BASED SPLIT
# ============================================================

def create_time_split(dataframe):
    """
    Split the dataset chronologically.

    Training:
        2013-01-01 through 2017-06-30

    Validation:
        2017-07-01 through 2017-08-15
    """

    validation_start = pd.Timestamp(
        "2017-07-01"
    )

    train = dataframe[
        dataframe["date"] < validation_start
    ].copy()

    validation = dataframe[
        dataframe["date"] >= validation_start
    ].copy()

    print("\n" + "=" * 60)
    print("TIME-BASED DATA SPLIT")
    print("=" * 60)

    print(
        f"\nTraining period: "
        f"{train['date'].min().date()} "
        f"to {train['date'].max().date()}"
    )

    print(
        f"Validation period: "
        f"{validation['date'].min().date()} "
        f"to {validation['date'].max().date()}"
    )

    print(
        f"\nTraining rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    return train, validation


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual,
    predicted,
):
    """
    Calculate MAE and RMSE.
    """

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    return mae, rmse


# ============================================================
# WEEKLY BASELINE
# ============================================================

def evaluate_weekly_baseline(
    validation,
):
    """
    Evaluate weekly naive baseline.

    Prediction =
    sales from 7 previous recorded observations
    for the same store-family series.
    """

    baseline = validation.dropna(
        subset=[
            "sales",
            "sales_lag_7",
        ]
    ).copy()

    actual = baseline["sales"]

    predicted = baseline[
        "sales_lag_7"
    ]

    mae, rmse = calculate_metrics(
        actual,
        predicted,
    )

    print("\n" + "=" * 60)
    print("WEEKLY NAIVE BASELINE")
    print("=" * 60)

    print(
        "\nStrategy:"
    )

    print(
        "Prediction = sales from "
        "7 previous recorded observations"
    )

    print(
        f"\nObservations evaluated: "
        f"{len(baseline):,}"
    )

    print(
        f"MAE:  {mae:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    return mae, rmse


# ============================================================
# XGBOOST EVALUATION
# ============================================================

def evaluate_xgboost(
    model,
    feature_columns,
    validation,
):
    """
    Evaluate the trained XGBoost model
    on the validation period.
    """

    evaluation_data = validation.dropna(
        subset=feature_columns + ["sales"]
    ).copy()

    # --------------------------------------------------------
    # Match categorical types used during training
    # --------------------------------------------------------

    evaluation_data["family"] = (
        evaluation_data["family"]
        .astype("category")
    )

    evaluation_data["store_nbr"] = (
        evaluation_data["store_nbr"]
        .astype("category")
    )

    # --------------------------------------------------------
    # Prepare features and target
    # --------------------------------------------------------

    X_validation = evaluation_data[
        feature_columns
    ]

    y_validation = evaluation_data[
        "sales"
    ]

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    raw_predictions = model.predict(
        X_validation
    )

    # Demand cannot be negative.
    predictions = np.maximum(
        raw_predictions,
        0,
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    mae, rmse = calculate_metrics(
        y_validation,
        predictions,
    )

    negative_raw_predictions = (
        raw_predictions < 0
    ).sum()

    print("\n" + "=" * 60)
    print("XGBOOST FORECAST MODEL")
    print("=" * 60)

    print(
        f"\nObservations evaluated: "
        f"{len(evaluation_data):,}"
    )

    print(
        f"Raw negative predictions: "
        f"{negative_raw_predictions:,}"
    )

    print(
        "Negative predictions after clipping: "
        f"{(predictions < 0).sum():,}"
    )

    print(
        f"\nMAE:  {mae:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    return mae, rmse


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    baseline_mae,
    baseline_rmse,
    xgb_mae,
    xgb_rmse,
):
    """
    Compare XGBoost performance
    against the weekly naive baseline.
    """

    mae_improvement = (
        (
            baseline_mae - xgb_mae
        )
        / baseline_mae
        * 100
    )

    rmse_improvement = (
        (
            baseline_rmse - xgb_rmse
        )
        / baseline_rmse
        * 100
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    comparison = pd.DataFrame(
        {
            "Model": [
                "Weekly Naive Baseline",
                "XGBoost",
            ],
            "MAE": [
                baseline_mae,
                xgb_mae,
            ],
            "RMSE": [
                baseline_rmse,
                xgb_rmse,
            ],
        }
    )

    print(
        "\n"
        + comparison.to_string(
            index=False,
            formatters={
                "MAE": lambda x: f"{x:,.2f}",
                "RMSE": lambda x: f"{x:,.2f}",
            },
        )
    )

    print(
        f"\nXGBoost MAE improvement "
        f"vs baseline: "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"XGBoost RMSE improvement "
        f"vs baseline: "
        f"{rmse_improvement:.2f}%"
    )

    return (
        mae_improvement,
        rmse_improvement,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Load data
    dataframe = load_data()

    # Load trained model
    model, feature_columns = (
        load_model()
    )

    # Create chronological split
    train, validation = (
        create_time_split(
            dataframe
        )
    )

    # Evaluate weekly baseline
    baseline_mae, baseline_rmse = (
        evaluate_weekly_baseline(
            validation
        )
    )

    # Evaluate XGBoost
    xgb_mae, xgb_rmse = (
        evaluate_xgboost(
            model,
            feature_columns,
            validation,
        )
    )

    # Compare models
    compare_models(
        baseline_mae,
        baseline_rmse,
        xgb_mae,
        xgb_rmse,
    )

    print("\n" + "=" * 60)
    print("STATISTICAL MODEL NOTE")
    print("=" * 60)

    print(
        "\nHolt-Winters is evaluated separately "
        "at company-wide daily aggregation level."
    )

    print(
        "Its metrics should not be directly compared "
        "with the store-family XGBoost metrics."
    )


if __name__ == "__main__":
    main()