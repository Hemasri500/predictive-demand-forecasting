from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_train.csv"
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "store_nbr",
    "family",
    "onpromotion",
    "is_holiday",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "quarter",
    "is_weekend",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "has_promotion",
]


# ============================================================
# BACKTEST WINDOWS
# ============================================================

BACKTEST_WINDOWS = [
    {
        "name": "April 2017",
        "validation_start": "2017-04-01",
        "validation_end": "2017-04-30",
    },
    {
        "name": "May 2017",
        "validation_start": "2017-05-01",
        "validation_end": "2017-05-31",
    },
    {
        "name": "June 2017",
        "validation_start": "2017-06-01",
        "validation_end": "2017-06-30",
    },
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load feature-engineered historical dataset.
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
# BUILD MODEL
# ============================================================

def build_model():
    """
    Build the same XGBoost configuration selected
    during model development.
    """

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )

    return model


# ============================================================
# RUN ONE BACKTEST FOLD
# ============================================================

def run_fold(
    dataframe,
    fold,
):
    """
    Train and evaluate one historical backtest window.
    """

    fold_name = fold["name"]

    validation_start = pd.Timestamp(
        fold["validation_start"]
    )

    validation_end = pd.Timestamp(
        fold["validation_end"]
    )

    print("\n" + "=" * 60)
    print(f"BACKTEST: {fold_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train = dataframe[
        dataframe["date"] < validation_start
    ].copy()

    validation = dataframe[
        (
            dataframe["date"]
            >= validation_start
        )
        &
        (
            dataframe["date"]
            <= validation_end
        )
    ].copy()

    # --------------------------------------------------------
    # Remove rows without required historical features
    # --------------------------------------------------------

    train = train.dropna(
        subset=FEATURE_COLUMNS + ["sales"]
    ).copy()

    validation = validation.dropna(
        subset=FEATURE_COLUMNS + ["sales"]
    ).copy()

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
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    train["family"] = (
        train["family"]
        .astype("category")
    )

    validation["family"] = pd.Categorical(
        validation["family"],
        categories=train["family"].cat.categories,
    )

    train["store_nbr"] = (
        train["store_nbr"]
        .astype("category")
    )

    validation["store_nbr"] = pd.Categorical(
        validation["store_nbr"],
        categories=train["store_nbr"].cat.categories,
    )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = train[
        FEATURE_COLUMNS
    ]

    y_train = train[
        "sales"
    ]

    X_validation = validation[
        FEATURE_COLUMNS
    ]

    y_validation = validation[
        "sales"
    ]

    # --------------------------------------------------------
    # Weekly baseline
    # --------------------------------------------------------

    baseline_predictions = validation[
        "sales_lag_7"
    ]

    baseline_mae, baseline_rmse = (
        calculate_metrics(
            y_validation,
            baseline_predictions,
        )
    )

    print("\nWeekly baseline:")
    print(
        f"MAE:  {baseline_mae:,.2f}"
    )
    print(
        f"RMSE: {baseline_rmse:,.2f}"
    )

    # --------------------------------------------------------
    # Train fresh XGBoost model
    # --------------------------------------------------------

    print(
        "\nTraining XGBoost..."
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    raw_predictions = model.predict(
        X_validation
    )

    predictions = np.maximum(
        raw_predictions,
        0,
    )

    # --------------------------------------------------------
    # XGBoost metrics
    # --------------------------------------------------------

    xgb_mae, xgb_rmse = (
        calculate_metrics(
            y_validation,
            predictions,
        )
    )

    print("\nXGBoost:")
    print(
        f"MAE:  {xgb_mae:,.2f}"
    )
    print(
        f"RMSE: {xgb_rmse:,.2f}"
    )

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

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

    print(
        f"\nMAE improvement: "
        f"{mae_improvement:.2f}%"
    )

    print(
        f"RMSE improvement: "
        f"{rmse_improvement:.2f}%"
    )

    return {
        "Period": fold_name,
        "Baseline_MAE": baseline_mae,
        "XGBoost_MAE": xgb_mae,
        "MAE_Improvement_%": mae_improvement,
        "Baseline_RMSE": baseline_rmse,
        "XGBoost_RMSE": xgb_rmse,
        "RMSE_Improvement_%": rmse_improvement,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = load_data()

    results = []

    for fold in BACKTEST_WINDOWS:

        fold_result = run_fold(
            dataframe,
            fold,
        )

        results.append(
            fold_result
        )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    print(
        "\n"
        + results_df.to_string(
            index=False,
            formatters={
                "Baseline_MAE":
                    lambda x: f"{x:,.2f}",

                "XGBoost_MAE":
                    lambda x: f"{x:,.2f}",

                "MAE_Improvement_%":
                    lambda x: f"{x:.2f}%",

                "Baseline_RMSE":
                    lambda x: f"{x:,.2f}",

                "XGBoost_RMSE":
                    lambda x: f"{x:,.2f}",

                "RMSE_Improvement_%":
                    lambda x: f"{x:.2f}%",
            },
        )
    )

    # --------------------------------------------------------
    # Average performance
    # --------------------------------------------------------

    average_baseline_mae = (
        results_df[
            "Baseline_MAE"
        ].mean()
    )

    average_xgb_mae = (
        results_df[
            "XGBoost_MAE"
        ].mean()
    )

    average_baseline_rmse = (
        results_df[
            "Baseline_RMSE"
        ].mean()
    )

    average_xgb_rmse = (
        results_df[
            "XGBoost_RMSE"
        ].mean()
    )

    # Calculate improvement from aggregate average errors.
    average_mae_improvement = (
        (
            average_baseline_mae
            - average_xgb_mae
        )
        / average_baseline_mae
        * 100
    )

    average_rmse_improvement = (
        (
            average_baseline_rmse
            - average_xgb_rmse
        )
        / average_baseline_rmse
        * 100
    )

    print("\n" + "-" * 60)
    print("AVERAGE BACKTEST PERFORMANCE")
    print("-" * 60)

    print(
        f"\nAverage Baseline MAE: "
        f"{average_baseline_mae:,.2f}"
    )

    print(
        f"Average XGBoost MAE: "
        f"{average_xgb_mae:,.2f}"
    )

    print(
        f"Average MAE improvement: "
        f"{average_mae_improvement:.2f}%"
    )

    print(
        f"\nAverage Baseline RMSE: "
        f"{average_baseline_rmse:,.2f}"
    )

    print(
        f"Average XGBoost RMSE: "
        f"{average_xgb_rmse:,.2f}"
    )

    print(
        f"Average RMSE improvement: "
        f"{average_rmse_improvement:.2f}%"
    )


if __name__ == "__main__":
    main()