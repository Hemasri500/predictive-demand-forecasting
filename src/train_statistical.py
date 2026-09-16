from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_train.csv"
)


def load_data():
    """
    Load the feature-engineered dataset.
    """

    dataframe = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    print(f"Loaded {len(dataframe):,} rows.")

    return dataframe


def aggregate_daily_sales(dataframe):
    """
    Aggregate sales across all stores and product families
    to create one daily company-wide sales time series.
    """

    daily_sales = (
        dataframe.groupby("date", as_index=False)["sales"]
        .sum()
        .sort_values("date")
    )

    return daily_sales


def create_time_split(daily_sales):
    """
    Create chronological training and validation periods.
    """

    validation_start = pd.Timestamp("2017-07-01")

    train = daily_sales[
        daily_sales["date"] < validation_start
    ].copy()

    validation = daily_sales[
        daily_sales["date"] >= validation_start
    ].copy()

    return train, validation


def train_exponential_smoothing(train):
    """
    Train Holt-Winters Exponential Smoothing model.

    Weekly seasonality is represented with seasonal_periods=7.
    """

    model = ExponentialSmoothing(
        train["sales"],
        trend="add",
        seasonal="add",
        seasonal_periods=7,
    )

    fitted_model = model.fit(
        optimized=True
    )

    return fitted_model


def evaluate_model(validation, predictions):
    """
    Calculate MAE and RMSE.
    """

    mae = mean_absolute_error(
        validation["sales"],
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            validation["sales"],
            predictions,
        )
    )

    return mae, rmse


def evaluate_daily_weekly_baseline(daily_sales, validation):
    """
    Evaluate a 7-day naive baseline at the company-daily level.
    """

    baseline_data = daily_sales.copy()

    baseline_data["weekly_baseline"] = (
        baseline_data["sales"].shift(7)
    )

    baseline_validation = baseline_data[
        baseline_data["date"] >= pd.Timestamp("2017-07-01")
    ].dropna(
        subset=["weekly_baseline"]
    )

    mae = mean_absolute_error(
        baseline_validation["sales"],
        baseline_validation["weekly_baseline"],
    )

    rmse = np.sqrt(
        mean_squared_error(
            baseline_validation["sales"],
            baseline_validation["weekly_baseline"],
        )
    )

    return mae, rmse


def main():

    dataframe = load_data()

    daily_sales = aggregate_daily_sales(
        dataframe
    )

    train, validation = create_time_split(
        daily_sales
    )

    print("\n" + "=" * 60)
    print("STATISTICAL FORECASTING MODEL")
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

    print("\nTraining Exponential Smoothing model...")

    fitted_model = train_exponential_smoothing(
        train
    )

    forecast_horizon = len(validation)

    predictions = fitted_model.forecast(
        forecast_horizon
    )

    mae, rmse = evaluate_model(
        validation,
        predictions,
    )
    
    baseline_mae, baseline_rmse = evaluate_daily_weekly_baseline(
        daily_sales,
        validation,
    )

    print("\nModel: Holt-Winters Exponential Smoothing")
    print("Seasonality: Weekly (7 days)")

    print(f"\nValidation observations: {len(validation):,}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print("\n" + "=" * 60)
    print("COMPANY-LEVEL MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Model':<30}"
        f"{'MAE':>15}"
        f"{'RMSE':>15}"
    )

    print("-" * 60)

    print(
        f"{'Weekly Naive Baseline':<30}"
        f"{baseline_mae:>15,.2f}"
        f"{baseline_rmse:>15,.2f}"
    )

    print(
        f"{'Holt-Winters':<30}"
        f"{mae:>15,.2f}"
        f"{rmse:>15,.2f}"
    )

    mae_improvement = (
        (baseline_mae - mae)
        / baseline_mae
        * 100
    )

    print(
        f"\nHolt-Winters MAE improvement vs baseline: "
        f"{mae_improvement:.2f}%"
    )


if __name__ == "__main__":
    main()