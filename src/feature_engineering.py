from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DATA_DIR / "processed_train.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "featured_train.csv"


def load_processed_data():
    """
    Load the processed training dataset.
    """

    dataframe = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"],
        dtype={
            "holiday_type": "string",
            "holiday_locale": "string",
            "holiday_name": "string",
        },
        low_memory=False,
    )

    # Replace missing holiday labels after loading
    dataframe["holiday_type"] = dataframe["holiday_type"].fillna("None")
    dataframe["holiday_locale"] = dataframe["holiday_locale"].fillna("None")
    dataframe["holiday_name"] = dataframe["holiday_name"].fillna("None")

    print(f"Loaded {len(dataframe):,} rows.")

    return dataframe


def create_calendar_features(dataframe):
    """
    Create calendar-based features from the date column.
    """

    dataframe = dataframe.copy()

    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["day"] = dataframe["date"].dt.day
    dataframe["day_of_week"] = dataframe["date"].dt.dayofweek
    dataframe["week_of_year"] = dataframe["date"].dt.isocalendar().week.astype(int)
    dataframe["quarter"] = dataframe["date"].dt.quarter

    dataframe["is_weekend"] = (
        dataframe["day_of_week"] >= 5
    ).astype(int)

    return dataframe


def create_lag_features(dataframe):
    """
    Create historical sales features for each store-product family.
    """

    dataframe = dataframe.copy()

    # Sort chronologically before creating lag features
    dataframe = dataframe.sort_values(
        ["store_nbr", "family", "date"]
    )

    grouped_sales = dataframe.groupby(
        ["store_nbr", "family"]
    )["sales"]

    for lag in [1, 7, 14, 28]:
        dataframe[f"sales_lag_{lag}"] = grouped_sales.shift(lag)

    return dataframe


def create_rolling_features(dataframe):
    """
    Create rolling historical sales statistics.

    shift(1) prevents today's sales from being used
    to predict today's sales.
    """

    dataframe = dataframe.copy()

    grouped_sales = dataframe.groupby(
        ["store_nbr", "family"]
    )["sales"]

    for window in [7, 14, 28]:

        dataframe[f"rolling_mean_{window}"] = (
            grouped_sales
            .transform(
                lambda series:
                series.shift(1).rolling(window).mean()
            )
        )

    dataframe["rolling_std_7"] = (
        grouped_sales
        .transform(
            lambda series:
            series.shift(1).rolling(7).std()
        )
    )

    return dataframe


def create_promotion_features(dataframe):
    """
    Create simplified promotion features.
    """

    dataframe = dataframe.copy()

    dataframe["has_promotion"] = (
        dataframe["onpromotion"] > 0
    ).astype(int)

    return dataframe


def validate_features(dataframe):
    """
    Display information about generated features.
    """

    feature_columns = [
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

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    print(f"\nRows: {len(dataframe):,}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\nCreated Features:")

    for feature in feature_columns:
        print(f"- {feature}")

    print("\nMissing Values in Engineered Features:")
    print(dataframe[feature_columns].isnull().sum())


def save_featured_data(dataframe):
    """
    Save feature-engineered dataset.
    """

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nFeature-engineered dataset saved to:")
    print(OUTPUT_FILE)


def main():

    print("Loading processed data...")
    dataframe = load_processed_data()

    print("Creating calendar features...")
    dataframe = create_calendar_features(dataframe)

    print("Creating lag features...")
    dataframe = create_lag_features(dataframe)

    print("Creating rolling features...")
    dataframe = create_rolling_features(dataframe)

    print("Creating promotion features...")
    dataframe = create_promotion_features(dataframe)

    validate_features(dataframe)

    save_featured_data(dataframe)


if __name__ == "__main__":
    main()