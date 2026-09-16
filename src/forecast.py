from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_demand_forecaster.joblib"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "test.csv"
)

STORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stores.csv"
)

HOLIDAYS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "holidays_events.csv"
)

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
)

FORECAST_PATH = (
    FORECAST_DIR
    / "xgboost_forecast.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load trained XGBoost model and feature list.
    """

    model_package = joblib.load(
        MODEL_PATH
    )

    model = model_package["model"]
    feature_columns = model_package["features"]

    print("XGBoost model loaded successfully.")

    return model, feature_columns


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load historical, future, store, and holiday data.
    """

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    stores = pd.read_csv(
        STORES_PATH,
        low_memory=False,
    )

    holidays = pd.read_csv(
        HOLIDAYS_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    print(f"Historical rows: {len(train):,}")
    print(f"Future rows: {len(test):,}")

    print(
        f"Historical period: "
        f"{train['date'].min().date()} to "
        f"{train['date'].max().date()}"
    )

    print(
        f"Forecast period: "
        f"{test['date'].min().date()} to "
        f"{test['date'].max().date()}"
    )

    return train, test, stores, holidays


# ============================================================
# CALENDAR FEATURES
# ============================================================

def add_calendar_features(dataframe):
    """
    Create calendar features using the same definitions
    used during training.
    """

    dataframe["year"] = dataframe["date"].dt.year
    dataframe["month"] = dataframe["date"].dt.month
    dataframe["day"] = dataframe["date"].dt.day

    dataframe["day_of_week"] = (
        dataframe["date"].dt.dayofweek
    )

    dataframe["week_of_year"] = (
        dataframe["date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    dataframe["quarter"] = (
        dataframe["date"].dt.quarter
    )

    dataframe["is_weekend"] = (
        dataframe["day_of_week"] >= 5
    ).astype(int)

    dataframe["has_promotion"] = (
        dataframe["onpromotion"] > 0
    ).astype(int)

    return dataframe


# ============================================================
# HOLIDAY FEATURE
# ============================================================

def add_holiday_feature(
    dataframe,
    stores,
    holidays,
):
    """
    Create store-aware holiday indicator.

    National holidays apply to every store.

    Regional holidays apply when the holiday region
    matches the store state.

    Local holidays apply when the holiday location
    matches the store city.
    """

    dataframe = dataframe.merge(
        stores[
            [
                "store_nbr",
                "city",
                "state",
            ]
        ],
        on="store_nbr",
        how="left",
    )

    holidays = holidays.copy()

    # Convert transferred column to reliable boolean.
    holidays["transferred"] = (
        holidays["transferred"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # A transferred holiday should not be treated as
    # an active holiday on its original date.
    #
    # Work Day records are compensating working days,
    # so they are also excluded.
    active_holidays = holidays[
        (~holidays["transferred"])
        & (holidays["type"] != "Work Day")
    ].copy()

    national_dates = set(
        active_holidays.loc[
            active_holidays["locale"] == "National",
            "date",
        ]
    )

    regional_holidays = set(
        zip(
            active_holidays.loc[
                active_holidays["locale"] == "Regional",
                "date",
            ],
            active_holidays.loc[
                active_holidays["locale"] == "Regional",
                "locale_name",
            ],
        )
    )

    local_holidays = set(
        zip(
            active_holidays.loc[
                active_holidays["locale"] == "Local",
                "date",
            ],
            active_holidays.loc[
                active_holidays["locale"] == "Local",
                "locale_name",
            ],
        )
    )

    dataframe["is_holiday"] = [
        int(
            date in national_dates
            or (date, state) in regional_holidays
            or (date, city) in local_holidays
        )
        for date, state, city in zip(
            dataframe["date"],
            dataframe["state"],
            dataframe["city"],
        )
    ]

    print(
        f"Future rows marked as holiday: "
        f"{dataframe['is_holiday'].sum():,}"
    )

    return dataframe


# ============================================================
# RECURSIVE FORECAST
# ============================================================

def generate_recursive_forecast(
    model,
    feature_columns,
    train,
    test,
    stores,
    holidays,
):
    """
    Forecast one future day at a time.

    Lag and rolling features use previous RECORDED
    observations, matching the feature engineering
    logic used during model training.
    """

    history = train[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
        ]
    ].copy()

    history = history.sort_values(
        [
            "store_nbr",
            "family",
            "date",
        ]
    )

    # --------------------------------------------------------
    # Known future features
    # --------------------------------------------------------

    test = add_calendar_features(
        test.copy()
    )

    test = add_holiday_feature(
        test,
        stores,
        holidays,
    )

    forecast_results = []

    forecast_dates = sorted(
        test["date"].unique()
    )

    print("\n" + "=" * 60)
    print("RECURSIVE XGBOOST FORECAST")
    print("=" * 60)

    # --------------------------------------------------------
    # Forecast one date at a time
    # --------------------------------------------------------

    for forecast_date in forecast_dates:

        forecast_date = pd.Timestamp(
            forecast_date
        )

        print(
            f"Forecasting "
            f"{forecast_date.date()}..."
        )

        current_day = test[
            test["date"] == forecast_date
        ].copy()

        # ----------------------------------------------------
        # Only observations available before forecast date
        # ----------------------------------------------------

        available_history = history[
            history["date"] < forecast_date
        ].copy()

        available_history = (
            available_history
            .sort_values(
                [
                    "store_nbr",
                    "family",
                    "date",
                ]
            )
        )

        # ----------------------------------------------------
        # Create lag features using previous observations
        #
        # This matches:
        # groupby(...).shift(1)
        # groupby(...).shift(7)
        # etc. used during training.
        # ----------------------------------------------------

        def create_series_features(group):

            sales = group["sales"]

            return pd.Series(
                {
                    "sales_lag_1": (
                        sales.iloc[-1]
                        if len(sales) >= 1
                        else np.nan
                    ),

                    "sales_lag_7": (
                        sales.iloc[-7]
                        if len(sales) >= 7
                        else np.nan
                    ),

                    "sales_lag_14": (
                        sales.iloc[-14]
                        if len(sales) >= 14
                        else np.nan
                    ),

                    "sales_lag_28": (
                        sales.iloc[-28]
                        if len(sales) >= 28
                        else np.nan
                    ),

                    "rolling_mean_7": (
                        sales.tail(7).mean()
                    ),

                    "rolling_mean_14": (
                        sales.tail(14).mean()
                    ),

                    "rolling_mean_28": (
                        sales.tail(28).mean()
                    ),

                    "rolling_std_7": (
                        sales.tail(7).std()
                    ),
                }
            )

        historical_features = (
            available_history
            .groupby(
                [
                    "store_nbr",
                    "family",
                ]
            )
            .apply(
                create_series_features,
                include_groups=False,
            )
            .reset_index()
        )

        # ----------------------------------------------------
        # Add historical features to today's rows
        # ----------------------------------------------------

        current_day = current_day.merge(
            historical_features,
            on=[
                "store_nbr",
                "family",
            ],
            how="left",
        )

        # ----------------------------------------------------
        # Match categorical feature types
        # ----------------------------------------------------

        current_day["family"] = (
            current_day["family"]
            .astype("category")
        )

        current_day["store_nbr"] = (
            current_day["store_nbr"]
            .astype("category")
        )

        # ----------------------------------------------------
        # Check feature completeness
        # ----------------------------------------------------

        missing_features = (
            current_day[
                feature_columns
            ]
            .isna()
            .sum()
        )

        missing_features = (
            missing_features[
                missing_features > 0
            ]
        )

        if not missing_features.empty:

            print(
                "\nWarning: Missing model features:"
            )

            print(
                missing_features.to_string()
            )

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        X_future = current_day[
            feature_columns
        ]

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = model.predict(
            X_future
        )

        # Demand cannot be negative.
        predictions = np.maximum(
            predictions,
            0,
        )

        current_day["sales"] = predictions

        # ----------------------------------------------------
        # Store today's forecast
        # ----------------------------------------------------

        forecast_results.append(
            current_day[
                [
                    "id",
                    "date",
                    "store_nbr",
                    "family",
                    "sales",
                ]
            ].copy()
        )

        # ----------------------------------------------------
        # Add today's predictions to history
        #
        # Tomorrow can now use today's forecast as lag 1.
        # ----------------------------------------------------

        new_history = current_day[
            [
                "date",
                "store_nbr",
                "family",
                "sales",
            ]
        ].copy()

        history = pd.concat(
            [
                history,
                new_history,
            ],
            ignore_index=True,
        )

    forecast = pd.concat(
        forecast_results,
        ignore_index=True,
    )

    return forecast


# ============================================================
# MAIN
# ============================================================

def main():

    model, feature_columns = load_model()

    train, test, stores, holidays = (
        load_data()
    )

    forecast = generate_recursive_forecast(
        model,
        feature_columns,
        train,
        test,
        stores,
        holidays,
    )

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FORECAST_PATH,
        index=False,
    )

    print("\n" + "=" * 60)
    print("FORECAST COMPLETE")
    print("=" * 60)

    print(
        f"\nForecast rows: "
        f"{len(forecast):,}"
    )

    print(
        f"Forecast dates: "
        f"{forecast['date'].nunique():,}"
    )

    print(
        f"Average forecast demand: "
        f"{forecast['sales'].mean():,.2f}"
    )

    print(
        f"Minimum forecast demand: "
        f"{forecast['sales'].min():,.2f}"
    )

    print(
        f"Maximum forecast demand: "
        f"{forecast['sales'].max():,.2f}"
    )

    print(
        f"Negative forecasts: "
        f"{(forecast['sales'] < 0).sum():,}"
    )

    print(
        f"\nForecast saved to: "
        f"{FORECAST_PATH}"
    )


if __name__ == "__main__":
    main()
