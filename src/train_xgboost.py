from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "featured_train.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "xgboost_demand_forecaster.joblib"
)

def load_data():
    """
    Load feature-engineered data.
    """

    dataframe = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        low_memory=False,
    )

    print(f"Loaded {len(dataframe):,} rows.")

    return dataframe


def prepare_features(dataframe):
    """
    Prepare features for XGBoost.
    """

    selected_columns = [
        "date",
        "store_nbr",
        "family",
        "sales",
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

    dataframe = dataframe[selected_columns].copy()

    # XGBoost can work with categorical features
    # when pandas categorical dtype is used.
    dataframe["family"] = dataframe["family"].astype("category")
    dataframe["store_nbr"] = dataframe["store_nbr"].astype("category")

    # Remove rows where lag/rolling features are unavailable.
    dataframe = dataframe.dropna(
        subset=[
            "sales_lag_1",
            "sales_lag_7",
            "sales_lag_14",
            "sales_lag_28",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",
            "rolling_std_7",
        ]
    )

    print(
        f"Rows after removing unavailable lag features: "
        f"{len(dataframe):,}"
    )

    return dataframe


def create_time_split(dataframe):
    """
    Split chronologically.
    """

    validation_start = pd.Timestamp("2017-07-01")

    train = dataframe[
        dataframe["date"] < validation_start
    ].copy()

    validation = dataframe[
        dataframe["date"] >= validation_start
    ].copy()

    print("\n" + "=" * 60)
    print("XGBOOST TIME-BASED SPLIT")
    print("=" * 60)

    print(f"\nTraining rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")

    return train, validation


def build_model(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
):
    """
    Create an XGBoost regression model.
    """

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
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


def evaluate_model(actual, predicted):
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



def show_feature_importance(model, feature_columns):
    """
    Display XGBoost feature importance.
    """

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        by="importance",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        importance.to_string(
            index=False
        )
    )

    return importance



def prediction_sanity_check(validation, predictions):
    """
    Inspect XGBoost predictions for unrealistic values
    and large forecasting errors.
    """

    results = validation[
        [
            "date",
            "store_nbr",
            "family",
            "sales",
        ]
    ].copy()

    results["prediction"] = predictions

    results["absolute_error"] = (
        results["sales"] - results["prediction"]
    ).abs()

    negative_predictions = (
        results["prediction"] < 0
    ).sum()

    print("\n" + "=" * 60)
    print("PREDICTION SANITY CHECK")
    print("=" * 60)

    print(
        f"\nNegative predictions: "
        f"{negative_predictions:,}"
    )

    print(
        f"Minimum prediction: "
        f"{results['prediction'].min():,.2f}"
    )

    print(
        f"Maximum prediction: "
        f"{results['prediction'].max():,.2f}"
    )

    print(
        f"Average actual sales: "
        f"{results['sales'].mean():,.2f}"
    )

    print(
        f"Average predicted sales: "
        f"{results['prediction'].mean():,.2f}"
    )

    print("\nLargest prediction errors:")

    largest_errors = results.nlargest(
        10,
        "absolute_error",
    )

    print(
        largest_errors[
            [
                "date",
                "store_nbr",
                "family",
                "sales",
                "prediction",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    return results


def tune_xgboost(
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Test a small number of XGBoost configurations
    and select the model with the lowest validation MAE.
    """

    configurations = [
        {
            "name": "Current Model",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
        },
        {
            "name": "Candidate A",
            "n_estimators": 700,
            "learning_rate": 0.05,
            "max_depth": 7,
        },
        {
            "name": "Candidate B",
            "n_estimators": 800,
            "learning_rate": 0.05,
            "max_depth": 6,
        },
        {
            "name": "Candidate C",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 8,
        },
    ]

    results = []

    best_model = None
    best_mae = float("inf")
    best_name = None

    print("\n" + "=" * 60)
    print("XGBOOST HYPERPARAMETER TUNING")
    print("=" * 60)

    for config in configurations:

        print(
            f"\nTraining {config['name']}..."
        )

        model = build_model(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_validation, y_validation)
            ],
            verbose=False,
        )

        predictions = model.predict(
            X_validation
        )

        # Demand cannot be negative.
        predictions = np.maximum(
            predictions,
            0,
        )

        mae, rmse = evaluate_model(
            y_validation,
            predictions,
        )

        results.append(
            {
                "Model": config["name"],
                "Trees": config["n_estimators"],
                "Learning Rate": config["learning_rate"],
                "Depth": config["max_depth"],
                "MAE": mae,
                "RMSE": rmse,
            }
        )

        print(f"MAE:  {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = config["name"]

    results_dataframe = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)

    print(
        results_dataframe.to_string(
            index=False
        )
    )

    print(
        f"\nBest model: {best_name}"
    )

    print(
        f"Best validation MAE: "
        f"{best_mae:,.2f}"
    )

    return best_model, results_dataframe


def save_model(model, feature_columns):
    """
    Save the trained XGBoost model and its feature list.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": model,
        "features": feature_columns,
    }

    joblib.dump(
        model_package,
        MODEL_PATH,
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )
    
    
def main():

    dataframe = load_data()

    dataframe = prepare_features(
        dataframe
    )

    train, validation = create_time_split(
        dataframe
    )

    feature_columns = [
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

    X_train = train[feature_columns]
    y_train = train["sales"]

    X_validation = validation[feature_columns]
    y_validation = validation["sales"]

    model, tuning_results = tune_xgboost(
        X_train,
        y_train,
        X_validation,
        y_validation,
    )

    print("\nTuning complete.")

    raw_predictions = model.predict(
        X_validation
    )

    predictions = np.maximum(
        raw_predictions,
        0,
    )

    mae, rmse = evaluate_model(
        y_validation,
        predictions,
    )
    
    show_feature_importance(
        model,
        feature_columns,
    )
    
    prediction_results = prediction_sanity_check(
        validation,
        predictions,
    )

    print("\n" + "=" * 60)
    print("XGBOOST VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nValidation observations: {len(y_validation):,}")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    baseline = validation.dropna(
        subset=["sales_lag_7"]
    )

    baseline_mae, baseline_rmse = evaluate_model(
        baseline["sales"],
        baseline["sales_lag_7"],
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\n{'Model':<25}"
        f"{'MAE':>15}"
        f"{'RMSE':>15}"
    )

    print("-" * 55)

    print(
        f"{'Weekly Baseline':<25}"
        f"{baseline_mae:>15,.2f}"
        f"{baseline_rmse:>15,.2f}"
    )

    print(
        f"{'XGBoost':<25}"
        f"{mae:>15,.2f}"
        f"{rmse:>15,.2f}"
    )

    improvement = (
        (baseline_mae - mae)
        / baseline_mae
        * 100
    )

    print(
        f"\nXGBoost MAE improvement vs baseline: "
        f"{improvement:.2f}%"
    )
    save_model(
        model,
        feature_columns,
    )


if __name__ == "__main__":
    main()