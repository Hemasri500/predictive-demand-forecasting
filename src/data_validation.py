from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_train_data():
    """
    Load the raw training dataset.
    """
    file_path = RAW_DATA_DIR / "train.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dataframe = pd.read_csv(file_path)

    # Convert date column to datetime
    dataframe["date"] = pd.to_datetime(dataframe["date"])

    return dataframe


def validate_missing_values(dataframe):
    """
    Check for missing values.
    """
    print("\n1. MISSING VALUES")
    print("-" * 50)

    missing_values = dataframe.isnull().sum()

    print(missing_values)

    total_missing = missing_values.sum()

    if total_missing == 0:
        print("PASS: No missing values found.")
    else:
        print(f"WARNING: {total_missing} missing values found.")


def validate_duplicates(dataframe):
    """
    Check for duplicate rows and duplicate business keys.
    """
    print("\n2. DUPLICATE CHECK")
    print("-" * 50)

    duplicate_rows = dataframe.duplicated().sum()

    print(f"Exact duplicate rows: {duplicate_rows}")

    if duplicate_rows == 0:
        print("PASS: No exact duplicate rows found.")
    else:
        print("WARNING: Exact duplicate rows were found.")

    # Each date-store-family combination should ideally appear once
    duplicate_keys = dataframe.duplicated(
        subset=["date", "store_nbr", "family"]
    ).sum()

    print(
        f"Duplicate date-store-family combinations: "
        f"{duplicate_keys}"
    )

    if duplicate_keys == 0:
        print("PASS: Business keys are unique.")
    else:
        print(
            "WARNING: Multiple records exist for the same "
            "date-store-family combination."
        )


def validate_sales(dataframe):
    """
    Check the sales target for invalid values.
    """
    print("\n3. SALES VALIDATION")
    print("-" * 50)

    negative_sales = (dataframe["sales"] < 0).sum()
    zero_sales = (dataframe["sales"] == 0).sum()

    print(f"Negative sales values: {negative_sales:,}")
    print(f"Zero sales values: {zero_sales:,}")

    if negative_sales == 0:
        print("PASS: No negative sales values found.")
    else:
        print("WARNING: Negative sales values found.")

    print(f"Minimum sales: {dataframe['sales'].min():,.2f}")
    print(f"Maximum sales: {dataframe['sales'].max():,.2f}")


def validate_promotions(dataframe):
    """
    Check promotion values.
    """
    print("\n4. PROMOTION VALIDATION")
    print("-" * 50)

    negative_promotions = (dataframe["onpromotion"] < 0).sum()

    print(f"Negative promotion values: {negative_promotions:,}")

    if negative_promotions == 0:
        print("PASS: No negative promotion values found.")
    else:
        print("WARNING: Negative promotion values found.")

    print(
        f"Minimum onpromotion: "
        f"{dataframe['onpromotion'].min()}"
    )

    print(
        f"Maximum onpromotion: "
        f"{dataframe['onpromotion'].max()}"
    )


def validate_dates(dataframe):
    """
    Check date range and date consistency.
    """
    print("\n5. DATE VALIDATION")
    print("-" * 50)

    start_date = dataframe["date"].min()
    end_date = dataframe["date"].max()

    print(f"Start date: {start_date.date()}")
    print(f"End date: {end_date.date()}")

    unique_dates = dataframe["date"].nunique()

    expected_dates = (
        pd.date_range(start=start_date, end=end_date).size
    )

    print(f"Unique dates present: {unique_dates:,}")
    print(f"Calendar days in full range: {expected_dates:,}")

    missing_dates = (
        pd.date_range(start=start_date, end=end_date)
        .difference(dataframe["date"].unique())
    )

    print(f"Completely missing dates: {len(missing_dates):,}")

    if len(missing_dates) == 0:
        print("PASS: No full dates are missing.")
    else:
        print("WARNING: Some dates are completely absent.")

        print("First missing dates:")
        print(list(missing_dates[:10]))


def validate_categories(dataframe):
    """
    Check store and product-family dimensions.
    """
    print("\n6. STORE AND PRODUCT VALIDATION")
    print("-" * 50)

    store_count = dataframe["store_nbr"].nunique()
    family_count = dataframe["family"].nunique()

    print(f"Unique stores: {store_count}")
    print(f"Unique product families: {family_count}")

    print("\nStore number range:")
    print(
        f"{dataframe['store_nbr'].min()} "
        f"to {dataframe['store_nbr'].max()}"
    )

    print("\nProduct families:")
    print(sorted(dataframe["family"].unique()))


def validate_data_types(dataframe):
    """
    Validate expected column data types.
    """
    print("\n7. DATA TYPE VALIDATION")
    print("-" * 50)

    print(dataframe.dtypes)

    expected_columns = {
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
    }

    actual_columns = set(dataframe.columns)

    missing_columns = expected_columns - actual_columns

    if not missing_columns:
        print("PASS: All expected columns are present.")
    else:
        print(f"WARNING: Missing columns: {missing_columns}")


def run_validation(dataframe):
    """
    Run all validation checks.
    """
    print("\n" + "=" * 60)
    print("DEMAND FORECASTING DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"\nRows: {len(dataframe):,}")
    print(f"Columns: {dataframe.shape[1]}")

    validate_missing_values(dataframe)
    validate_duplicates(dataframe)
    validate_sales(dataframe)
    validate_promotions(dataframe)
    validate_dates(dataframe)
    validate_categories(dataframe)
    validate_data_types(dataframe)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


def main():
    train = load_train_data()
    run_validation(train)


if __name__ == "__main__":
    main()