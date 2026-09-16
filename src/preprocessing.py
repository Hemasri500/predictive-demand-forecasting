from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    """
    Load all datasets needed for preprocessing.
    """

    train = pd.read_csv(RAW_DATA_DIR / "train.csv")
    stores = pd.read_csv(RAW_DATA_DIR / "stores.csv")
    transactions = pd.read_csv(RAW_DATA_DIR / "transactions.csv")
    oil = pd.read_csv(RAW_DATA_DIR / "oil.csv")
    holidays = pd.read_csv(RAW_DATA_DIR / "holidays_events.csv")

    return train, stores, transactions, oil, holidays


def convert_dates(train, transactions, oil, holidays):
    """
    Convert date columns from strings to datetime.
    """

    train["date"] = pd.to_datetime(train["date"])
    transactions["date"] = pd.to_datetime(transactions["date"])
    oil["date"] = pd.to_datetime(oil["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    return train, transactions, oil, holidays


def prepare_holidays(holidays):
    """
    Prepare holiday data before merging.

    Some dates can contain more than one holiday/event record,
    so we reduce them to one row per date.
    """

    holidays = holidays.copy()

    holidays["is_holiday"] = 1

    holiday_summary = (
        holidays.groupby("date", as_index=False)
        .agg(
            is_holiday=("is_holiday", "max"),
            holiday_type=("type", "first"),
            holiday_locale=("locale", "first"),
            holiday_name=("description", "first"),
            transferred=("transferred", "max"),
        )
    )

    return holiday_summary


def merge_datasets(
    train,
    stores,
    transactions,
    oil,
    holidays,
):
    """
    Merge all datasets into one analytical dataframe.
    """

    dataframe = train.copy()

    # Add store information
    dataframe = dataframe.merge(
        stores,
        on="store_nbr",
        how="left",
    )

    # Add transaction information
    dataframe = dataframe.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left",
    )

    # Add oil prices
    dataframe = dataframe.merge(
        oil,
        on="date",
        how="left",
    )

    # Prepare and add holiday information
    holiday_summary = prepare_holidays(holidays)

    dataframe = dataframe.merge(
        holiday_summary,
        on="date",
        how="left",
    )

    return dataframe


def handle_missing_values(dataframe):
    """
    Handle missing values created after merging datasets.
    """

    dataframe = dataframe.copy()

    # Missing holiday means there was no holiday/event that day
    dataframe["is_holiday"] = dataframe["is_holiday"].fillna(0)

    dataframe["holiday_type"] = dataframe["holiday_type"].fillna("None")
    dataframe["holiday_locale"] = dataframe["holiday_locale"].fillna("None")
    dataframe["holiday_name"] = dataframe["holiday_name"].fillna("None")
    dataframe["transferred"] = dataframe["transferred"].fillna(False)

    # Oil prices are not available for every calendar day.
    # Forward-fill and backward-fill missing values.
    dataframe["dcoilwtico"] = (
        dataframe["dcoilwtico"]
        .ffill()
        .bfill()
    )

    return dataframe


def validate_processed_data(dataframe):
    """
    Display basic checks after preprocessing.
    """

    print("\n" + "=" * 60)
    print("PROCESSED DATA SUMMARY")
    print("=" * 60)

    print(f"\nRows: {len(dataframe):,}")
    print(f"Columns: {dataframe.shape[1]}")

    print("\nColumns:")
    print(dataframe.columns.tolist())

    print("\nMissing Values:")
    print(dataframe.isnull().sum())

    print("\nFirst 5 Rows:")
    print(dataframe.head())


def save_processed_data(dataframe):
    """
    Save processed dataset to the processed data folder.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = PROCESSED_DATA_DIR / "processed_train.csv"

    dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"\nProcessed dataset saved to:")
    print(output_path)


def main():
    print("Loading raw datasets...")

    train, stores, transactions, oil, holidays = load_data()

    print("Converting date columns...")

    train, transactions, oil, holidays = convert_dates(
        train,
        transactions,
        oil,
        holidays,
    )

    print("Merging datasets...")

    dataframe = merge_datasets(
        train,
        stores,
        transactions,
        oil,
        holidays,
    )

    print("Handling missing values...")

    dataframe = handle_missing_values(dataframe)

    validate_processed_data(dataframe)

    save_processed_data(dataframe)


if __name__ == "__main__":
    main()