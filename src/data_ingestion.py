from pathlib import Path

import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to raw data folder
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_csv(file_name):
    """
    Load a CSV file from the raw data directory.

    Parameters:
        file_name (str): Name of the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    file_path = RAW_DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"\nLoading {file_name}...")

    dataframe = pd.read_csv(file_path)

    print(f"{file_name} loaded successfully.")
    print(f"Rows: {dataframe.shape[0]:,}")
    print(f"Columns: {dataframe.shape[1]}")

    return dataframe


def inspect_dataframe(dataframe, name):
    """
    Display basic information about a dataframe.
    """

    print("\n" + "=" * 60)
    print(f"DATASET: {name}")
    print("=" * 60)

    print("\nColumn Names:")
    print(dataframe.columns.tolist())

    print("\nData Types:")
    print(dataframe.dtypes)

    print("\nMissing Values:")
    print(dataframe.isnull().sum())

    print("\nFirst 5 Rows:")
    print(dataframe.head())


def main():

    # Load all raw datasets
    train = load_csv("train.csv")
    test = load_csv("test.csv")
    stores = load_csv("stores.csv")
    transactions = load_csv("transactions.csv")
    oil = load_csv("oil.csv")
    holidays = load_csv("holidays_events.csv")
    sample_submission = load_csv("sample_submission.csv")

    # Inspect the main training dataset
    inspect_dataframe(train, "TRAIN DATA")

    # Additional information important for forecasting
    train["date"] = pd.to_datetime(train["date"])

    print("\n" + "=" * 60)
    print("FORECASTING DATA SUMMARY")
    print("=" * 60)

    print(f"\nStart Date: {train['date'].min()}")
    print(f"End Date: {train['date'].max()}")

    print(f"\nNumber of Stores: {train['store_nbr'].nunique()}")
    print(f"Number of Product Families: {train['family'].nunique()}")

    print("\nSales Statistics:")
    print(train["sales"].describe())

    print("\nPromotion Statistics:")
    print(train["onpromotion"].describe())


if __name__ == "__main__":
    main()