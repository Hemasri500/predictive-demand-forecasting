from pathlib import Path

import pandas as pd
import pyodbc


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stores.csv"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train.csv"
)

FORECAST_PATH = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "xgboost_forecast.csv"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

SERVER = r".\SQLEXPRESS"
DATABASE = "PredictiveDemandForecasting"

CONNECTION_STRING = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)


def connect_to_database():
    """
    Connect to SQL Server using Windows Authentication.
    """

    connection = pyodbc.connect(
        CONNECTION_STRING
    )

    print("Connected to SQL Server successfully.")

    return connection


# ============================================================
# CLEAR EXISTING DATA
# ============================================================

def clear_tables(connection):
    """
    Remove existing rows before loading fresh data.
    """

    cursor = connection.cursor()

    # Child tables must be cleared before Stores
    # because of foreign-key relationships.

    cursor.execute(
        "DELETE FROM dbo.Forecasts;"
    )

    cursor.execute(
        "DELETE FROM dbo.HistoricalSales;"
    )

    cursor.execute(
        "DELETE FROM dbo.Stores;"
    )

    connection.commit()

    print("Existing SQL table data cleared.")


# ============================================================
# LOAD STORES
# ============================================================

def load_stores(connection):
    """
    Load stores.csv into dbo.Stores.
    """

    stores = pd.read_csv(
        STORES_PATH
    )

    # SQL table uses store_type instead of type.
    stores = stores.rename(
        columns={
            "type": "store_type",
        }
    )

    cursor = connection.cursor()

    cursor.fast_executemany = True

    rows = list(
        stores[
            [
                "store_nbr",
                "city",
                "state",
                "store_type",
                "cluster",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    cursor.executemany(
        """
        INSERT INTO dbo.Stores
        (
            store_nbr,
            city,
            state,
            store_type,
            cluster
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.commit()

    print(
        f"Stores loaded: {len(stores):,}"
    )


# ============================================================
# LOAD HISTORICAL SALES
# ============================================================

def load_historical_sales(connection):
    """
    Load train.csv into dbo.HistoricalSales
    in chunks to avoid loading unnecessary SQL insert
    operations one row at a time.
    """

    chunk_size = 100_000

    total_loaded = 0

    cursor = connection.cursor()

    cursor.fast_executemany = True

    print("\nLoading historical sales...")

    for chunk in pd.read_csv(
        TRAIN_PATH,
        chunksize=chunk_size,
    ):

        rows = list(
            chunk[
                [
                    "id",
                    "date",
                    "store_nbr",
                    "family",
                    "sales",
                    "onpromotion",
                ]
            ].itertuples(
                index=False,
                name=None,
            )
        )

        cursor.executemany(
            """
            INSERT INTO dbo.HistoricalSales
            (
                id,
                sales_date,
                store_nbr,
                family,
                sales,
                onpromotion
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        connection.commit()

        total_loaded += len(chunk)

        print(
            f"Historical rows loaded: "
            f"{total_loaded:,}"
        )

    print(
        f"\nHistorical sales complete: "
        f"{total_loaded:,}"
    )


# ============================================================
# LOAD FORECASTS
# ============================================================

def load_forecasts(connection):
    """
    Load XGBoost future forecasts into dbo.Forecasts.
    """

    forecast = pd.read_csv(
        FORECAST_PATH
    )

    cursor = connection.cursor()

    cursor.fast_executemany = True

    rows = list(
        forecast[
            [
                "id",
                "date",
                "store_nbr",
                "family",
                "sales",
            ]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    cursor.executemany(
        """
        INSERT INTO dbo.Forecasts
        (
            id,
            forecast_date,
            store_nbr,
            family,
            predicted_sales
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )

    connection.commit()

    print(
        f"Forecasts loaded: "
        f"{len(forecast):,}"
    )


# ============================================================
# VALIDATE DATABASE
# ============================================================

def validate_database(connection):
    """
    Verify SQL Server row counts.
    """

    query = """
    SELECT
        (SELECT COUNT(*) FROM dbo.Stores)
            AS stores,

        (SELECT COUNT(*) FROM dbo.HistoricalSales)
            AS historical_sales,

        (SELECT COUNT(*) FROM dbo.Forecasts)
            AS forecasts;
    """

    result = pd.read_sql(
        query,
        connection,
    )

    print("\n" + "=" * 60)
    print("SQL DATABASE VALIDATION")
    print("=" * 60)

    print(
        f"\nStores: "
        f"{result.loc[0, 'stores']:,}"
    )

    print(
        f"Historical sales: "
        f"{result.loc[0, 'historical_sales']:,}"
    )

    print(
        f"Forecasts: "
        f"{result.loc[0, 'forecasts']:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    connection = connect_to_database()

    try:

        clear_tables(
            connection
        )

        load_stores(
            connection
        )

        load_historical_sales(
            connection
        )

        load_forecasts(
            connection
        )

        validate_database(
            connection
        )

    finally:

        connection.close()

        print(
            "\nSQL Server connection closed."
        )


if __name__ == "__main__":
    main()