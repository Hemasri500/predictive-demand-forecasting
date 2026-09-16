USE PredictiveDemandForecasting;
GO


-- ============================================================
-- CLEAR EXISTING DATA
-- ============================================================

DELETE FROM dbo.Forecasts;
DELETE FROM dbo.HistoricalSales;
DELETE FROM dbo.Stores;
GO


-- ============================================================
-- LOAD STORES
-- ============================================================

BULK INSERT dbo.Stores
FROM 'C:\Users\Hemasri Reddy\OneDrive\Desktop\Projects\Predictive-Demand-Forecasting\data\raw\stores.csv'
WITH
(
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    TABLOCK
);
GO


-- ============================================================
-- LOAD HISTORICAL SALES
-- ============================================================

BULK INSERT dbo.HistoricalSales
FROM 'C:\Users\Hemasri Reddy\OneDrive\Desktop\Projects\Predictive-Demand-Forecasting\data\raw\train.csv'
WITH
(
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    TABLOCK
);
GO


-- ============================================================
-- LOAD FORECASTS
-- ============================================================

BULK INSERT dbo.Forecasts
FROM 'C:\Users\Hemasri Reddy\OneDrive\Desktop\Projects\Predictive-Demand-Forecasting\data\forecasts\xgboost_forecast.csv'
WITH
(
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDQUOTE = '"',
    TABLOCK
);
GO


-- ============================================================
-- VALIDATE ROW COUNTS
-- ============================================================

SELECT
    'Stores' AS table_name,
    COUNT(*) AS row_count
FROM dbo.Stores

UNION ALL

SELECT
    'HistoricalSales',
    COUNT(*)
FROM dbo.HistoricalSales

UNION ALL

SELECT
    'Forecasts',
    COUNT(*)
FROM dbo.Forecasts;
GO