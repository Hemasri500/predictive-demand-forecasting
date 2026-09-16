USE PredictiveDemandForecasting;
GO


-- ============================================================
-- VIEW 1: DETAILED FORECAST
-- ============================================================
-- Combines forecast predictions with store/location information.
-- This becomes the main detailed forecast dataset for Power BI.

CREATE OR ALTER VIEW dbo.vw_ForecastDetail
AS

SELECT
    f.id,
    f.forecast_date,
    f.store_nbr,
    s.city,
    s.state,
    s.store_type,
    s.cluster,
    f.family,
    f.predicted_sales

FROM dbo.Forecasts AS f

INNER JOIN dbo.Stores AS s
    ON f.store_nbr = s.store_nbr;
GO


-- ============================================================
-- VIEW 2: DAILY FORECAST
-- ============================================================
-- Total predicted demand for each future date.

CREATE OR ALTER VIEW dbo.vw_DailyForecast
AS

SELECT
    forecast_date,
    SUM(predicted_sales) AS total_predicted_sales

FROM dbo.Forecasts

GROUP BY
    forecast_date;
GO


-- ============================================================
-- VIEW 3: STORE FORECAST
-- ============================================================
-- Total predicted demand by store.

CREATE OR ALTER VIEW dbo.vw_StoreForecast
AS

SELECT
    f.store_nbr,
    s.city,
    s.state,
    s.store_type,
    s.cluster,
    SUM(f.predicted_sales) AS total_predicted_sales

FROM dbo.Forecasts AS f

INNER JOIN dbo.Stores AS s
    ON f.store_nbr = s.store_nbr

GROUP BY
    f.store_nbr,
    s.city,
    s.state,
    s.store_type,
    s.cluster;
GO


-- ============================================================
-- VIEW 4: PRODUCT FAMILY FORECAST
-- ============================================================
-- Total predicted demand by product family.

CREATE OR ALTER VIEW dbo.vw_FamilyForecast
AS

SELECT
    family,
    SUM(predicted_sales) AS total_predicted_sales

FROM dbo.Forecasts

GROUP BY
    family;
GO


-- ============================================================
-- VIEW 5: REGIONAL FORECAST
-- ============================================================
-- Forecast demand aggregated by state and city.

CREATE OR ALTER VIEW dbo.vw_RegionalForecast
AS

SELECT
    s.state,
    s.city,
    SUM(f.predicted_sales) AS total_predicted_sales

FROM dbo.Forecasts AS f

INNER JOIN dbo.Stores AS s
    ON f.store_nbr = s.store_nbr

GROUP BY
    s.state,
    s.city;
GO


-- ============================================================
-- VIEW 6: HISTORICAL DAILY SALES
-- ============================================================
-- Historical company-wide demand trend.

CREATE OR ALTER VIEW dbo.vw_HistoricalDailySales
AS

SELECT
    sales_date,
    SUM(sales) AS total_sales

FROM dbo.HistoricalSales

GROUP BY
    sales_date;
GO


-- ============================================================
-- VERIFY REPORTING VIEWS
-- ============================================================

SELECT
    TABLE_SCHEMA,
    TABLE_NAME

FROM INFORMATION_SCHEMA.VIEWS

WHERE TABLE_SCHEMA = 'dbo'

ORDER BY
    TABLE_NAME;
GO