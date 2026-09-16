USE PredictiveDemandForecasting;
GO


-- ============================================================
-- STORES
-- ============================================================

IF OBJECT_ID('dbo.Stores', 'U') IS NOT NULL
    DROP TABLE dbo.Stores;
GO

CREATE TABLE dbo.Stores
(
    store_nbr INT NOT NULL PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    store_type VARCHAR(20) NOT NULL,
    cluster INT NOT NULL
);
GO


-- ============================================================
-- HISTORICAL SALES
-- ============================================================

IF OBJECT_ID('dbo.HistoricalSales', 'U') IS NOT NULL
    DROP TABLE dbo.HistoricalSales;
GO

CREATE TABLE dbo.HistoricalSales
(
    id BIGINT NOT NULL PRIMARY KEY,
    sales_date DATE NOT NULL,
    store_nbr INT NOT NULL,
    family VARCHAR(100) NOT NULL,
    sales FLOAT NOT NULL,
    onpromotion INT NOT NULL,

    CONSTRAINT FK_HistoricalSales_Stores
        FOREIGN KEY (store_nbr)
        REFERENCES dbo.Stores(store_nbr)
);
GO


-- ============================================================
-- FORECASTS
-- ============================================================

IF OBJECT_ID('dbo.Forecasts', 'U') IS NOT NULL
    DROP TABLE dbo.Forecasts;
GO

CREATE TABLE dbo.Forecasts
(
    id BIGINT NOT NULL PRIMARY KEY,
    forecast_date DATE NOT NULL,
    store_nbr INT NOT NULL,
    family VARCHAR(100) NOT NULL,
    predicted_sales FLOAT NOT NULL,

    CONSTRAINT FK_Forecasts_Stores
        FOREIGN KEY (store_nbr)
        REFERENCES dbo.Stores(store_nbr)
);
GO


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX IX_HistoricalSales_Date
ON dbo.HistoricalSales(sales_date);
GO

CREATE INDEX IX_HistoricalSales_StoreFamily
ON dbo.HistoricalSales(store_nbr, family);
GO

CREATE INDEX IX_Forecasts_Date
ON dbo.Forecasts(forecast_date);
GO

CREATE INDEX IX_Forecasts_StoreFamily
ON dbo.Forecasts(store_nbr, family);
GO


-- ============================================================
-- CONFIRM TABLE CREATION
-- ============================================================

SELECT
    TABLE_SCHEMA,
    TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY TABLE_NAME;
GO