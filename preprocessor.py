# etl/preprocessor.py
"""
ETL Pipeline — Data Ingestion, Validation, Cleaning & Feature Engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ── 1. Schema Validation ─────────────────────────────────────────────────────

REQUIRED_COLUMNS = ["sale_date", "sku_id", "store_id", "quantity_sold", "unit_price"]
OPTIONAL_COLUMNS = ["category", "promotion_flag", "channel", "cost_price"]


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the uploaded DataFrame has the required columns and
    correct data types.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if errors:
        return False, errors

    # Check data types
    try:
        df["sale_date"] = pd.to_datetime(df["sale_date"])
    except Exception:
        errors.append("Column 'sale_date' cannot be parsed as a date.")

    if not pd.api.types.is_numeric_dtype(df["quantity_sold"]):
        errors.append("Column 'quantity_sold' must be numeric.")

    if not pd.api.types.is_numeric_dtype(df["unit_price"]):
        errors.append("Column 'unit_price' must be numeric.")

    # Check for negative quantities
    if (df["quantity_sold"] < 0).any():
        errors.append("Column 'quantity_sold' contains negative values.")

    # Check for negative prices
    if (df["unit_price"] < 0).any():
        errors.append("Column 'unit_price' contains negative values.")

    return len(errors) == 0, errors


# ── 2. Data Cleaning ──────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw sales data:
    - Remove exact duplicates
    - Handle missing values
    - Detect and cap outliers
    - Standardize column formats
    """
    original_rows = len(df)

    # Parse date
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date

    # Remove exact duplicates
    df = df.drop_duplicates(subset=["sale_date", "sku_id", "store_id"])
    logger.info(f"Removed {original_rows - len(df)} duplicate rows.")

    # Fill optional columns
    if "promotion_flag" not in df.columns:
        df["promotion_flag"] = False
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    if "channel" not in df.columns:
        df["channel"] = "in-store"

    df["promotion_flag"] = df["promotion_flag"].fillna(False).astype(bool)

    # Handle missing quantity_sold via forward fill within SKU group
    df = df.sort_values(["sku_id", "sale_date"])
    df["quantity_sold"] = (
        df.groupby("sku_id")["quantity_sold"]
        .transform(lambda x: x.fillna(method="ffill").fillna(0))
    )

    # Cap outliers using IQR per SKU
    df = _cap_outliers(df, column="quantity_sold", group_by="sku_id")

    logger.info(f"Data cleaning complete. Rows: {len(df)}")
    return df


def _cap_outliers(df: pd.DataFrame, column: str, group_by: str, factor: float = 3.0) -> pd.DataFrame:
    """Cap values above Q3 + factor*IQR per group."""
    def cap_group(group):
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + factor * IQR
        group[column] = group[column].clip(upper=upper)
        return group

    return df.groupby(group_by, group_keys=False).apply(cap_group)


# ── 3. Aggregation ───────────────────────────────────────────────────────────

def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to daily SKU-level totals (sum across stores).
    Adds revenue column.
    """
    daily = (
        df.groupby(["sale_date", "sku_id", "category"])
        .agg(
            quantity_sold=("quantity_sold", "sum"),
            unit_price=("unit_price", "mean"),
            promotion_flag=("promotion_flag", "max"),
        )
        .reset_index()
    )
    daily["revenue"] = daily["quantity_sold"] * daily["unit_price"]
    daily["sale_date"] = pd.to_datetime(daily["sale_date"])
    return daily.sort_values(["sku_id", "sale_date"]).reset_index(drop=True)


# ── 4. Feature Engineering ───────────────────────────────────────────────────

HOLIDAYS_2024_2025 = [
    "2024-01-01", "2024-01-26", "2024-08-15", "2024-10-02",
    "2024-11-01", "2024-12-25", "2025-01-01", "2025-01-26",
    "2025-08-15", "2025-10-02", "2025-12-25",
]


def engineer_features(df: pd.DataFrame, holiday_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create time-series features for ML models:
    - Temporal features (day of week, month, quarter, year)
    - Lag features (7, 14, 28 days)
    - Rolling statistics (mean, std over 7 and 28 days)
    - Holiday flag
    - Trend component (days since first sale)
    """
    if holiday_dates is None:
        holiday_dates = HOLIDAYS_2024_2025

    holidays = pd.to_datetime(holiday_dates)
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    # ── Temporal features
    df["day_of_week"]  = df["sale_date"].dt.dayofweek       # 0=Mon
    df["day_of_month"] = df["sale_date"].dt.day
    df["month"]        = df["sale_date"].dt.month
    df["quarter"]      = df["sale_date"].dt.quarter
    df["year"]         = df["sale_date"].dt.year
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["holiday_flag"] = df["sale_date"].isin(holidays).astype(int)

    # ── Lag & rolling features per SKU
    df = df.sort_values(["sku_id", "sale_date"])

    for lag in [7, 14, 28]:
        df[f"lag_{lag}"] = (
            df.groupby("sku_id")["quantity_sold"]
            .transform(lambda x: x.shift(lag))
        )

    for window in [7, 28]:
        df[f"rolling_mean_{window}"] = (
            df.groupby("sku_id")["quantity_sold"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("sku_id")["quantity_sold"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    # ── Trend: days since first sale for each SKU
    first_dates = df.groupby("sku_id")["sale_date"].transform("min")
    df["trend"] = (df["sale_date"] - first_dates).dt.days

    # Drop rows where lags are NaN (first 28 days per SKU)
    df = df.dropna(subset=["lag_28"]).reset_index(drop=True)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


# ── 5. Train/Test Split ──────────────────────────────────────────────────────

def time_series_split(df: pd.DataFrame, test_weeks: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets preserving temporal order.
    Test set = last `test_weeks` weeks of data.
    """
    cutoff = df["sale_date"].max() - timedelta(weeks=test_weeks)
    train = df[df["sale_date"] <= cutoff].copy()
    test  = df[df["sale_date"] >  cutoff].copy()
    logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows | Cutoff: {cutoff.date()}")
    return train, test


# ── 6. Full Pipeline ─────────────────────────────────────────────────────────

def run_etl_pipeline(raw_df: pd.DataFrame) -> Dict:
    """
    Execute the full ETL pipeline end-to-end.

    Steps:
        1. Schema validation
        2. Data cleaning
        3. Daily aggregation
        4. Feature engineering

    Returns a dict with processed DataFrame and quality report.
    """
    logger.info("Starting ETL pipeline...")

    # Step 1: Validate
    is_valid, errors = validate_schema(raw_df)
    if not is_valid:
        return {"success": False, "errors": errors, "data": None}

    # Step 2: Clean
    cleaned = clean_data(raw_df)

    # Step 3: Aggregate
    daily = aggregate_to_daily(cleaned)

    # Step 4: Feature engineering
    featured = engineer_features(daily)

    quality_report = {
        "original_rows"  : len(raw_df),
        "cleaned_rows"   : len(cleaned),
        "final_rows"     : len(featured),
        "unique_skus"    : featured["sku_id"].nunique(),
        "date_range"     : f"{featured['sale_date'].min().date()} → {featured['sale_date'].max().date()}",
        "missing_pct"    : round(featured.isnull().mean().mean() * 100, 2),
    }

    logger.info(f"ETL complete. Quality report: {quality_report}")
    return {"success": True, "errors": [], "data": featured, "report": quality_report}
