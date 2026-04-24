# utils/metrics.py
"""
Forecast Evaluation Metrics
MAPE, MAE, RMSE, Forecast Bias
"""

import numpy as np
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard forecast accuracy metrics.

    Args:
        y_true: Actual demand values.
        y_pred: Predicted demand values.

    Returns:
        Dictionary with mape, mae, rmse, forecast_bias.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    # Avoid division by zero for MAPE
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        mape = 0.0
    else:
        mape = float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)

    mae          = float(np.mean(np.abs(y_true - y_pred)))
    rmse         = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    forecast_bias= float(np.mean(y_pred - y_true))   # positive = over-forecast

    return {
        "mape"          : round(mape, 4),
        "mae"           : round(mae, 4),
        "rmse"          : round(rmse, 4),
        "forecast_bias" : round(forecast_bias, 4),
    }


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — handles zero actuals better than standard MAPE."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Mean Absolute Scaled Error — scale-independent metric.
    Compares forecast error against naive (lag-1) in-sample error.
    """
    naive_errors = np.abs(np.diff(y_train))
    mae_naive    = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    mae_model    = np.mean(np.abs(y_true - y_pred))
    return float(mae_model / (mae_naive + 1e-8))
