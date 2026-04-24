# models/arima_model.py
"""
ARIMA / SARIMA Forecasting Model
Uses pmdarima (auto_arima) for automatic order selection.
Best suited for stable SKUs with low volatility and no complex seasonality.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Tuple, Optional
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.base_model import BaseForecaster, ForecastResult
from utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class ARIMAForecaster(BaseForecaster):
    """
    Auto ARIMA wrapper with optional seasonal component (SARIMA).

    Hyperparameters:
        seasonal (bool)      : Enable seasonal component (SARIMA)
        m (int)              : Seasonal period (7=weekly, 12=monthly)
        max_p, max_q (int)   : Search bounds for AR/MA orders
        information_criterion: 'aic' | 'bic' for order selection
    """

    def __init__(self, seasonal: bool = True, m: int = 7,
                 max_p: int = 3, max_q: int = 3,
                 information_criterion: str = "aic"):
        super().__init__(name="arima")
        self.seasonal   = seasonal
        self.m          = m
        self.max_p      = max_p
        self.max_q      = max_q
        self.ic         = information_criterion
        self._model     = None
        self._fitted    = None
        self.order_     = None
        self.seasonal_order_ = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame, target_col: str = "quantity_sold") -> "ARIMAForecaster":
        """
        Fit auto_arima on the target time series.

        Args:
            train      : DataFrame with 'sale_date' and target_col sorted ascending.
            target_col : Column to forecast.
        """
        y = train.set_index("sale_date")[target_col].asfreq("D").fillna(0)

        logger.info(f"[ARIMA] Fitting on {len(y)} observations (seasonal={self.seasonal}, m={self.m})")

        self._model = auto_arima(
            y,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p, max_q=self.max_q,
            max_P=2, max_Q=2,
            d=None,          # Auto-detect differencing
            D=None,
            information_criterion=self.ic,
            stepwise=True,   # Faster than exhaustive search
            suppress_warnings=True,
            error_action="ignore",
            n_jobs=-1,
        )

        self.order_          = self._model.order
        self.seasonal_order_ = self._model.seasonal_order
        self._is_fitted      = True

        logger.info(f"[ARIMA] Best order: {self.order_} | Seasonal: {self.seasonal_order_} | AIC: {self._model.aic():.2f}")
        return self

    # ── Forecasting ───────────────────────────────────────────────────────────

    def predict(self, horizon: int, confidence: float = 0.95) -> ForecastResult:
        """
        Generate out-of-sample forecasts with confidence intervals.

        Args:
            horizon   : Number of days to forecast.
            confidence: Confidence level for intervals (0.80 or 0.95).

        Returns:
            ForecastResult with point forecasts and intervals.
        """
        self._check_fitted()

        forecast, conf_int = self._model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=1 - confidence,
        )

        # Also get 80% CI
        _, conf_int_80 = self._model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=0.20,
        )

        # Clip negatives (demand cannot be negative)
        forecast      = np.clip(forecast, 0, None)
        lower_95      = np.clip(conf_int[:, 0], 0, None)
        upper_95      = conf_int[:, 1]
        lower_80      = np.clip(conf_int_80[:, 0], 0, None)
        upper_80      = conf_int_80[:, 1]

        return ForecastResult(
            model_name  = self.name,
            horizon     = horizon,
            forecast    = forecast,
            lower_80    = lower_80,
            upper_80    = upper_80,
            lower_95    = lower_95,
            upper_95    = upper_95,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test: pd.DataFrame, target_col: str = "quantity_sold") -> Dict[str, float]:
        """
        Evaluate model on a held-out test set.
        Refits on train portion then scores on test.
        """
        self._check_fitted()
        y_true = test[target_col].values
        result = self.predict(horizon=len(y_true))
        y_pred = result.forecast

        metrics = calculate_metrics(y_true, y_pred)
        self.metrics_ = metrics
        logger.info(f"[ARIMA] Evaluation → {metrics}")
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"[ARIMA] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ARIMAForecaster":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"[ARIMA] Model loaded from {path}")
        return obj

    def get_params(self) -> Dict:
        return {
            "algorithm"           : "arima",
            "order"               : self.order_,
            "seasonal_order"      : self.seasonal_order_,
            "seasonal"            : self.seasonal,
            "m"                   : self.m,
            "information_criterion": self.ic,
        }
