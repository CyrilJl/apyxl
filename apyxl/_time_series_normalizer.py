import numpy as np
import pandas as pd

from ._misc import check_params
from ._xgb import XGBRegressorWrapper


class TimeSeriesNormalizer:
    """
    TimeSeriesNormalizer normalizes a target time series based on external features using XGBoost regression.

    Attributes:
        freq_trend (pd.tseries.offsets.DateOffset): Frequency for trend calculation.
        xgb (XGBRegressorWrapper): XGBoost regressor wrapper for normalization.
    """

    def __init__(self, freq_trend: str, max_evals: int = 15):
        """
        Initializes the TimeSeriesNormalizer class.

        Args:
            freq_trend (str): Frequency string for resampling the time series trend.
            max_evals (int, optional): Maximum number of evaluations for the XGBoost model. Defaults to 15.
        """
        self.xgb = XGBRegressorWrapper(max_evals=max_evals)
        self.freq_trend = pd.tseries.frequencies.to_offset(freq_trend)

    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None, target: str = None):
        """
        Preprocesses the input data by selecting the target column and checking the index.

        Args:
            X (pd.DataFrame): DataFrame containing external features.
            y (pd.Series, optional): Target time series. If None, the target column in X will be used.
            target (str, optional): Name of the target column in X. Required if y is None.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Processed X and y.

        Raises:
            AssertionError: If neither `y` nor `target` is provided.
            ValueError: If the target column in X contains NaN values.
        """
        assert (y is not None) or (target is not None), "'y' or 'target' must be provided."

        if y is None:
            if target not in X.columns:
                raise ValueError(f"Target column '{target}' not found in X.")
            y = X[target].copy()
            X = X.drop(columns=target)
        else:
            check_params(param=y, types=pd.Series)

        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("Data must be a pandas DataFrame or Series.")
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("The index is not a DatetimeIndex.")

        y = y.dropna()
        X = X.reindex(index=y.index)
        X['time_numeric'] = (X.index - X.index.min()).floor(self.freq_trend) / self.freq_trend
        return X, y

    @staticmethod
    def shift_auto(trend):
        n_bins = int(np.power(len(trend), 1/3))
        counts, bins = np.histogram(trend.values, bins=max(2, n_bins))
        i0 = counts.argmax()
        return -0.5*(bins[i0] + bins[i0+1])

    @classmethod
    def apply_shift(cls, trend: pd.Series, y: pd.Series, shift="auto") -> pd.Series:
        """
        Applies the specified shift to the trend.

        Args:
            trend (pd.Series): The normalized trend series.
            y (pd.Series): The original target time series.
            shift (float or str, optional): Value to shift the trend by. Can be a float/int or 'mean'. Defaults to None.

        Returns:
            pd.Series: Shifted trend series.

        Raises:
            ValueError: If the shift value is invalid.
        """
        if shift == 'auto':
            s = cls.shift_auto(trend)
        elif shift == 'mean':
            s = y.mean()
        elif shift is None:
            s = 0.0
        elif isinstance(shift, (float, int)):
            s = shift
        else:
            raise ValueError("Shift must be either 'auto', 'mean', None, or a numeric value.")
        return trend + s

    def normalize(self, X: pd.DataFrame, y: pd.Series = None, target: str = None, shift: float = None) -> pd.Series:
        """
        Normalizes the target time series using external features.

        Args:
            X (pd.DataFrame): DataFrame containing external features.
            y (pd.Series, optional): Target time series. If None, the target column in X will be used.
            target (str, optional): Name of the target column in X. Required if y is None.
            shift (float or str, optional): Value to shift the trend by. Can be a float/int or 'mean'. Defaults to None.

        Returns:
            pd.Series: Normalized time series.
        """
        Xp, yp = self.preprocess_data(X=X, y=y, target=target)

        self.xgb.fit(X=Xp, y=yp)
        shap_values = self.xgb.compute_shap_values(Xp)
        trend = pd.Series(shap_values[:, 'time_numeric'].values, index=Xp.index, name=f'normalized {yp.name}')
        trend_shifted = self.apply_shift(trend=trend, y=yp, shift=shift)
        return trend_shifted
