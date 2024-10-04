import numpy as np
import pandas as pd
import pytest

from apyxl import TimeSeriesNormalizer


@pytest.fixture
def sample_data():
    n = 8760
    time = pd.date_range(start='2024-01-01', freq='h', periods=n)

    cov = [[1, 0.7], [0.7, 1]]
    mean = [0, 5]

    df = np.random.multivariate_normal(cov=cov, mean=mean, size=n)
    df[:, 1] *= 2

    df[6000:7000, 1] += 2

    return pd.DataFrame(df, columns=['a', 'b'], index=time)


def test_time_series_normalizer_normalize(sample_data):
    tsn = TimeSeriesNormalizer(freq_trend='1d')
    trend = tsn.normalize(sample_data, target='b')

    assert isinstance(trend, pd.Series)
    assert len(trend) == len(sample_data)
    assert trend.index.equals(sample_data.index)


def test_time_series_normalizer_output_shape(sample_data):
    tsn = TimeSeriesNormalizer(freq_trend='1d')
    trend = tsn.normalize(sample_data, target='b')

    assert trend.shape == (8760,)


def test_time_series_normalizer_with_different_freq(sample_data):
    tsn_hourly = TimeSeriesNormalizer(freq_trend='1h')
    tsn_daily = TimeSeriesNormalizer(freq_trend='1d')

    trend_hourly = tsn_hourly.normalize(sample_data, target='b')
    trend_daily = tsn_daily.normalize(sample_data, target='b')

    assert not np.allclose(trend_hourly, trend_daily)  # The trends should be different
