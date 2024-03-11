from pandas import Series
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import List, Callable


def downsample_time_series(time_series: Series) -> Series:
    """
    Description:
    This function takes a pandas Series `time_series` as input and
    downsamples it by week using the median of the data within each week.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.

    Returns:
    - Series: The resampled time series.
    """
    # Downsample by week
    resampled_time_series = time_series.resample("W").median()

    # Fill missing values by interpolation
    resampled_time_series = resampled_time_series.interpolate()

    # Fill initial missing values by backward fill
    resampled_time_series = resampled_time_series.bfill()

    return resampled_time_series


def moving_std_filter(
    time_series: Series, window_size: int = 26, std_range: int = 4
) -> Series:
    """
    Description:
    This function takes a pandas Series `time_series` as input and filters
    the data based on the rolling mean and standard deviation
    within a specified window.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.
    - `window_size` (int): The size of the window used to compute the rolling mean and standard deviation (default: 26).
    - `std_range` (int): The number of standard deviations used to calculate the upper and lower bounds (default: 4).

    Returns:
    - Series: The filtered time series.

    """
    # Compute rolling metrics
    rolling_mean = time_series.rolling(window=window_size).mean()
    rolling_std = time_series.rolling(window=window_size).std()

    # Compute thresholds
    rolling_threshold_ub = rolling_mean + (std_range / 2) * rolling_std
    rolling_threshold_lb = rolling_mean - (std_range / 2) * rolling_std

    # Filter out of bounds values
    filtered_time_series = time_series[
        (time_series < rolling_threshold_ub) & (
            time_series > rolling_threshold_lb)
    ]

    # Interpolate missing values
    filtered_time_series = filtered_time_series.resample("W").interpolate()

    # Fill initial missing values by backward fill
    filtered_time_series = filtered_time_series.bfill()

    return filtered_time_series


def holt_winters_filter(time_series: Series, period: int = 52) -> Series:
    """
    Description:
    This function takes a pandas Series `time_series` as input and smooths it
    using the Holt-Winters exponential smoothing method.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.
    - `period` (int): The number of periods per season for the seasonal component (default: 52).

    Returns:
    - Series: The smoothed time series using the Holt-Winters method.
    """

    model = ExponentialSmoothing(
        time_series, trend="add", seasonal="add", seasonal_periods=period
    )
    fit = model.fit()
    hw_smoothed_time_series = fit.fittedvalues

    return hw_smoothed_time_series


def denoise_signal(time_series: Series, processors: List[Callable]) -> Series:
    """
    Description:
    Iterator pattern for denoising procedure.

    Parameters:
    - `time_series` (Series): The input pandas Series containing the time series data.
    - `processors` (List[Callable]): List of functions to apply sequentially to the time series data.

    Returns:
    - Series: The processed time series.
    """
    denoised_ts = time_series.copy()
    for processor in processors:
        denoised_ts = processor(denoised_ts)

    return denoised_ts
