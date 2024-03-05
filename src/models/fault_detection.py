import pandas as pd
import numpy as np
import reservoirpy as rpy

from foreutils.denoising import (
    denoise_time_series,
    downsample_time_series,
    moving_std_filter,
    holt_winters_filter,
)
from foreutils.utils import create_training_data
from foreutils.forecasting import Forecaster


def simulate_signal(
    signal: pd.Series,
    ESN_signal: rpy.model.Model,
    num_features: int = 104,
    forecasted_steps: int = 52,
) -> dict:

    # Denoise signal
    denoised_signal_series = denoise_time_series(
        signal, [downsample_time_series, moving_std_filter, holt_winters_filter]
    )
    denoised_signal = denoised_signal_series.to_numpy()

    # Training data setup
    X, y = create_training_data(denoised_signal, num_features=num_features)
    y = y.reshape(-1, 1)

    # Parameters setup
    T = len(X)
    s = T - forecasted_steps

    # Signal forecasting
    Xtrain_signal = X[:s]
    ytrain_signal = y[:s]

    signal_forecaster = Forecaster(ESN_signal, num_features=num_features)
    signal_forecaster.train(Xtrain_signal, ytrain_signal)

    warmup_X_signal = Xtrain_signal[-num_features:, :]
    denoised_signal_prediction = signal_forecaster.forecast(
        T=T - s, warmup_X=warmup_X_signal
    )

    # Compute lower bound
    lower_bound = denoised_signal_prediction[:T]

    # Fault detection
    forecasts = []

    for i in range(0, forecasted_steps):

        # Iteration parameters
        start_index = s + i
        end_index = start_index + 1
        forecast_len = T - start_index

        # Iteration dataset
        curr_X = X[start_index:end_index]
        curr_y = y[start_index:end_index]

        # Train signal forecaster
        signal_forecaster.train(curr_X, curr_y)

        # Predict denoised signal in generative mode
        curr_warmup_X = X[end_index - num_features : end_index]
        forecast = signal_forecaster.forecast(T=forecast_len, warmup_X=curr_warmup_X)

        # Store forecast
        forecasts.append(forecast)

    # Return fault detection data structure
    return {
        "lower_bound": lower_bound,
        "forecasts": forecasts,
    }


def detect_fault(
    N: int,
    k: float,
    lower_bound: np.ndarray,
    forecasts: list,
) -> float:

    flag = False

    for i in range(len(forecasts)):
        forecast = forecasts[i]

        for j in range(len(forecast) - N + 1):
            assert forecast.shape == lower_bound.shape
            flag = np.all(forecast[j : j + N] < k * lower_bound[i + j : i + j + N])

            if flag:
                return float(flag)
        return float(flag)
