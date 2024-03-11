import pandas as pd
import numpy as np
import reservoirpy as rpy
from typing import Dict, Tuple

from src.features.denoise_ndvi_signal import (
    downsample_time_series,
    moving_std_filter,
    holt_winters_filter,
    denoise_signal,
)
from src.features.slice_time_series import create_training_data
from src.models.forecaster import Forecaster


def simulate_signal(
    signal: pd.Series = None,
    model: rpy.model.Model = None,
    num_features: int = None,
    forecasted_steps: int = None,
    step_size: int = None,
) -> Dict:

    denoised_signal_series = denoise_signal(
        signal, [
            downsample_time_series,
            moving_std_filter,
            holt_winters_filter,
        ]
    )

    dates = denoised_signal_series.index
    y_dates = dates[num_features:].to_numpy()

    denoised_signal = denoised_signal_series.to_numpy()
    X, y = create_training_data(denoised_signal, num_features=num_features)
    y = y.reshape(-1, 1)

    signal_length = X.shape[0]
    end_of_training_index = signal_length - forecasted_steps

    Xtrain_signal = X[:end_of_training_index]
    ytrain_signal = y[:end_of_training_index]

    signal_forecaster = Forecaster(model, num_features=num_features)
    signal_forecaster.train(Xtrain_signal, ytrain_signal)

    warmup_X_signal = Xtrain_signal[-num_features:, :]

    signal_prediction_length = signal_length - end_of_training_index

    lower_bound = signal_forecaster.forecast(
        prediction_length=signal_prediction_length,
        warmup_X=warmup_X_signal,
    )

    lower_bound_dates = y_dates[end_of_training_index:]

    forecasts = []

    for i in range(0, forecasted_steps, step_size):

        start_index = end_of_training_index + i
        end_index = start_index + step_size
        forecast_len = signal_length - start_index

        curr_X = X[start_index: end_index]
        curr_y = y[start_index: end_index]

        signal_forecaster.train(curr_X, curr_y)

        curr_warmup_X = X[end_index - num_features: end_index]

        forecast = signal_forecaster.forecast(
            prediction_length=forecast_len,
            warmup_X=curr_warmup_X,
        )

        forecasts.append(forecast)

    return {
        "lower_bound": lower_bound,
        "lower_bound_dates": lower_bound_dates,
        "forecasts": forecasts,
    }


def detect_fault(
    N: int = None,
    k: float = None,
    step_size: int = None,
    non_change_placeholder_date: str = None,
    lower_bound: np.ndarray = None,
    lower_bound_dates: np.ndarray = None,
    forecasts: list = None,
) -> Tuple[float, pd.Timestamp]:

    flag = False
    i = 0

    for forecast in forecasts:

        forecast_length = len(forecast)

        for j in range(forecast_length - N + 1):

            flag = np.all(
                forecast[j: j + N] < k * lower_bound[i + j: i + j + N]
            )

            if flag:
                return (float(flag), lower_bound_dates[i + j])

        i += step_size

    return (float(flag), pd.Timestamp(non_change_placeholder_date))
