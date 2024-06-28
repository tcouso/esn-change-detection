"""
This module provides tools for simulating NDVI dynamics for individual pixels using a trained ESN model,
as well as parallel processing capabilities for efficiency.

Classes:
    PixelSimulation: Stores and manages the simulation of NDVI dynamics for a single pixel.
    _LocIndexer: Helper class for PandasWrapper to support .loc indexing with lazy loading.
    _iLocIndexer: Helper class for PandasWrapper to support .iloc indexing with lazy loading.

Functions:
    simulate_pixel(model, pixel_index, signal, num_features=104, forecasted_steps=52) -> PixelSimulation:
        Simulates the future signal dynamics for a pixel using a trained ESN model.
    simulate_pixel_caller(args):
        Helper function to unpack arguments for simulate_pixel.
    parallel_simulate_pixels(model_path, signal_df, num_features=104, forecasted_steps=52, num_processes=4) -> List[PixelSimulation]:
        Parallel process each pixel for NDVI forecasting using a pool of workers.
"""

from typing import Tuple, List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import pickle
from tqdm import tqdm

from src.inference.forecaster import Forecaster
from src.features.slice_time_series import create_training_data
from src.features.denoise_signal import (
    resample_time_series,
    moving_std_filter,
    holt_winters_filter,
    denoise_signal,
)


class PixelSimulation:
    """
    A class to store and manage a simulation of the NDVI dynamics for a single pixel.

    Attributes
    ----------
    index : Tuple[int, int]
        The (row, column) index of the pixel in the image.
    actual_signal : pd.Series
        The actual NDVI signal for the pixel.
    _pred_signal : List[pd.Series]
        A list of predicted NDVI signals for the pixel.
    _date_to_index : dict[pd.Timestamp, int]
        A mapping from date to the index of the corresponding predicted signal in _pred_signal.
    _index_to_date : dict[int, pd.Timestamp]
        A mapping from the index of a predicted signal in _pred_signal to its corresponding date.

    Methods
    -------
    __getitem__(index) -> pd.Series
        Returns the predicted signal at the specified index.
    __iter__() -> 'PixelSimulation'
        Initializes the iterator and returns the iterator object.
    __next__() -> pd.Series
        Returns the next predicted signal in the iteration.
    __len__() -> int
        Returns the number of predicted signals.
    append(pred: pd.Series) -> None
        Appends a predicted signal to the list of predicted signals.
    get_pred_from_date(date: pd.Timestamp) -> pd.Series
        Returns the predicted signal for a specified date.
    get_date_from_pred(index: int) -> pd.Timestamp
        Returns the date for a specified predicted signal index.
    """

    def __init__(self, index: Tuple[int, int], actual_signal: pd.Series):
        self.index = index
        self.actual_signal = actual_signal
        self._pred_signal = []
        self._date_to_index = {}
        self._index_to_date = {}
        self.__index = 0

    def __repr__(self) -> str:
        return f"PixelSimulation_{self.index[0]}_{self.index[1]}"

    def __str__(self) -> str:
        return f"PixelSimulation_{self.index[0]}_{self.index[1]}"

    def __getitem__(self, index) -> pd.Series:
        try:
            return self._pred_signal[index]
        except IndexError:
            raise IndexError("Index out of range.")

    def __iter__(self):
        self.__index = 0
        return self

    def __next__(self) -> pd.Series:
        if self.__index < len(self._pred_signal):
            result = self._pred_signal[self.__index]
            self.__index += 1
            return result
        else:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._pred_signal)

    def append(self, pred: pd.Series) -> None:
        date = pred.index[0]
        index = len(self._pred_signal)
        self._date_to_index[date] = index
        self._index_to_date[index] = date
        self._pred_signal.append(pred)

    def get_pred_from_date(self, date: pd.Timestamp) -> pd.Series:
        try:
            index = self._date_to_index[date]
            return self._pred_signal[index]
        except KeyError:
            raise KeyError(f"No prediction found for date {date}.")

    def get_date_from_pred(self, index: int) -> pd.Timestamp:
        try:
            date = self._index_to_date[index]
            return date
        except KeyError:
            raise KeyError(f"No date found for index {index}.")


def simulate_pixel(
    model: object,
    pixel_index: Tuple[int, int],
    signal: pd.Series,
    num_features: int,
    forecasted_steps: int,
) -> PixelSimulation:
    """
    Simulates the future signal dynamics for a pixel using a trained ESN model.

    Parameters
    ----------
    model : object
        The trained ESN model.
    pixel_index : Tuple[int, int]
        The (row, column) index of the pixel in the image.
    signal : pd.Series
        The actual NDVI signal for the pixel.
    num_features : int, optional
        Number of features used for training, by default 104.
    forecasted_steps : int, optional
        Number of steps to forecast, by default 52.

    Returns
    -------
    PixelSimulation
        An object containing the simulation results for the pixel.
    """
    denoised_signal_series = denoise_signal(
        signal, [
            resample_time_series,
            moving_std_filter,
            holt_winters_filter,
        ]
    )

    denoised_signal_dates = denoised_signal_series.index.to_numpy().reshape(-1, 1)
    X_dates = denoised_signal_dates[num_features:, :]

    denoised_signal = denoised_signal_series.to_numpy()
    X, _ = create_training_data(denoised_signal, num_features=num_features)

    signal_forecaster = Forecaster(model, num_features=num_features)
    num_forecasts = X.shape[0] - forecasted_steps

    pixel_simulation = PixelSimulation(
        index=pixel_index, actual_signal=denoised_signal_series)

    for i in range(num_forecasts):
        warmup = X[i:i + num_features, :]

        pred_signal = signal_forecaster.forecast(
            prediction_length=forecasted_steps,
            warmup_X=warmup,
        )
        pred_dates = X_dates[i: i + forecasted_steps, :]

        pred_series = pd.Series(pred_signal.flatten(),
                                index=pred_dates.flatten())
        pixel_simulation.append(pred_series)

    return pixel_simulation


def simulate_pixel_caller(args):
    """
    Helper function to unpack arguments for simulate_pixel.

    Parameters
    ----------
    args : tuple
        The arguments to be passed to simulate_pixel.

    Returns
    -------
    PixelSimulation
        The result of the simulate_pixel function.
    """
    model, pixel_index, signal, num_features, forecasted_steps = args
    return simulate_pixel(model, pixel_index, signal, num_features, forecasted_steps)


def parallel_simulate_pixels(
    model_path: str,
    signal_df: pd.DataFrame,
    num_features: int = 104,
    forecasted_steps: int = 52,
    num_processes: int = 4
) -> List[PixelSimulation]:
    """
    Parallel process each pixel for NDVI forecasting using a pool of workers.

    Parameters
    ----------
    model_path : str
        Path to the pretrained model pickle file.
    signal_df : pd.DataFrame
        DataFrame containing the NDVI signals.
    num_features : int, optional
        Number of features used for training, by default 104.
    forecasted_steps : int, optional
        Number of steps to forecast, by default 52.
    num_processes : int, optional
        Number of parallel processes, by default 4.

    Returns
    -------
    List[PixelSimulation]
        List of PixelSimulation objects for each pixel.
    """
    with open(model_path, "rb") as file:
        pretrained_model = pickle.load(file)

    pixel_indices = signal_df.index

    args_list = [(copy.deepcopy(pretrained_model),
                  idx,
                  signal_df.loc[idx],
                  num_features,
                  forecasted_steps) for idx in pixel_indices]

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(simulate_pixel_caller, args)
                   for args in args_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Simulating pixels"):
            results.append(future.result())

    return results