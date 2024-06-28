"""
This module provides tools for detecting changes in pixel NDVI simulations using parallel processing.

Classes:
    PixelChangeDetection: Stores detected changes for a single pixel.

Functions:
    detect_changes_in_pixel(pixel_simulation, N=10, k=0.8, offset=1, forecasted_steps=52) -> List[pd.Timestamp]:
        Detects changes in a pixel simulation by comparing forecasted values with actual values.
    detect_changes_in_pixel_caller(args):
        Helper function to unpack arguments for detect_changes_in_pixel.
    parallel_detect_changes_in_pixels(simulations, num_processes, change_detection_forecasted_steps, N_values, k_values, offset_values) -> List[PixelChangeDetection]:
        Parallel process to detect changes in multiple pixel simulations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from itertools import product

from src.inference.pixel_simulation import PixelSimulation
from src.inference.pandas_wrapper import PandasWrapper


class PixelChangeDetection:
    """
    Stores detected changes for a single pixel.

    Attributes
    ----------
    index : Tuple[int, int]
        The (row, column) index of the pixel.
    detected_changes : pd.Series
        The series of detected changes.
    offset : int
        The offset between actual and predicted signals.
    N : int
        The length of the detection window.
    k : float
        The threshold factor for detecting significant changes.

    Methods
    -------
    __repr__() -> str
        Returns a string representation of the object.
    __str__() -> str
        Returns a string representation of the object.
    """

    def __init__(self, index: Tuple[int, int], detected_changes: pd.Series, offset: int, N: int, k: float):
        self.index = index
        self.detected_changes = detected_changes
        self.offset = offset
        self.N = N
        self.k = k

    def __repr__(self) -> str:
        return f"PixelChangeDetection_{self.index[0]}_{self.index[1]}"

    def __str__(self) -> str:
        return f"PixelChangeDetection_{self.index[0]}_{self.index[1]}"


def detect_changes_in_pixel(
    pixel_simulation: PixelSimulation,
    N: int,
    k: float,
    offset: int,
    forecasted_steps: int,
) -> PixelChangeDetection:
    """
    Detects changes in a pixel simulation.

    Parameters
    ----------
    pixel_simulation : PixelSimulation
        The PixelSimulation object containing the actual and predicted NDVI signals.
    N : int
        The length of the detection window.
    k : float
        The threshold factor for detecting significant changes.
    offset : int
        The offset between the baseline signal and the predicted signal to compare.
    forecasted_steps : int
        The number of steps in each forecast.

    Returns
    -------
    pd.Series
        A Series of binary values (0 or 1) indicating significant changes, indexed by timestamps.
    """
    detections = []
    dates = []

    num_of_detections = (len(pixel_simulation) - 1) - offset
    num_of_windows = forecasted_steps - offset

    for i in range(num_of_detections):
        for j in range(num_of_windows):
            detection_window = pixel_simulation[i + offset].iloc[j:j+N].to_numpy(
            ) < k * pixel_simulation[i].iloc[j:j+N].to_numpy()

            flag = np.all(detection_window)
            date = pixel_simulation[i].index[j]

            detections.append(float(flag))
            dates.append(date)

    detection_series = pd.Series(detections, index=pd.to_datetime(dates))

    pixel_change_detection = PixelChangeDetection(
        index=pixel_simulation.index,
        offset=offset,
        N=N,
        k=k,
        detected_changes=detection_series,
    )

    return pixel_change_detection


def detect_changes_in_pixel_caller(args):
    """
    Helper function to unpack arguments for detect_changes_in_pixel.

    Parameters
    ----------
    args : tuple
        The arguments to be passed to detect_changes_in_pixel.

    Returns
    -------
    List[pd.Timestamp]
        The result of the detect_changes_in_pixel function.
    """
    pixel_simulation, N, k, offset, forecasted_steps = args
    return detect_changes_in_pixel(pixel_simulation, N, k, offset, forecasted_steps)


def parallel_detect_changes_in_pixels(
    simulations: PandasWrapper,
    num_processes: int,
    change_detection_forecasted_steps: int,
    N_values: List[int],
    k_values: List[float],
    offset_values: List[int]
) -> List[PixelChangeDetection]:
    """
    Parallel process to detect changes in multiple pixel simulations.

    Parameters
    ----------
    simulations : PandasWrapper
        Wrapper containing the pixel simulations.
    num_processes : int
        Number of parallel processes.
    change_detection_forecasted_steps : int
        Number of steps in each forecast.
    N_values : List[int]
        List of detection window lengths.
    k_values : List[float]
        List of threshold factors for detecting significant changes.
    offset_values : List[int]
        List of offsets between the actual signal and the predicted signal to compare.

    Returns
    -------
    List[PixelChangeDetection]
        List of PixelChangeDetection objects for each pixel.
    """
    pixel_indices = simulations.index

    args_list = list(product(pixel_indices, N_values, k_values,
                     offset_values, [change_detection_forecasted_steps]))
    args_list = [(simulations.loc[id], n, k, offset, steps)
                 for id, n, k, offset, steps in args_list]

    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(
            detect_changes_in_pixel_caller, args) for args in args_list]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting changes"):
            results.append(future.result())

    return results
