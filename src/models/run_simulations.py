import pickle
import yaml
import pandas as pd
from copy import deepcopy
from pathlib import Path
import reservoirpy as rpy
from typing import Dict
import logging

from src import paths
from src.models.fault_detection import simulate_signal
from src.data.utils import create_output_paths


def save_signal_simulations(signal_simulations: Dict = None,
                            simulations_path: Path = None,
                            ):

    with open(simulations_path, "wb") as file:
        pickle.dump(signal_simulations, file)


def run_simulations(
    trained_esn_path: Path = paths.models_dir(
        "trained-esn-non-feedback.pickle"),
    params_path: Path = paths.config_dir("params.yaml"),
    pre_megadrought_fault_detection_dataset_path: Path = paths.data_processed_dir(
        "fault_detection", "pre_megadrought_fault_detection_dataset.csv"),
    pre_megadrought_fault_detection_metadata_path: Path = paths.data_processed_dir(
        "fault_detection", "pre_megadrought_fault_detection_metadata.csv"),
    simulations_path: Path = paths.data_interim_dir(
        "simulations", "simulations.pickle"),
):
    logger = logging.getLogger(__name__)

    logger.info("Starting signal simulations")

    rpy.verbosity(0)

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    create_output_paths([simulations_path])

    save_interval: int = params["save_interval"]
    weeks_after_change_offset: int = params["weeks_after_change_offset"]
    esn_features_dim: int = params["esn_features_dim"]
    fault_detection_forecasted_steps: int = params["fault_detection_forecasted_steps"]
    step_size: int = params["step_size"]

    date_offset = pd.DateOffset(
        weeks=weeks_after_change_offset
    )

    pre_megadrought_fault_detection_df = pd.read_csv(
        pre_megadrought_fault_detection_dataset_path, index_col=["ID", "IDpix"])
    pre_megadrought_fault_detection_metadata_df = pd.read_csv(
        pre_megadrought_fault_detection_metadata_path, index_col=["ID", "IDpix"])

    # Sample for plug test
    # index_sample = pre_megadrought_fault_detection_df.iloc[:20].index

    # y = pre_megadrought_fault_detection_metadata_df["label"].loc[index_sample]
    y = pre_megadrought_fault_detection_metadata_df["label"]

    # X = pre_megadrought_fault_detection_df.loc[index_sample]
    X = pre_megadrought_fault_detection_df

    X.columns = pd.to_datetime(X.columns)

    change_start_dates = pre_megadrought_fault_detection_metadata_df["change_start"]
    change_start_dates = pd.to_datetime(change_start_dates)

    with open(trained_esn_path, "rb") as file:
        esn = pickle.load(file)

    num_pixles = len(X.index)
    logger.info(
        f"Begining iteration of {num_pixles} pixels for signal simulation"
    )

    signal_simulations = []

    for i, index in enumerate(X.index):

        signal = X.loc[index]

        if y.loc[index] == 0:
            bounded_signal = signal
        else:
            event_date = change_start_dates.loc[index]
            bounded_signal = signal[: event_date + date_offset]

        signal_simulation = simulate_signal(
            signal=bounded_signal,
            model=deepcopy(esn),
            num_features=esn_features_dim,
            forecasted_steps=fault_detection_forecasted_steps,
            step_size=step_size,
        )

        signal_simulations.append(signal_simulation)

        if (i + 1) % save_interval == 0:
            completion_percentage = ((i + 1) / num_pixles) * 100
            save_signal_simulations(signal_simulations, simulations_path)
            logger.info(
                f"Completed: {i+1} iterations; {completion_percentage:.2f}% of total iterations"
            )

    save_signal_simulations(signal_simulations, simulations_path)
    logger.info("Iterations completed")
