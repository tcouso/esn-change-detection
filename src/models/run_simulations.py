import pickle
import yaml
import pandas as pd
from copy import deepcopy
from pathlib import Path
import reservoirpy as rpy
from typing import Dict
from tqdm import tqdm

from src.config import get_logger
from src.notifications import send_telegram_notification
from src import paths
from src.models.fault_detection import simulate_signal
from src.data.utils import create_output_path


def save_signal_simulations(signal_simulations: Dict = None,
                            simulations_path: Path = None,
                            ):

    with open(simulations_path, "wb") as file:
        pickle.dump(signal_simulations, file)


def run_simulations(
    # trained_esn_path: Path = paths.models_dir(
    #     "trained-esn-non-feedback.pickle"),
    # pre_megadrought_fault_detection_dataset_path: Path = paths.data_processed_dir(
    #     "fault_detection", "pre_megadrought_fault_detection_dataset.csv"),
    # pre_megadrought_fault_detection_metadata_path: Path = paths.data_processed_dir(
    #     "fault_detection", "pre_megadrought_fault_detection_metadata.csv"),
    # simulations_path: Path = paths.data_interim_dir(
    #     "simulations", "simulations.pickle"),
    trained_esn_dir: Path = paths.models_dir(),
    fault_detection_dir: Path = paths.data_processed_dir("fault_detection"),
    simulations_dir: Path = paths.data_interim_dir("simulations"),
    params_path: Path = paths.config_dir("params.yaml"),
):
    rpy.verbosity(0)

    logger = get_logger()

    msg = "Loading parameters"
    logger.info(msg)
    send_telegram_notification(msg)

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    save_interval: int = params["save_interval"]
    weeks_after_change_offset: int = params["weeks_after_change_offset"]
    esn_features_dim: int = params["esn_features_dim"]
    fault_detection_forecasted_steps: int = params["fault_detection_forecasted_steps"]
    step_size: int = params["step_size"]

    date_offset = pd.DateOffset(
        weeks=weeks_after_change_offset
    )

    msg = "Loading datasets"
    logger.info(msg)
    send_telegram_notification(msg)

    fault_detection_metadata_filename = "fault_detection_metadata_" + selected_band + ".csv"
    fault_detection_metadata_path = fault_detection_dir / \
        fault_detection_metadata_filename

    fault_detection_signal_filename = "fault_detection_signal_" + selected_band + ".csv"
    fault_detection_signal_path = fault_detection_dir / fault_detection_signal_filename

    fault_detection_metadata_df = pd.read_csv(
        fault_detection_metadata_path, index_col=["ID", "IDpix"])
    fault_detection_signal_df = pd.read_csv(
        fault_detection_signal_path, index_col=["ID", "IDpix"])

    y = fault_detection_metadata_df["label"]
    X = fault_detection_signal_df

    X.columns = pd.to_datetime(X.columns)

    change_start_dates = fault_detection_metadata_df["change_start"]
    change_start_dates = pd.to_datetime(change_start_dates)

    msg = "Loading model"
    logger.info(msg)
    send_telegram_notification(msg)

    model_filename = "trained_esn_" + selected_band + ".pickle"
    model_path = trained_esn_dir / model_filename

    with open(model_path, "rb") as file:
        esn = pickle.load(file)

    num_pixles = len(X.index)

    msg = f"Begining iteration of {num_pixles} pixels for signal simulation"
    logger.info(msg)
    send_telegram_notification(msg)

    simulations_filename = "signal_simulations_" + selected_band + ".pickle"
    simulations_path = simulations_dir / simulations_filename

    create_output_path(simulations_path)

    signal_simulations = []

    with tqdm(total=num_pixles) as pbar:
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

            pbar.update(1)

            if (i + 1) % save_interval == 0:
                completion_percentage = ((i + 1) / num_pixles) * 100

                save_signal_simulations(signal_simulations, simulations_path)
                msg = f"Completed: {i+1} iterations; {completion_percentage:.2f}% of total iterations"
                logger.info(msg)
                send_telegram_notification(msg)

        save_signal_simulations(signal_simulations, simulations_path)

    msg = "Iterations completed"
    logger.info(msg)
    send_telegram_notification(msg)
