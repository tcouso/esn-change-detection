import yaml
from pathlib import Path
from src import paths
from itertools import product
from typing import List
import pandas as pd
import pickle

from src.models.change_detection import detect_change
from src.data.utils import create_output_path


def predict_over_pixels(
    simulations_dir: Path = paths.data_interim_dir("simulations"),
    fault_detection_dir: Path = paths.data_processed_dir("fault_detection"),
    params_path: Path = paths.config_dir("params.yaml"),
) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    step_size: int = params["step_size"]
    non_change_placeholder_date: str = params["non_change_placeholder_date"]

    fault_detection_metadata_filename = "fault_detection_metadata_" + selected_band + ".csv"
    fault_detection_metadata_path = fault_detection_dir / \
        fault_detection_metadata_filename
    fault_detection_metadata_df = pd.read_csv(
        fault_detection_metadata_path, index_col=["ID", "IDpix"])

    simulations_filename = "signal_simulations_" + selected_band + ".pickle"
    simulations_path = simulations_dir / simulations_filename

    with open(simulations_path, "rb") as file:
        signal_simulations = pickle.load(file)

    for N, k in product(N_values, k_values):

        simulation_results_dict = {
            "prediction": [],
            "event_date": [],
        }

        for simulation in signal_simulations:

            fault_detection_params = simulation
            fault_detection_params["N"] = N
            fault_detection_params["k"] = k
            fault_detection_params["step_size"] = step_size
            fault_detection_params["non_change_placeholder_date"] = non_change_placeholder_date

            pred, date = detect_change(**fault_detection_params)

            simulation_results_dict["prediction"].append(pred)
            simulation_results_dict["event_date"].append(date)

        y_pred = pd.DataFrame(simulation_results_dict)
        y_pred.index = fault_detection_metadata_df.index
        y_pred["event_date"] = pd.to_datetime(y_pred["event_date"])

        filename = f"predictions_N={N}_k={k}_" + selected_band + ".csv"
        y_pred_path = paths.data_processed_dir("pixel_predictions", filename)
        create_output_path(y_pred_path)

        y_pred.to_csv(y_pred_path)
