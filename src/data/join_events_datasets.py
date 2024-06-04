from pathlib import Path
import yaml
from src import paths
import pandas as pd
from src.data.utils import create_output_path


def join_events_datasets() -> None:

    # Read files
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    stable_signal_path = paths.data_interim_dir(
        "stable", "stable_" + selected_band + ".csv")
    drought_signal_path = paths.data_interim_dir(
        "drought", "drought_" + selected_band + ".csv")
    logging_signal_path = paths.data_interim_dir(
        "logging", "logging_" + selected_band + ".csv")
    fire_signal_path = paths.data_interim_dir(
        "fire", "fire_" + selected_band + ".csv")

    # Dataframe merging
    stable_df = pd.read_csv(stable_signal_path, index_col=["ID", "IDpix"])
    drought_df = pd.read_csv(drought_signal_path, index_col=["ID", "IDpix"])
    logging_df = pd.read_csv(logging_signal_path, index_col=["ID", "IDpix"])
    fire_df = pd.read_csv(fire_signal_path, index_col=["ID", "IDpix"])

    df = pd.concat((stable_df, drought_df, logging_df, fire_df), axis=0)

    # Save files
    dataset_filename = "_".join(["dataset", selected_band])
    dataset_filename += ".csv"

    df.to_csv(paths.data_interim_dir(dataset_filename))
