# TODO: Slice signal and store it as processed features. Keep instances dates as metadata.

import pandas as pd
from pathlib import Path
# import reservoirpy as rpy
import yaml
from src import paths
# from src.data.utils import create_output_path
from src.features.slice_time_series import create_training_data_for_multiple_pixels


def run_signal_slicing() -> None:

    params_path: Path = paths.config_dir("params.yaml")

    # rpy.verbosity(0)

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    esn_features_dim: int = params["esn_features_dim"]

    # Transform the following params to train date upper bound
    esn_training_years: int = params["esn_training_years"]
    weeks_per_year: int = params["weeks_per_year"]

    # denoised_esn_signal_filename = "esn_signal_denoised_" + selected_band + ".csv"
    denoised_train_signal_filename = "_".join(
        ["denoised_train_signal_filtered_dataset", selected_band])
    denoised_train_signal_filename += ".csv"
    # denoised_signal_path = denoised_esn_signal_dir / denoised_esn_signal_filename

    denoised_esn_signal_df = pd.read_csv(paths.features_dir(
        denoised_train_signal_filename), index_col=["ID", "IDpix"])
    denoised_esn_signal_df.columns = pd.to_datetime(
        denoised_esn_signal_df.columns)

    denoised_ts = denoised_esn_signal_df.to_numpy()

    train_size = weeks_per_year * esn_training_years

    # TODO: Instead of train_size, would it work to pass a specific top date?
    # Get the train size associated with certain date

    Xtrain, ytrain, Xval, yval = create_training_data_for_multiple_pixels(
        denoised_ts, num_features=esn_features_dim, train_size=train_size)

    ytrain = ytrain.reshape(-1, 1)
    yval = yval.reshape(-1, 1)

    # TODO: Save Files

