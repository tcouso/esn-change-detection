import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from src import paths
from src.features.slice_time_series import create_training_data_for_multiple_pixels


def run_signal_slicing() -> None:
    params_path: Path = paths.config_dir("params.yaml")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    esn_features_dim: int = params["esn_features_dim"]
    validation_limit_date: str = params["validation_limit_date"]

    denoised_train_signal_filename = "_".join(
        ["denoised_train_signal_filtered_dataset", selected_band])
    denoised_train_signal_filename += ".csv"

    denoised_esn_signal_df = pd.read_csv(paths.features_dir(
        denoised_train_signal_filename), index_col=["ID", "IDpix"])
    denoised_esn_signal_df.columns = pd.to_datetime(
        denoised_esn_signal_df.columns)

    # Extract training size according to training threshold date
    train_size = denoised_esn_signal_df.loc[:, denoised_esn_signal_df.columns < pd.Timestamp(
        validation_limit_date)].shape[1]

    denoised_ts = denoised_esn_signal_df.to_numpy()
    Xtrain, ytrain, Xval, yval = create_training_data_for_multiple_pixels(
        denoised_ts, num_features=esn_features_dim, train_size=train_size)

    ytrain = ytrain.reshape(-1, 1)
    yval = yval.reshape(-1, 1)

    np.save(paths.features_dir("Xtrain.npy"), Xtrain)
    np.save(paths.features_dir("ytrain.npy"), ytrain)

    np.save(paths.features_dir("Xval.npy"), Xval)
    np.save(paths.features_dir("yval.npy"), yval)
