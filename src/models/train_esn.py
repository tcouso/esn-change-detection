import pandas as pd
from pathlib import Path
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, RLS
import pickle
import json
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

from src import paths
from src.data.utils import create_output_path
from src.features.slice_time_series import create_training_data_for_multiple_pixels


def train_esn(
    denoised_esn_signal_dir: Path = paths.data_processed_dir("esn"),
    trained_esn_dir: Path = paths.models_dir(),
    metrics_dir: Path = paths.reports_metrics_dir(),
    params_path: Path = paths.config_dir("params.yaml"),
) -> None:

    rpy.verbosity(0)

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    esn_features_dim: int = params["esn_features_dim"]
    esn_training_years: int = params["esn_training_years"]
    weeks_per_year: int = params["weeks_per_year"]
    esn_num_units: int = params["esn_num_units"]
    esn_lr: float = params["esn_lr"]
    esn_sr: float = params["esn_sr"]

    denoised_esn_signal_filename = "esn_signal_denoised_" + selected_band + ".csv"
    denoised_signal_path = denoised_esn_signal_dir / denoised_esn_signal_filename

    denoised_esn_signal_df = pd.read_csv(
        denoised_signal_path, index_col=["ID", "IDpix"])
    denoised_esn_signal_df.columns = pd.to_datetime(
        denoised_esn_signal_df.columns)

    denoised_ts = denoised_esn_signal_df.to_numpy()

    train_size = weeks_per_year * esn_training_years

    Xtrain, ytrain, Xval, yval = create_training_data_for_multiple_pixels(
        denoised_ts, num_features=esn_features_dim, train_size=train_size)

    ytrain = ytrain.reshape(-1, 1)
    yval = yval.reshape(-1, 1)

    reservoir = Reservoir(
        units=esn_num_units,
        lr=esn_lr,
        sr=esn_sr,
    )
    readout = RLS()
    esn_model = reservoir >> readout

    esn_model.train(Xtrain, ytrain)
    ypred = esn_model.run(Xval)

    mape = mean_absolute_percentage_error(yval, ypred)
    mae = mean_absolute_error(yval, ypred)
    mse = mean_squared_error(yval, ypred)
    r2 = r2_score(yval, ypred)

    metrics = {
        "mape": mape,
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }

    model_filename = "trained_esn_" + selected_band + ".pickle"
    model_path = trained_esn_dir / model_filename
    create_output_path(model_path)

    with open(model_path, 'wb') as file:
        pickle.dump(esn_model, file)

    metrics_filename = "esn_regression_metrics_" + selected_band + ".json"
    metrics_path = metrics_dir / metrics_filename
    create_output_path(metrics_path)

    with open(metrics_path, "w") as outfile:
        json.dump(metrics, outfile)
