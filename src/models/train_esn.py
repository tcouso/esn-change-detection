import numpy as np
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


def train_esn() -> None:

    params_path: Path = paths.config_dir("params.yaml")

    rpy.verbosity(0)

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    esn_num_units: int = params["esn_num_units"]
    esn_lr: float = params["esn_lr"]
    esn_sr: float = params["esn_sr"]

    # Load data
    Xtrain = np.load(paths.features_dir("Xtrain.npy"))
    ytrain = np.load(paths.features_dir("ytrain.npy"))

    Xval = np.load(paths.features_dir("Xval.npy"))
    yval = np.load(paths.features_dir("yval.npy"))

    # Instance model
    reservoir = Reservoir(
        units=esn_num_units,
        lr=esn_lr,
        sr=esn_sr,
    )
    readout = RLS()
    esn_model = reservoir >> readout

    # Train model
    esn_model.train(Xtrain, ytrain)
    ypred = esn_model.run(Xval)

    # Compute metrics
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

    # Save files
    model_filename = "_".join(["trained_esn", selected_band])
    model_filename += ".pickle"

    with open(paths.models_dir(model_filename), 'wb') as file:
        pickle.dump(esn_model, file)

    metrics_filename = "_".join(["esn_regression_metrics", selected_band])
    metrics_filename += ".json"

    with open(paths.reports_metrics_dir(metrics_filename), "w") as outfile:
        json.dump(metrics, outfile)
