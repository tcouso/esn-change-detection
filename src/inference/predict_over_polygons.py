import yaml
from pathlib import Path
from src import paths
from itertools import product
from typing import List
import pandas as pd

from src.data.utils import create_output_path


def predict_over_polygons(
    params_path: Path = paths.config_dir("params.yaml"),
) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    th_values: List[float] = params["voting_thresholds"]

    for N, k, th in product(N_values, k_values, th_values):

        filename = f"predictions_N={N}_k={k}_" + selected_band + ".csv"
        pix_pred_path = paths.data_processed_dir("pixel_predictions", filename)

        pix_pred = pd.read_csv(pix_pred_path, index_col=["ID", "IDpix"])

        poly_pred = pix_pred.groupby("ID")["prediction"].mean().apply(
            lambda x: 1.0 if x >= th else 0.0)
        poly_pred_detection_dates = pix_pred.groupby("ID")["event_date"].max()

        poly_pred_df = pd.DataFrame({"prediction": poly_pred, "event_date": poly_pred_detection_dates})

        filename = f"predictions_N={N}_k={k}_th={th}_" + selected_band + ".csv"
        poly_pred_path = paths.data_processed_dir("poly_predictions", filename)
        create_output_path(poly_pred_path)

        poly_pred_df.to_csv(poly_pred_path)
