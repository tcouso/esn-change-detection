from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from typing import List
from itertools import product
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

from src import paths
from src.data.utils import create_output_path


def enumerated_product(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def scores_over_parameters(
    params_path: Path = paths.config_dir("params.yaml"),
    fault_detection_dir: Path = paths.data_processed_dir("fault_detection"),
    metrics_dir: Path = paths.reports_dir("metrics"),
) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    parameter_study_selected_vegetation: str = params["parameter_study_selected_vegetation"]

    fault_detection_metadata_filename = "fault_detection_metadata_" + selected_band + ".csv"
    fault_detection_metadata_path = fault_detection_dir / \
        fault_detection_metadata_filename
    pixel_true_values_df = pd.read_csv(
        fault_detection_metadata_path, index_col=["ID", "IDpix"])

    poly_true_values_df = pixel_true_values_df.groupby(
        "ID")[["vegetation_type", "label"]].min()

    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    th_values: List[float] = params["voting_thresholds"]

    num_N_values = len(N_values)
    num_k_values = len(k_values)
    num_th_values = len(th_values)

    acc_scores = np.zeros(
        (num_th_values, num_N_values, num_k_values))
    recall_scores = np.zeros(
        (num_th_values, num_N_values, num_k_values))
    precision_scores = np.zeros(
        (num_th_values, num_N_values, num_k_values))
    f1_scores = np.zeros(
        (num_th_values, num_N_values, num_k_values))

    veg_type_mask = (
        poly_true_values_df["vegetation_type"] == parameter_study_selected_vegetation)

    for triad_index, triad in enumerated_product(th_values, N_values, k_values):
        th, N, k = triad
        th_index, N_index, k_index = triad_index

        filename = f"predictions_N={N}_k={k}_th={th}.csv"

        poly_pred_path = paths.data_processed_dir("poly_predictions", filename)
        poly_pred = pd.read_csv(poly_pred_path, index_col="ID")

        sel_y_true = poly_true_values_df[veg_type_mask]["label"]
        sel_y_pred = poly_pred[veg_type_mask]["prediction"]

        acc_scores[th_index][N_index][k_index] = accuracy_score(
            sel_y_true, sel_y_pred)
        recall_scores[th_index][N_index][k_index] = recall_score(
            sel_y_true, sel_y_pred)
        precision_scores[th_index][N_index][k_index] = precision_score(
            sel_y_true, sel_y_pred, zero_division=0)
        f1_scores[th_index][N_index][k_index] = f1_score(
            sel_y_true, sel_y_pred)

    acc_scores_filename = "acc_scores_" + selected_band + ".npy"
    f1_scores_filename = "f1_scores_" + selected_band + ".npy"
    precision_scores_filename = "precision_scores_" + selected_band + ".npy"
    recall_scores_filename = "recall_scores_" + selected_band + ".npy"

    acc_scores_path = metrics_dir / acc_scores_filename
    f1_scores_path = metrics_dir / f1_scores_filename
    precision_scores_path = metrics_dir / precision_scores_filename
    recall_scores_path = metrics_dir / recall_scores_filename

    out_paths = [
        acc_scores_path,
        f1_scores_path,
        precision_scores_path,
        recall_scores_path
    ]

    for out_path in out_paths:
        create_output_path(out_path)

    np.save(acc_scores_path, acc_scores)
    np.save(f1_scores_path, f1_scores)
    np.save(precision_scores_path, precision_scores)
    np.save(recall_scores_path, recall_scores)
