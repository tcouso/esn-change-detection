import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import List
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from itertools import product

from src import paths


def optimal_model_scores(
    params_path: Path = paths.config_dir("params.yaml"),
    metrics_dir: Path = paths.reports_dir("metrics"),
    fault_detection_dir: Path = paths.data_processed_dir("fault_detection"),
):
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    th_values: List[float] = params["voting_thresholds"]
    selected_score: str = params["parameter_study_max_metric_prefix"]

    num_N_values = len(N_values)
    num_k_values = len(k_values)
    num_th_values = len(th_values)

    fault_detection_metadata_filename = "_".join(
        ["fault_detection_metadata", selected_band])
    fault_detection_metadata_filename += ".csv"

    fault_detection_metadata_path = fault_detection_dir / \
        fault_detection_metadata_filename

    pixel_true_values_df = pd.read_csv(
        fault_detection_metadata_path, index_col=["ID", "IDpix"])
    poly_true_values_df = pixel_true_values_df.groupby(
        "ID")[["change_type", "change_start", "vegetation_type", "label"]].min()

    veg_type_mask = (
        poly_true_values_df["vegetation_type"] == "native")

    scores_filename = "_".join([selected_score, selected_band])
    scores_filename += ".npy"
    scores_path = metrics_dir / scores_filename

    scores = np.load(scores_path)

    max_score_index = np.argmax(scores)
    max_score_th_index, max_score_N_index, max_score_k_index = np.unravel_index(
        max_score_index, (num_th_values, num_N_values, num_k_values))

    th = th_values[max_score_th_index]
    N = N_values[max_score_N_index]
    k = k_values[max_score_k_index]

    filename = f"predictions_N={N}_k={k}_th={th}_{selected_band}.csv"
    poly_pred_path = paths.data_processed_dir("poly_predictions", filename)
    poly_pred = pd.read_csv(poly_pred_path, index_col="ID")

    y_true = poly_true_values_df.loc[veg_type_mask]["label"]
    y_pred = poly_pred.loc[veg_type_mask]["prediction"]

    cm = confusion_matrix(y_true, y_pred)

    detailed_cm = poly_true_values_df.copy()
    detailed_cm.loc[:, "prediction"] = poly_pred

    detailed_cm_abosolutes = detailed_cm[veg_type_mask][[
        "change_type", "label", "prediction"]].value_counts(normalize=False)
    detailed_cm_precentages = detailed_cm[veg_type_mask][[
        "change_type", "label", "prediction"]].value_counts(normalize=True)

    detailed_cm_abosolutes_filename = "_".join(
        ["max",
         selected_score,
         "detailed_abs_cm", selected_band,
         ]
    )
    detailed_cm_abosolutes_filename += ".csv"
    detailed_cm_abosolutes_path = metrics_dir / detailed_cm_abosolutes_filename
    detailed_cm_abosolutes.to_csv(detailed_cm_abosolutes_path)

    detailed_cm_percentages_filename = "_".join(
        ["max",
         selected_score,
         "detailed_per_cm", selected_band,
         ]
    )
    detailed_cm_percentages_filename += ".csv"
    detailed_cm_percentages_path = metrics_dir / detailed_cm_percentages_filename
    detailed_cm_precentages.to_csv(detailed_cm_percentages_path)

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    confusion_matrix_filename = "_".join(
        ["max",
         selected_score,
         "cm",
         selected_band])
    confusion_matrix_filename += ".npy"
    confusion_matrix_path = metrics_dir / confusion_matrix_filename
    np.save(confusion_matrix_path, cm)

    classification_report_filename = "_".join(
        ["max",
         selected_score,
         "classification_report",
         selected_band]
    )
    classification_report_filename += ".csv"
    classification_report_path = metrics_dir / classification_report_filename
    report_df.to_csv(classification_report_path, index=False)


def optimal_model_scores_by_event_type(
    params_path: Path = paths.config_dir("params.yaml"),
    metrics_dir: Path = paths.reports_dir("metrics"),
    fault_detection_dir: Path = paths.data_processed_dir("fault_detection"),
):
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    th_values: List[float] = params["voting_thresholds"]
    # parameter_study_selected_vegetation: str = params["parameter_study_selected_vegetation"]
    selected_score: str = params["parameter_study_max_metric_prefix"]
    stable_event_types: List[str] = params["stable_event_types"]
    change_event_types: List[str] = params["change_event_types"]

    num_N_values = len(N_values)
    num_k_values = len(k_values)
    num_th_values = len(th_values)

    fault_detection_metadata_filename = "_".join(
        ["fault_detection_metadata", selected_band])
    fault_detection_metadata_filename += ".csv"
    fault_detection_metadata_path = fault_detection_dir / \
        fault_detection_metadata_filename

    pixel_true_values_df = pd.read_csv(
        fault_detection_metadata_path, index_col=["ID", "IDpix"])
    poly_true_values_df = pixel_true_values_df.groupby(
        "ID")[["change_type", "change_start", "vegetation_type", "label"]].min()

    veg_type_mask = (
        poly_true_values_df["vegetation_type"] == "native")

    scores_filename = "_".join([selected_score, selected_band])
    scores_filename += ".npy"
    scores_path = metrics_dir / scores_filename

    scores = np.load(scores_path)

    max_score_index = np.argmax(scores)
    max_score_th_index, max_score_N_index, max_score_k_index = np.unravel_index(
        max_score_index, (num_th_values, num_N_values, num_k_values))

    th = th_values[max_score_th_index]
    N = N_values[max_score_N_index]
    k = k_values[max_score_k_index]

    filename = f"predictions_N={N}_k={k}_th={th}_{selected_band}.csv"
    poly_pred_path = paths.data_processed_dir("poly_predictions", filename)
    poly_pred = pd.read_csv(poly_pred_path, index_col="ID")

    for non_change_type, change_type in product(stable_event_types, change_event_types):

        stable_type_mask = (poly_true_values_df["change_type"] == non_change_type)
        change_type_mask = (poly_true_values_df["change_type"] == change_type)

        mask = veg_type_mask & (change_type_mask | stable_type_mask)

        y_true = poly_true_values_df.loc[mask]["label"]
        y_pred = poly_pred.loc[mask]["prediction"]

        cm = confusion_matrix(y_true, y_pred)

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        confusion_matrix_filename = "_".join([
            "max",
            selected_score,
            "cm",
            selected_band,
            non_change_type,
            change_type
        ])
        confusion_matrix_filename += ".npy"
        confusion_matrix_path = metrics_dir / confusion_matrix_filename
        np.save(confusion_matrix_path, cm)

        classification_report_filename = "_".join([
            "max",
            selected_score,
            "classification_report",
            selected_band,
            non_change_type,
            change_type
        ])
        classification_report_filename += ".csv"
        classification_report_path = metrics_dir / classification_report_filename
        report_df.to_csv(classification_report_path, index=False)
