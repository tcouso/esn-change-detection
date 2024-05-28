import numpy as np
import pandas as pd
import yaml
import json
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
) -> None:
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
        "ID")[["change_type", "change_start", "last_non_change_date", "vegetation_type", "label"]].min()

    poly_true_values_df["change_start"] = pd.to_datetime(
        poly_true_values_df["change_start"])
    poly_true_values_df["last_non_change_date"] = pd.to_datetime(
        poly_true_values_df["last_non_change_date"])

    veg_type_mask = (
        poly_true_values_df["vegetation_type"] == "native")

    scores_filename = "_".join([selected_score, selected_band])
    scores_filename += ".npy"
    scores_path = metrics_dir / scores_filename

    scores = np.load(scores_path)

    # Optimal model params

    max_score_index = np.argmax(scores)
    max_score_th_index, max_score_N_index, max_score_k_index = np.unravel_index(
        max_score_index, (num_th_values, num_N_values, num_k_values))

    th = th_values[max_score_th_index]
    N = N_values[max_score_N_index]
    k = k_values[max_score_k_index]

    optimal_model_params = {
        "th": th,
        "N": N,
        "k": k
    }

    filename = f"predictions_N={N}_k={k}_th={th}_{selected_band}.csv"
    poly_pred_path = paths.data_processed_dir("poly_predictions", filename)
    poly_pred = pd.read_csv(poly_pred_path, index_col="ID")
    poly_pred["event_date"] = pd.to_datetime(poly_pred["event_date"])

    y_true = poly_true_values_df.loc[veg_type_mask]["label"]
    y_pred = poly_pred.loc[veg_type_mask]["prediction"]

    # Confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    detailed_cm = poly_true_values_df.copy()
    detailed_cm.loc[:, "prediction"] = poly_pred["prediction"]

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

    # Classification report

    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Detection dates metrics

    true_positives_indices = y_true[
            (y_true == y_pred) &
            (y_true == 1)
        ].index

    false_positives_indices = y_true[
        (y_true != y_pred) &
        (y_true == 0)
    ].index

    time_deltas_non_change_to_detection = (
        poly_pred.loc[true_positives_indices]["event_date"] -
        poly_true_values_df.loc[true_positives_indices]["last_non_change_date"]
    )
    time_deltas_detenction_to_change_start = (
        poly_pred.loc[true_positives_indices]["event_date"] - 
        poly_true_values_df.loc[true_positives_indices]["change_start"]
    )

    false_positives_detection_dates_description = poly_pred.loc[false_positives_indices]["event_date"].describe(
    )

    # Saving

    # Optimal model params

    optimal_model_params_filename = "_".join(
        ["max",
         selected_score,
         "params",
         selected_band])
    optimal_model_params_filename += ".json"
    optimal_model_params_path = metrics_dir / optimal_model_params_filename

    with open(optimal_model_params_path, "w") as file:
        json.dump(optimal_model_params, file)

    # Confusion matrix

    confusion_matrix_filename = "_".join(
        ["max",
         selected_score,
         "cm",
         selected_band])
    confusion_matrix_filename += ".npy"
    confusion_matrix_path = metrics_dir / confusion_matrix_filename
    np.save(confusion_matrix_path, cm)

    # Classification report

    classification_report_filename = "_".join(
        ["max",
         selected_score,
         "classification_report",
         selected_band]
    )
    classification_report_filename += ".csv"
    classification_report_path = metrics_dir / classification_report_filename

    report_df.to_csv(classification_report_path, index=True)

    # Detection dates

    non_change_to_detection_filename = "_".join(
        ["max",
         selected_score,
         "non_change_to_detection",
         selected_band]
    )
    non_change_to_detection_filename += ".csv"
    non_change_to_detection_path = metrics_dir / non_change_to_detection_filename

    time_deltas_non_change_to_detection.to_csv(non_change_to_detection_path)

    detection_to_change_start_filename = "_".join(
        ["max",
         selected_score,
         "detection_to_change_start",
         selected_band]
    )
    detection_to_change_start_filename += ".csv"
    detection_to_change_start_path = metrics_dir / detection_to_change_start_filename

    time_deltas_detenction_to_change_start.to_csv(
        detection_to_change_start_path)

    false_positives_detection_dates_filename = "_".join(
        ["max",
         selected_score,
         "false_positives_detection_dates",
         selected_band]
    )
    false_positives_detection_dates_filename += ".csv"
    false_positives_detection_dates_path = metrics_dir / \
        false_positives_detection_dates_filename

    false_positives_detection_dates_description.to_csv(
        false_positives_detection_dates_path)


def optimal_model_scores_by_event_type(
    params_path: Path = paths.config_dir("params.yaml"),
    metrics_dir: Path = paths.reports_dir("metrics"),
    fault_detection_dir: Path = paths.data_processed_dir("fault_detection"),
) -> None:
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    th_values: List[float] = params["voting_thresholds"]

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
        "ID")[["change_type", "change_start", "last_non_change_date", "vegetation_type", "label"]].min()

    poly_true_values_df["change_start"] = pd.to_datetime(
        poly_true_values_df["change_start"])
    poly_true_values_df["last_non_change_date"] = pd.to_datetime(
        poly_true_values_df["last_non_change_date"])

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
    poly_pred["event_date"] = pd.to_datetime(poly_pred["event_date"])

    for non_change_type, change_type in product(stable_event_types, change_event_types):

        stable_type_mask = (
            poly_true_values_df["change_type"] == non_change_type)
        change_type_mask = (poly_true_values_df["change_type"] == change_type)

        mask = veg_type_mask & (change_type_mask | stable_type_mask)

        y_true = poly_true_values_df.loc[mask]["label"]
        y_pred = poly_pred.loc[mask]["prediction"]

        # Confusion matrix

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

        # Classification report

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
        report_df.to_csv(classification_report_path, index=True)

        # Detection dates

        # Detection dates metrics

        true_positives_indices = y_true[
            (y_true == y_pred) &
            (y_true == 1)
        ].index

        false_positives_indices = y_true[
            (y_true != y_pred) &
            (y_true == 0)
        ].index

        time_deltas_non_change_to_detection = (
            poly_pred.loc[true_positives_indices]["event_date"] -
            poly_true_values_df.loc[true_positives_indices]["last_non_change_date"]
        )
        time_deltas_detenction_to_change_start = (
            poly_pred.loc[true_positives_indices]["event_date"] -
            poly_true_values_df.loc[true_positives_indices]["change_start"]
        )

        false_positives_detection_dates_description = poly_pred.loc[false_positives_indices]["event_date"].describe(
        )

        non_change_to_detection_filename = "_".join(
            ["max",
             selected_score,
             "non_change_to_detection",
             selected_band,
             non_change_type,
             change_type]
        )
        non_change_to_detection_filename += ".csv"
        non_change_to_detection_path = metrics_dir / non_change_to_detection_filename

        time_deltas_non_change_to_detection.to_csv(
            non_change_to_detection_path)

        detection_to_change_start_filename = "_".join(
            ["max",
             selected_score,
             "detection_to_change_start",
             selected_band,
             non_change_type,
             change_type]
        )
        detection_to_change_start_filename += ".csv"
        detection_to_change_start_path = metrics_dir / detection_to_change_start_filename

        time_deltas_detenction_to_change_start.to_csv(
            detection_to_change_start_path)

        false_positives_detection_dates_filename = "_".join(
            ["max",
             selected_score,
             "false_positives_detection_dates",
             selected_band,
             non_change_type,
             change_type]
        )
        false_positives_detection_dates_filename += ".csv"
        false_positives_detection_dates_path = metrics_dir / \
            false_positives_detection_dates_filename

        false_positives_detection_dates_description.to_csv(
            false_positives_detection_dates_path)
