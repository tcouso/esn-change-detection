from pathlib import Path
import yaml
from src import paths
import pandas as pd
import numpy as np
from typing import List
from src.data.utils import create_output_paths

def esn_fault_detection_partition(
        esn_metadata_path: Path = paths.data_interim_dir("esn", "esn_metadata.csv"),
        esn_dataset_path: Path = paths.data_interim_dir("esn", "esn_dataset.csv"),
        fault_detection_metadata_path: Path = paths.data_interim_dir("fault_detection", "fault_detection_metadata.csv"),
        fault_detection_dataset_path: Path = paths.data_interim_dir("fault_detection", "fault_detection_dataset.csv"),
        params_path: Path = paths.config_dir("params.yaml"),
        ) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    # Parameters
    esn_training_percentage: float = params["esn_training_percentage"]
    metadata_columns: List[str] = params["metadata_columns"]
    event_threshold_date: str = params["event_threshold"]
    random_seed: int = params["random_seed"]
    non_change_placeholder_date = params["non_change_placeholder_date"]

    rng = np.random.default_rng(random_seed)

    stable_ndvi_path = paths.data_interim_dir("stable", "stable_ndvi.csv")
    drought_ndvi_path = paths.data_interim_dir("drought", "drought_ndvi.csv")
    logging_ndvi_path = paths.data_interim_dir("logging", "logging_ndvi.csv")
    fire_ndvi_path = paths.data_interim_dir("fire", "fire_ndvi.csv")

    # Dataframes
    stable_df = pd.read_csv(stable_ndvi_path, index_col=["ID", "IDpix"])
    drought_df = pd.read_csv(drought_ndvi_path, index_col=["ID", "IDpix"])
    logging_df = pd.read_csv(logging_ndvi_path, index_col=["ID", "IDpix"])
    fire_df = pd.read_csv(fire_ndvi_path, index_col=["ID", "IDpix"])

    # ESN dataset
    non_change_df = pd.concat((stable_df, drought_df), axis=0)
    non_change_signal_columns = list(filter(lambda col: col not in metadata_columns, (col for col in non_change_df.columns)))

    non_change_signal_df = non_change_df[non_change_signal_columns]
    non_change_signal_df.columns = pd.to_datetime(non_change_signal_df.columns)
    non_change_signal_df = non_change_signal_df.reindex(sorted(non_change_signal_df.columns), axis=1)

    non_change_metadata_df = non_change_df[metadata_columns]

    num_polygons = non_change_metadata_df.index.get_level_values(0).unique().size
    esn_n = int(num_polygons * esn_training_percentage)

    esn_indices = pd.Index(rng.choice(non_change_metadata_df.index.get_level_values(0).unique(), size=esn_n, replace=False))
    fault_detection_indices = non_change_metadata_df.index.get_level_values(0).unique()[~non_change_metadata_df.index.get_level_values(0).unique().isin(esn_indices)]

    non_change_fault_detection_metadata_df = non_change_metadata_df.loc[fault_detection_indices]
    non_change_fault_detection_signal_df = non_change_signal_df.loc[fault_detection_indices]

    esn_metadata_df = non_change_metadata_df.loc[esn_indices]    
    esn_signal_df = non_change_signal_df.loc[esn_indices]

    # Fault detection dataset
    change_df = pd.concat((logging_df, fire_df), axis=0)

    change_signal_columns = list(filter(lambda col: col not in metadata_columns, (col for col in change_df.columns)))

    change_metadata_df = change_df[metadata_columns]

    change_signal_df = change_df[change_signal_columns]
    change_signal_df.columns = pd.to_datetime(change_signal_df.columns)
    change_signal_df = change_signal_df.reindex(sorted(change_signal_df.columns), axis=1)

    # non_change_event_dates = pd.Series(non_change_fault_detection_signal_df.columns.max() +  pd.DateOffset(years=1), index=non_change_fault_detection_signal_df.index) # dummy event date for non_change signal
    non_change_fault_detection_metadata_df["change_start"] = pd.Timestamp(non_change_placeholder_date)
    non_change_fault_detection_metadata_df["label"] = 0

    event_threshold = pd.Timestamp(event_threshold_date)

    change_metadata_df.loc[:, "change_start"] = pd.to_datetime(change_metadata_df["change_start"], dayfirst=True)
    selected_change_polygons_indices = change_metadata_df.loc[change_metadata_df["change_start"] > event_threshold].index
    change_metadata_df = change_metadata_df.loc[selected_change_polygons_indices]
    change_metadata_df["label"] = 1
        
    change_signal_df = change_signal_df.loc[selected_change_polygons_indices]

    fault_detection_metadata_df = pd.concat((non_change_fault_detection_metadata_df, change_metadata_df))

    fault_detection_signal_df = pd.concat((non_change_fault_detection_signal_df, change_signal_df))
    fault_detection_signal_df = fault_detection_signal_df.reindex(sorted(fault_detection_signal_df.columns), axis=1)

    # Dataframes saving
    out_paths = [
        esn_metadata_path,
        esn_dataset_path,
        fault_detection_metadata_path,
        fault_detection_dataset_path,
    ]

    create_output_paths(out_paths)

    esn_metadata_df.to_csv(esn_metadata_path)
    esn_signal_df.to_csv(esn_dataset_path)

    fault_detection_metadata_df.to_csv(fault_detection_metadata_path)
    fault_detection_signal_df.to_csv(fault_detection_dataset_path)

