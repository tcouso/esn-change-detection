# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import yaml
from src import paths
import pandas as pd
import numpy as np
from typing import Dict, List

def join_ndvi_time_series(
    params_path: Path = paths.config_dir("params.yaml"),
    ) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    directory_file_prefix_map: Dict[str, str] = params["directory_file_prefix_map"]
    selected_directories: List[str] = params["selected_directories"]
    column_conventions: Dict[str, str] = params["column_conventions"]
    vegetation_type_conventions: Dict[str, str]  = params["vegetation_type_conventions"]
    change_type_conventions: Dict[str, str] = params["change_type_conventions"]
    excluded_cols: List[str] = params["excluded_cols"]


    for directory in selected_directories:
        filename_prefix = directory_file_prefix_map[directory]
        change_type = filename_prefix.lower()

        ndvi_filename = filename_prefix + "_TimeSerie_ndvi.csv"
        aux_filename = filename_prefix + "_AuxiliarFix.csv"

        ndvi_filepath = paths.data_raw_dir(directory, ndvi_filename)

        aux_filepath = paths.data_raw_dir(directory, aux_filename)

        aux_df = pd.read_csv(aux_filepath, index_col=0)
        ndvi_df = pd.read_csv(ndvi_filepath, index_col=0)

        df = pd.merge(aux_df, ndvi_df, left_on='IDpix', right_on='IDpix', how='outer')
        df = df.drop(columns=excluded_cols)
        df = df.set_index(["ID", "IDpix"])

        df = df.rename(columns=column_conventions)
        df["vegetation_type"] = df["vegetation_type"].map(vegetation_type_conventions)
        df["change_type"] = df["change_type"].map(change_type_conventions)

        out_directory = change_type_conventions[change_type]
        out_filename = change_type_conventions[change_type] + "_ndvi.csv"
        out_path = paths.data_interim_dir(out_directory, out_filename)

        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()

        df.to_csv(out_path)

def esn_fault_detection_partition(
        params_path: Path = paths.config_dir("params.yaml"),
        esn_metadata_path: Path = paths.data_interim_dir("esn", "esn_metadata.csv"),
        esn_dataset_path: Path = paths.data_interim_dir("esn", "esn_dataset.csv"),
        fault_detection_metadata_path: Path = paths.data_processed_dir("fault_detection", "fault_detection_metadata.csv"),
        fault_detection_dataset_path: Path = paths.data_processed_dir("fault_detection", "fault_detection_dataset.csv"),
        ) -> None:
    
    # Ensure paths existence
    out_paths = [
        esn_metadata_path,
        esn_dataset_path,
        fault_detection_metadata_path,
        fault_detection_dataset_path,
    ]

    for out_path in out_paths:
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    # Parameters
    esn_training_percentage: float = params["esn_training_percentage"]
    metadata_columns: List[str] = params["metadata_columns"]
    event_threshold_date: str = params["event_threshold"]
    random_seed: int = params["random_seed"]

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

    non_change_event_dates = pd.Series(non_change_fault_detection_signal_df.columns.max() +  pd.DateOffset(years=1), index=non_change_fault_detection_signal_df.index) # dummy event date for non_change signal
    non_change_fault_detection_metadata_df["change_start"] = non_change_event_dates

    event_threshold = pd.Timestamp(event_threshold_date)

    change_metadata_df.loc[:, "change_start"] = pd.to_datetime(change_metadata_df["change_start"], dayfirst=True)
    selected_change_polygons_indices = change_metadata_df.loc[change_metadata_df["change_start"] > event_threshold].index
    change_metadata_df = change_metadata_df.loc[selected_change_polygons_indices]
        
    change_signal_df = change_signal_df.loc[selected_change_polygons_indices]

    fault_detection_metadata_df = pd.concat((non_change_fault_detection_metadata_df, change_metadata_df))

    fault_detection_signal_df = pd.concat((non_change_fault_detection_signal_df, change_signal_df))
    fault_detection_signal_df = fault_detection_signal_df.reindex(sorted(fault_detection_signal_df.columns), axis=1)

    # Dataframes saving
    esn_metadata_df.to_csv(esn_metadata_path)
    esn_signal_df.to_csv(esn_dataset_path)

    fault_detection_metadata_df.to_csv(fault_detection_metadata_path)
    fault_detection_signal_df.to_csv(fault_detection_dataset_path)

def main() -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # logger.info("Merging NDVI files")
    # join_ndvi_time_series()

    logger.info("Creating ESN and fault detection datasets")
    esn_fault_detection_partition()

    logger.info("Done")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
