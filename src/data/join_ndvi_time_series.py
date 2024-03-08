import pandas as pd
import yaml
from pathlib import Path
from src import paths
from typing import List, Dict
from src.data.utils import create_output_paths

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
        
        create_output_paths([out_path])

        df.to_csv(out_path)

