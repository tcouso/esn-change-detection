import pandas as pd
import yaml
from pathlib import Path
from src import paths
from typing import List, Dict
from src.data.utils import create_output_path


def join_signal_to_aux_datasets() -> None:

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    directory_file_prefix_map: Dict[str,
                                    str] = params["directory_file_prefix_map"]
    selected_directories: List[str] = params["selected_directories"]
    column_conventions: Dict[str, str] = params["column_conventions"]
    vegetation_type_conventions: Dict[str,
                                      str] = params["vegetation_type_conventions"]
    change_type_conventions: Dict[str, str] = params["change_type_conventions"]
    excluded_cols: List[str] = params["excluded_cols"]
    test_at_scale: bool = params["test_at_scale"]
    test_at_scale_sample_size: int = params["test_at_scale_sample_size"]

    for directory in selected_directories:
        filename_prefix = directory_file_prefix_map[directory]
        change_type = filename_prefix.lower()

        signal_filename = filename_prefix + "_TimeSerie_" + selected_band + ".csv"
        aux_filename = filename_prefix + "_AuxiliarFix.csv"

        signal_filepath = paths.data_raw_dir(directory, signal_filename)
        aux_filepath = paths.data_raw_dir(directory, aux_filename)

        aux_df = pd.read_csv(aux_filepath, index_col=0)
        signal_df = pd.read_csv(signal_filepath, index_col=0)

        df = pd.merge(aux_df, signal_df, left_on='IDpix',
                      right_on='IDpix', how='outer')
        df = df.drop(columns=excluded_cols)
        df = df.set_index(["ID", "IDpix"])

        df = df.rename(columns=column_conventions)
        df["vegetation_type"] = df["vegetation_type"].map(
            vegetation_type_conventions)
        df["change_type"] = df["change_type"].map(change_type_conventions)

        out_directory = change_type_conventions[change_type]
        out_filename = change_type_conventions[change_type] + \
            "_" + selected_band + ".csv"
        out_path = paths.data_interim_dir(out_directory, out_filename)

        create_output_path(out_path)

        if test_at_scale:
            df = df.sample(n=test_at_scale_sample_size)

        df.to_csv(out_path)
