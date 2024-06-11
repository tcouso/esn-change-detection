import yaml
from src import paths
import pandas as pd


def make_signal_and_metadata_datasets() -> None:

    # Read files
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    metadata_columns = params["metadata_columns"]

    dataset_filename = "_".join(["filtered", "dataset", selected_band])
    dataset_filename += ".csv"

    df = pd.read_csv(paths.data_interim_dir(dataset_filename), index_col=["ID", "IDpix"])

    # Signal and metadata datsets
    metadata_df = df.loc[:, metadata_columns]
    metadata_df["change_start_date"] = pd.to_datetime(metadata_df["change_start_date"], dayfirst=True)
    metadata_df["last_non_change_date"] = pd.to_datetime(metadata_df["last_non_change_date"], dayfirst=True)
    metadata_df["change_ending_date"] = pd.to_datetime(metadata_df["change_ending_date"], dayfirst=True)

    # Assign labels according to polygon categories
    # metadata_df["label"] = 0
    negative_class_indices = metadata_df[(metadata_df["change_type"] == "stable") | (metadata_df["change_type"] == "drought")].index
    positive_class_indices = metadata_df[(metadata_df["change_type"] == "fire") | (metadata_df["change_type"] == "logging")].index

    metadata_df.loc[negative_class_indices, "label"] = 0
    metadata_df.loc[positive_class_indices, "label"] = 1

    signal_columns = list(filter(lambda col: col not in metadata_columns, (col for col in df.columns)))
    signal_df = df[signal_columns]
    signal_df.columns = pd.to_datetime(signal_df.columns)
    signal_df = signal_df.reindex(sorted(signal_df.columns), axis=1)

    # Save files
    signal_filename = "_".join(["signal", dataset_filename])
    metadata_filename = "_".join(["metadata", dataset_filename])

    # Metadata goes to processed dir, since no more processing is needed
    metadata_df.to_csv(paths.data_processed_dir(metadata_filename))
    signal_df.to_csv(paths.data_interim_dir(signal_filename))
