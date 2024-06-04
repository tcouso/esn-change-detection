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

    df = pd.read_csv(paths.data_interim_dir(
        dataset_filename), index_col=["ID", "IDpix"])

    # Signal and metadata datsets
    metadata_df = df.loc[:, metadata_columns]

    signal_columns = list(
        filter(lambda col: col not in metadata_columns, (col for col in df.columns)))
    signal_df = df[signal_columns]
    signal_df.columns = pd.to_datetime(signal_df.columns)
    signal_df = signal_df.reindex(sorted(signal_df.columns), axis=1)

    # Save files
    signal_filename = "_".join(["signal", dataset_filename])
    metadata_filename = "_".join(["metadata", dataset_filename])

    # Metadata goes to processed dir, since no more processing is needed
    metadata_df.to_csv(paths.data_processed_dir(metadata_filename))
    signal_df.to_csv(paths.data_interim_dir(signal_filename))
