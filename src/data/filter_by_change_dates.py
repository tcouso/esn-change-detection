from src import paths
import pandas as pd
import yaml


def make_train_and_test_datasets() -> None:

    # Read files
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    # training_limit_date = params["training_limit_date"]

    test_signal_filename = "_".join(
        ["train", "signal", "filtered", "dataset", selected_band])
    test_signal_filename += ".csv"

    test_signal_df = pd.read_csv(paths.data_interim_dir(
        test_signal_filename), index_col=["ID", "IDpix"])
    test_signal_df.columns = pd.to_datetime(test_signal_df.columns)

    # Filter events out of detection range
    test_signal_df = test_signal_df.loc[
        test_signal_df["l"]
    ]

    # Files saving
    test_signal_df.to_csv(paths.data_processed_dir(test_signal_filename))
