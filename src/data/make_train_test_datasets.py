from src import paths
import pandas as pd
import yaml


def make_train_and_test_datasets() -> None:

    # Read files
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band = params["selected_band"]
    training_limit_date = params["training_limit_date"]

    signal_filename = "_".join(
        ["signal", "filtered", "dataset", selected_band])
    signal_filename += ".csv"

    signal_df = pd.read_csv(paths.data_interim_dir(
        signal_filename), index_col=["ID", "IDpix"])
    signal_df.columns = pd.to_datetime(signal_df.columns)

    # Dataset train-test partition
    train_signal_df = signal_df.loc[:, signal_df.columns < pd.Timestamp(
        training_limit_date)]
    test_signal_df = signal_df.loc[:, signal_df.columns >= pd.Timestamp(
        training_limit_date)]

    # Files saving
    train_signal_filename = "_".join(["train", signal_filename])
    test_signal_filename = "_".join(["test", signal_filename])

    train_signal_df.to_csv(paths.data_processed_dir(train_signal_filename))
    test_signal_df.to_csv(paths.data_processed_dir(test_signal_filename))
