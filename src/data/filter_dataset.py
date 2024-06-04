import yaml
from src import paths
import pandas as pd


def filter_dataset() -> None:

    # Read files
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    dataset_filename = "_".join(["dataset", selected_band])
    dataset_filename += ".csv"

    df = pd.read_csv(paths.data_interim_dir(
        dataset_filename), index_col=["ID", "IDpix"])

    # Apply filters
    selected_vegetation = params["selected_vegetation"]

    df = df.loc[df["vegetation_type"] == selected_vegetation, :]

    # Save files
    filtered_dataset_filename = "_".join(["filtered", dataset_filename])
    df.to_csv(paths.data_interim_dir(filtered_dataset_filename))
