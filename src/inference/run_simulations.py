import pickle
import yaml
import pandas as pd

from src import paths
from src.inference.pixel_simulation import parallel_simulate_pixels


def run_simulations() -> None:
    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

    change_detection_forecasted_steps: int = params["change_detection_forecasted_steps"]
    esn_features_dim: int = params["esn_features_dim"]
    change_detection_num_processes: int = params["change_detection_num_processes"]

    signal_df = pd.read_csv(paths.data_processed_dir(
        "test_signal_filtered_dataset_ndvi.csv"), index_col=["ID", "IDpix"])
    signal_df.columns = pd.to_datetime(signal_df.columns)

    results = parallel_simulate_pixels(
        model_path=paths.models_dir("trained_esn_ndvi.pk"),
        signal_df=signal_df,
        num_features=esn_features_dim,
        forecasted_steps=change_detection_forecasted_steps,
        num_processes=change_detection_num_processes
    )

    indexed_results = dict(
        (result.index, result) for result in results
    )

    with open(paths.results_simulations_dir("pixel_simulations.pk"), "wb") as file:
        pickle.dump(indexed_results, file)


if __name__ == "__main__":
    run_simulations()
