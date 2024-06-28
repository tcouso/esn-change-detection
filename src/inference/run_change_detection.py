import yaml
import pickle

from src import paths
from src.inference.pandas_wrapper import PandasWrapper
from src.inference.change_detection import parallel_detect_changes_in_pixels


def run_change_detection():

    with open(paths.config_dir("params.yaml"), "r") as file:
        params = yaml.safe_load(file)

        change_detection_forecasted_steps = params["change_detection_forecasted_steps"]
        N_values = params["N_values"]
        k_values = params["k_values"]
        offset_values = params["offset_values"]
        change_detection_num_processes = params["change_detection_num_processes"]

        with open(paths.results_simulations_dir("pixel_simulations.pk"), "rb") as file:
            results_simulations = pickle.load(file)

        simulations_df = PandasWrapper(
            results_simulations, object_name="simulation")

        results = parallel_detect_changes_in_pixels(simulations_df,
                                                    change_detection_forecasted_steps=change_detection_forecasted_steps,
                                                    N_values=N_values,
                                                    k_values=k_values,
                                                    offset_values=offset_values,
                                                    num_processes=change_detection_num_processes)

        change_detections_by_params = {}

        for result in results:
            params_index = (result.N, result.k)
            pixel_index = result.index

            if params_index not in change_detections_by_params:
                change_detections_by_params[params_index] = {}

            change_detections_by_params[params_index][pixel_index] = result

        with open(paths.results_change_detections_dir("change_detections.pk"), "wb") as file:
            pickle.dump(change_detections_by_params, file)
