import pandas as pd
from pathlib import Path
import yaml

from src.features.denoise_signal import denoise_signal, downsample_time_series, moving_std_filter, holt_winters_filter
from src import paths
# from src.data.utils import create_output_path


def run_signal_denoising() -> None:
    
    # esn_dir: Path = paths.data_interim_dir("esn")
    # denoised_esn_dir: Path = paths.data_processed_dir("esn")
    params_path: Path = paths.config_dir("params.yaml")

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    # esn_metadata_filename = "esn_metadata_" + selected_band + ".csv"
    train_signal_filename = "_".join(["train_signal_filtered_dataset", selected_band])
    train_signal_filename += ".csv"

    # esn_metadata_path = esn_dir / esn_metadata_filename
    # esn_signal_path = esn_dir / esn_signal_filename

    # metadata_df = pd.read_csv(paths.data_processed_dir("metadata_filtered_dataset_ndvi.csv"), index_col=["ID", "IDpix"])
    signal_df = pd.read_csv(paths.data_processed_dir(train_signal_filename), index_col=["ID", "IDpix"])
    signal_df.columns = pd.to_datetime(signal_df.columns)

    denoised_train_signal = []
    num_pixels = signal_df.shape[0]
    max_start_date = pd.Timestamp.min
    min_end_date = pd.Timestamp.max

    for pix_index in range(num_pixels):
        # Denoise polygon
        denoised_ts = denoise_signal(
            signal_df.iloc[pix_index], [downsample_time_series,
                                             moving_std_filter,
                                             holt_winters_filter]
        )
        denoised_train_signal.append(denoised_ts)

        # Identify max start date
        if denoised_ts.index.min() > max_start_date:
            max_start_date = denoised_ts.index.min()

        # Identify min end date
        if denoised_ts.index.max() < min_end_date:
            min_end_date = denoised_ts.index.max()

    for pix_index in range(num_pixels):
        # Adjust polygon dates for equal size
        denoised_train_signal[pix_index] = denoised_train_signal[pix_index][max_start_date:]
        denoised_train_signal[pix_index] = denoised_train_signal[pix_index][:min_end_date]

    denoised_train_signal_df = pd.DataFrame(denoised_train_signal)
    denoised_train_signal_df.index = signal_df.index

    # denoised_esn_metadata_filename = "esn_metadata_denoised_" + selected_band + ".csv"
    # denoised_esn_signal_filename = "esn_signal_denoised_" + selected_band + ".csv"

    # denoised_metadata_path = denoised_esn_dir / denoised_esn_metadata_filename
    # denoised_signal_path = denoised_esn_dir / denoised_esn_signal_filename

    # out_paths = [
    #     denoised_signal_path,
    #     denoised_metadata_path,
    # ]

    # for out_path in out_paths:
    #     create_output_path(out_path)

    # metadata_df.to_csv(denoised_metadata_path)
    denoised_train_signal_filename = "_".join(["denoised", train_signal_filename])
    denoised_train_signal_df.to_csv(paths.features_dir(denoised_train_signal_filename))
