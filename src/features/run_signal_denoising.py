import pandas as pd
from pathlib import Path
import yaml

from src.features.denoise_ndvi_signal import denoise_signal, downsample_time_series, moving_std_filter, holt_winters_filter
from src import paths
from src.data.utils import create_output_path


def run_signal_denoising(
    esn_dir: Path = paths.data_interim_dir("esn"),
    denoised_esn_dir: Path = paths.data_processed_dir("esn"),
    params_path: Path = paths.config_dir("params.yaml"),
) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]

    esn_metadata_filename = "esn_metadata_" + selected_band + ".csv"
    esn_signal_filename = "esn_signal_" + selected_band + ".csv"

    esn_metadata_path = esn_dir / esn_metadata_filename
    esn_signal_path = esn_dir / esn_signal_filename

    esn_metadata_df = pd.read_csv(esn_metadata_path, index_col=["ID", "IDpix"])
    esn_signal_df = pd.read_csv(esn_signal_path, index_col=["ID", "IDpix"])
    esn_signal_df.columns = pd.to_datetime(esn_signal_df.columns)

    denoised_esn_signal = []
    num_time_series = esn_signal_df.shape[0]
    max_start_date = pd.Timestamp.min
    min_end_date = pd.Timestamp.max

    for pol_ts_idx in range(num_time_series):
        # Denoise polygon
        denoised_ts = denoise_signal(
            esn_signal_df.iloc[pol_ts_idx], [downsample_time_series,
                                             moving_std_filter,
                                             holt_winters_filter]
        )
        denoised_esn_signal.append(denoised_ts)

        # Identify max start date
        if denoised_ts.index.min() > max_start_date:
            max_start_date = denoised_ts.index.min()

        # Identify min end date
        if denoised_ts.index.max() < min_end_date:
            min_end_date = denoised_ts.index.max()

    for pol_ts_idx in range(num_time_series):
        # Adjust polygon dates for equal size
        denoised_esn_signal[pol_ts_idx] = denoised_esn_signal[pol_ts_idx][max_start_date:]
        denoised_esn_signal[pol_ts_idx] = denoised_esn_signal[pol_ts_idx][:min_end_date]

    denoised_esn_signal_df = pd.DataFrame(denoised_esn_signal)
    denoised_esn_signal_df.index = esn_metadata_df.index

    denoised_esn_metadata_filename = "esn_metadata_denoised_" + selected_band + ".csv"
    denoised_esn_signal_filename = "esn_signal_denoised_" + selected_band + ".csv"

    denoised_metadata_path = denoised_esn_dir / denoised_esn_metadata_filename
    denoised_signal_path = denoised_esn_dir / denoised_esn_signal_filename

    out_paths = [
        denoised_signal_path,
        denoised_metadata_path,
    ]

    for out_path in out_paths:
        create_output_path(out_path)

    esn_metadata_df.to_csv(denoised_metadata_path)
    denoised_esn_signal_df.to_csv(denoised_signal_path)
