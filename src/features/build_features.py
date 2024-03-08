import logging
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from src.features.build_features import denoise_ndvi_signal, downsample_time_series, moving_std_filter, holt_winters_filter
from src import paths
from pathlib import Path


def main(
        signal_path: Path = paths.data_interim_dir("esn", "esn_dataset.csv"),
        metadata_path: Path = paths.data_interim_dir("esn", "esn_dataset.csv"),
        denoised_signal_path: Path = paths.data_processed_dir(
            "esn", "denoised_esn_signal.csv"),
        denoised_metadata_path: Path = paths.data_processed_dir(
        "esn", "denoised_esn_metadata.csv"),
) -> None:
    
    logger = logging.getLogger(__name__)

    logger.info("Starting esn dataset denoising")

    esn_signal_df = pd.read_csv(signal_path, index_col=["ID", "IDpix"])
    esn_signal_df.columns = pd.to_datetime(esn_signal_df.columns)

    esn_metadata_df = pd.read_csv(metadata_path, index_col=["ID", "IDpix"])

    denoised_esn_signal = []
    num_time_series = esn_signal_df.shape[0]
    max_start_date = pd.Timestamp.min
    min_end_date = pd.Timestamp.max

    for pol_ts_idx in range(num_time_series):
        # Denoise polygon
        denoised_ts = denoise_ndvi_signal(esn_signal_df.iloc[pol_ts_idx], [
                                          downsample_time_series, moving_std_filter, holt_winters_filter])
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

    out_paths = [
        denoised_signal_path,
        denoised_metadata_path,
    ]

    for out_path in out_paths:
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()

    esn_metadata_df.to_csv(denoised_metadata_path)
    denoised_esn_signal_df.to_csv(denoised_signal_path)

    logger.info("Done")


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()