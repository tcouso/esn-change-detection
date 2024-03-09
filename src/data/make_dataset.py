from src.config import get_logger
from src.data.join_ndvi_time_series import join_ndvi_time_series
from src.data.esn_fault_detection_partition import esn_fault_detection_partition
from src.data.exclude_megadrought_signal import exclude_megadrought_signal

def main() -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = get_logger()

    logger.info("Merging NDVI files")
    join_ndvi_time_series()

    logger.info("Creating ESN and fault detection datasets")
    esn_fault_detection_partition()

    logger.info("Excluding signal previous to megadrought")
    exclude_megadrought_signal()

    logger.info("Done")


if __name__ == '__main__':
    main()
