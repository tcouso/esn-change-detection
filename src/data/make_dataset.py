from src.config import get_logger
from src.data.join_signal_to_aux_datasets import join_signal_to_aux_datasets
from src.data.esn_fault_detection_partition import esn_fault_detection_partition
from src.notifications import send_telegram_notification


def main() -> None:
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = get_logger()

    try:
        message = "Starting data processing"
        logger.info(message)
        send_telegram_notification(message)

        logger.info("Merging aux and signal files")
        join_signal_to_aux_datasets()

        logger.info("Creating ESN and fault detection datasets")
        esn_fault_detection_partition()

        message = "Data processing completed successfully"
        logger.info(message)
        send_telegram_notification(message)

    except Exception as e:
        message = f"Data processing failed: {e}"
        logger.error(message)
        send_telegram_notification(message)


if __name__ == '__main__':
    main()
