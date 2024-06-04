from src.config import get_logger
from src.data.join_signal_to_aux_datasets import join_signal_to_aux_datasets
from src.data.join_events_datasets import join_events_datasets
from src.data.filter_dataset import filter_dataset
from src.data.make_signal_and_metadata_datasets import make_signal_and_metadata_datasets
from src.data.make_train_test_datasets import make_train_and_test_datasets
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

        logger.info("Joining aux and signal files")
        join_signal_to_aux_datasets()

        logger.info("Joining events datasets")
        join_events_datasets()

        logger.info("Filtering dataset")
        filter_dataset()

        logger.info("Making signal and metadata datasets")
        make_signal_and_metadata_datasets()

        logger.info("Creating training and testing datasets")
        make_train_and_test_datasets()

        message = "Data processing completed successfully"
        logger.info(message)
        send_telegram_notification(message)

    except Exception as e:
        message = f"Data processing failed: {e}"
        logger.error(message)
        send_telegram_notification(message)


if __name__ == '__main__':
    main()
