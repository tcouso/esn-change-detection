import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from src.models.run_simulations import run_simulations
from src.models.predict_over_pixels import predict_over_pixels
from src.models.predict_over_polygons import predict_over_polygons
from src.notifications import send_telegram_notification

def main():
    logger = logging.getLogger(__name__)

    message = "Starting signal simulations"
    logger.info(message)
    send_telegram_notification(message)
    run_simulations()

    message = "Starting pixel predictions"
    logger.info(message)
    send_telegram_notification(message)
    predict_over_pixels()

    message = "Starting polygon predictions"
    logger.info(message)
    send_telegram_notification(message)
    predict_over_polygons()

    message = "Done"
    logger.info(message)
    send_telegram_notification(message)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
