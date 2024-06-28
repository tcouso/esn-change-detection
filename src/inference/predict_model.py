from src.config import get_logger
from src.inference.run_simulations import run_simulations
from src.inference.run_change_detection import run_change_detection
# from src.change_detection.predict_over_pixels import predict_over_pixels
# from src.change_detection.predict_over_polygons import predict_over_polygons
from src.notifications import send_telegram_notification


def main():
    logger = get_logger()

    try:
        message = "Starting model prediction"
        logger.info(message)
        send_telegram_notification(message)

        message = "Starting signal simulations"
        logger.info(message)
        send_telegram_notification(message)
        run_simulations()

        message = "Starting change detection"
        logger.info(message)
        send_telegram_notification(message)
        run_change_detection()

        # message = "Starting pixel predictions"
        # logger.info(message)
        # send_telegram_notification(message)
        # predict_over_pixels()

        # message = "Starting polygon predictions"
        # logger.info(message)
        # send_telegram_notification(message)
        # predict_over_polygons()

        message = "Model prediction done"
        logger.info(message)
        send_telegram_notification(message)

    except KeyboardInterrupt:
        message = "Execution interrupted by keyboard"
        logger.info(message)
        send_telegram_notification(message)

    except Exception as e:
        message = f"Execution crashed: {e}"
        logger.error(message)
        send_telegram_notification(message)


if __name__ == '__main__':
    main()
