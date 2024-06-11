from src.config import get_logger
from src.features.run_signal_denoising import run_signal_denoising
from src.features.run_signal_slicing import run_signal_slicing
from src.notifications import send_telegram_notification


def main():
    logger = get_logger()

    try:
        message = "Starting features building"
        logger.info(message)
        send_telegram_notification(message)
    
        message = "Starting training dataset denoising"
        logger.info(message)
        send_telegram_notification(message)

        run_signal_denoising()

        message = "Starting training dataset slicing"
        logger.info(message)
        send_telegram_notification(message)

        run_signal_slicing()

        message = "Feature building done"
        logger.info(message)
        send_telegram_notification(message)

    except Exception as e:
        message = f"Features building failed: {e}"
        logger.error(message)
        send_telegram_notification(message)


if __name__ == "__main__":
    main()
