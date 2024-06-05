from src.config import get_logger
from src.features.run_signal_denoising import run_signal_denoising
from src.notifications import send_telegram_notification


def main():
    logger = get_logger()

    try:
        message = "Starting training dataset denoising"
        logger.info(message)
        send_telegram_notification(message)

        run_signal_denoising()

        message = "Training dataset denoising completed successfully"
        logger.info(message)
        send_telegram_notification(message)

    except Exception as e:
        message = f"Training dataset denoising failed: {e}"
        logger.error(message)
        send_telegram_notification(message)


if __name__ == "__main__":
    main()
