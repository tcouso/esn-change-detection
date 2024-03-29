from src.models.train_esn import train_esn
from src.config import get_logger
from src.notifications import send_telegram_notification


def main():
    logger = get_logger()

    try:
        message = "Starting ESN training"
        logger.info(message)
        send_telegram_notification(message)

        logger.info("Training ESN")
        train_esn()

        message = "ESN training completed successfully"
        logger.info(message)
        send_telegram_notification(message)

    except Exception as e:
        message = f"ESN training failed: {e}"
        logger.error(message)
        send_telegram_notification(message)


if __name__ == "__main__":
    main()
