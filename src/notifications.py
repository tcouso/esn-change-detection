import requests
import os
from dotenv import find_dotenv, load_dotenv
from src.config import get_logger

load_dotenv(find_dotenv())

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

logger = get_logger()


def send_telegram_notification(message):
    url = API_URL + "/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}

    try:

        response = requests.post(url, data=payload)
        response.raise_for_status()

    except requests.RequestException:

        logger.info("Failed to send Telegram notification")
