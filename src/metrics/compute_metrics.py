from src.config import get_logger
from src.metrics.scores_over_parameters import scores_over_parameters


def main():
    logger = get_logger()

    logger.info("Computing scores over all parameters")
    scores_over_parameters()

    logger.info("Done")


if __name__ == "__main__":
    main()
