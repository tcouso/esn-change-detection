from src.config import get_logger
from src.metrics.scores_over_parameters import scores_over_parameters
from src.metrics.optimal_model_scores import (optimal_model_scores,
                                              optimal_model_scores_by_event_type)


def main():
    logger = get_logger()

    logger.info("Computing scores over all parameters")
    scores_over_parameters()

    logger.info("Computing scores over optimal parameters")
    optimal_model_scores()

    logger.info("Computing scores by eventy type")
    optimal_model_scores_by_event_type()

    logger.info("Done")


if __name__ == "__main__":
    main()
