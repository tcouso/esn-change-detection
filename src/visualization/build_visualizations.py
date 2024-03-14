from src.config import get_logger
from src.visualization.plot_scores_over_parameters import plot_scores_over_parameters
from src.visualization.plot_optimal_model_scores import (
    plot_optimal_model_scores,
    plot_optimal_model_scores_by_event_type
)


def main():
    logger = get_logger()

    logger.info("Plotting scores over parameters")
    plot_scores_over_parameters()

    logger.info("Plotting optimal model scores")
    plot_optimal_model_scores()

    logger.info("Plotting optimal model scores by event type")
    plot_optimal_model_scores_by_event_type()

    logger.info("Done")


if __name__ == "__main__":
    main()
