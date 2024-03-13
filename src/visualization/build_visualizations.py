from src.config import get_logger
from src.visualization.plot_scores_over_parameters import plot_scores_over_parameters


def main():
    logger = get_logger()

    logger.info("Plotting scores over parameters")
    plot_scores_over_parameters()

    logger.info("Done")


if __name__ == "__main__":
    main()
