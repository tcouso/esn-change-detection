import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def get_season_dates(year: int) -> tuple:
    """
    Function to generate season date ranges for a given year.

    Parameters:
        year (int): The year for which to generate the season date ranges.

    Returns:
        tuple: A tuple containing the start and end dates for the spring & summer and autumn & winter seasons.
    """
    autumn_winter_start = mdates.datestr2num(f"{year}-03-22")
    autumn_winter_end = mdates.datestr2num(f"{year}-09-20") 
    spring_summer_start = mdates.datestr2num(f"{year}-09-22")  
    spring_summer_end = mdates.datestr2num(f"{year+1}-03-20")  

    return (
        spring_summer_start,
        spring_summer_end,
        autumn_winter_start,
        autumn_winter_end,
    )


def plot_pixel_signal(
    *pol_list: pd.Series,
    labels: list = None,
    start_date: pd.Timestamp = pd.Timestamp("2000-01-03"),
    end_date: pd.Timestamp = pd.Timestamp("2022-12-25"),
    size: tuple = (25, 5),
    show_dots: bool = False,
):
    """
    Function to generate a time series plot of multiple polygons with colored background according to season.

    Parameters:
        *pol_list (pd.Series): Variable number of pandas Series objects representing the time series data.
        labels (list, optional): List of labels for each time series (default: None).
        start_date (pd.Timestamp, optional): The start date for the plot (default: January 3, 2000).
        end_date (pd.Timestamp, optional): The end date for the plot (default: December 25, 2022).
        size (tuple, optional): The size of the figure (default: (25, 5)).
        show_dots (bool, optional): Whether to show dots for each data point (default: False).

    Returns:
        None
    """
    _, ax = plt.subplots()

    for i, pol in enumerate(pol_list):
        pol = pol[(pol.index >= start_date) & (pol.index <= end_date)]

        if show_dots:
            ax.scatter(pol.index, pol)

        if labels:
            ax.plot(pol, label=labels[i])
        else:
            ax.plot(pol)

    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.gcf().set_size_inches(*size)

    start_year = pol.index.min().year
    end_year = pol.index.max().year

    for year in range(start_year, end_year + 1):
        (
            spring_summer_start,
            spring_summer_end,
            autumn_winter_start,
            autumn_winter_end,
        ) = get_season_dates(year)

        ax.axvspan(spring_summer_start, spring_summer_end, facecolor="red", alpha=0.3)
        ax.axvspan(autumn_winter_start, autumn_winter_end, facecolor="blue", alpha=0.3)

    ax.legend()

    ax.set_xlabel("Date")
    ax.set_ylabel("NDVI")

    plt.xticks(rotation=45)
    plt.show()
