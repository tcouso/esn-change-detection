import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

from src import paths
from src.data.utils import create_output_path


def plot_scores_over_parameters(
    params_path=paths.config_dir("params.yaml"),
    metrics_dir: Path = paths.reports_dir("metrics"),
    plots_dir: Path = paths.reports_dir("figures"),
) -> None:

    with open(params_path, "r") as file:
        params = yaml.safe_load(file)

    selected_band: str = params["selected_band"]
    N_values: List[int] = params["N_values"]
    k_values: List[float] = params["k_values"]
    th_values: List[float] = params["voting_thresholds"]

    acc_scores_filename = "_".join(["acc_scores", selected_band])
    acc_scores_filename += ".npy"

    f1_scores_filename = "_".join(["f1_scores", selected_band])
    f1_scores_filename += ".npy"

    precision_scores_filename = "_".join(["precision_scores", selected_band])
    precision_scores_filename += ".npy"

    recall_scores_filename = "_".join(["recall_scores", selected_band])
    recall_scores_filename += ".npy"

    acc_scores_path = metrics_dir / acc_scores_filename
    f1_scores_path = metrics_dir / f1_scores_filename
    precision_scores_path = metrics_dir / precision_scores_filename
    recall_scores_path = metrics_dir / recall_scores_filename

    acc_scores = np.load(acc_scores_path)
    f1_scores = np.load(f1_scores_path)
    precision_scores = np.load(precision_scores_path)
    recall_scores = np.load(recall_scores_path)

    metrics_dict = {
        "acc_scores": acc_scores,
        "f1_scores": f1_scores,
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
    }

    for metrics_name, metrics in metrics_dict.items():

        overall_min = metrics.min()
        overall_max = metrics.max()

        plt.rcParams['font.size'] = 14

        n = len(th_values)

        plt.figure(figsize=(n * 10, 8))

        for th_index, th in enumerate(th_values):

            plt.subplot(1, n, th_index + 1)

            df = pd.DataFrame(metrics[th_index])
            df.columns = k_values
            df.index = N_values

            sns.heatmap(df, annot=True, cmap="YlGnBu", cbar_kws={
                        'label': 'F1-score'}, vmin=overall_min, vmax=overall_max)

            plt.title(f"Threshold={th}")
            plt.xlabel('k')
            plt.ylabel('N')

        plt.tight_layout()

        plot_filename = "_".join([selected_band, metrics_name, "over_params"])
        plot_filename += ".png"
        plot_path = plots_dir / plot_filename

        create_output_path(plot_path)

        plt.savefig(plot_path)
        plt.close()
