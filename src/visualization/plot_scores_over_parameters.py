import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np
from typing import List

from src import paths
from src.data.utils import create_output_path

def plot_scores_over_parameters(
  params_path = paths.config_dir("params.yaml"),
  metrics_dir: Path = paths.reports_dir("metrics"),
  plots_dir: Path = paths.reports_dir("figures"),
) -> None:
  
  with open(params_path, "r") as file:
      params = yaml.safe_load(file)

  selected_band: str = params["selected_band"]
  th_values: List[float] = params["voting_thresholds"]

  acc_scores_filename = "acc_scores_" + selected_band + ".npy"
  f1_scores_filename = "f1_scores_" + selected_band + ".npy"
  precision_scores_filename = "precision_scores_" + selected_band + ".npy"
  recall_scores_filename = "recall_scores_" + selected_band + ".npy"

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

          sns.heatmap(metrics[th_index], annot=True, cmap="YlGnBu", cbar_kws={
                      'label': 'F1-score'}, vmin=overall_min, vmax=overall_max)

          plt.title(f"Threshold={th}")
          plt.xlabel('k')
          plt.ylabel('N')

      plt.tight_layout()

      plot_filename = selected_band + "_" + metrics_name + "_scores_over_params.png"
      plot_path = plots_dir / plot_filename

      create_output_path(plot_path)

      plt.savefig(plot_path)
      plt.close()
