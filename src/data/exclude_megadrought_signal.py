import pandas as pd
import yaml
from pathlib import Path
from src import paths
from src.data.utils import create_output_paths

def exclude_megadrought_signal(
    fault_detection_dataset_path: Path = paths.data_interim_dir("fault_detection", "fault_detection_dataset.csv"),
    fault_detection_metadata_path: Path = paths.data_interim_dir("fault_detection", "fault_detection_metadata.csv"),
    pre_megadrought_fault_detection_dataset_path: Path = paths.data_processed_dir("fault_detection", "pre_megadrought_fault_detection_dataset.csv"),
    pre_megadrought_fault_detection_metadata_path: Path = paths.data_processed_dir("fault_detection", "pre_megadrought_fault_detection_metadata.csv"),
    params_path: Path = paths.config_dir("params.yaml"),
) -> None:
  
  with open(params_path, "r") as file:
    params = yaml.safe_load(file)

  megadrought_threshold = params["megadrought_threshold"]
  non_change_placeholder_date = params["non_change_placeholder_date"]


  fault_detection_df = pd.read_csv(fault_detection_dataset_path, index_col=["ID", "IDpix"])
  fault_detection_df.columns = pd.to_datetime(fault_detection_df.columns ) 

  fault_detection_metadata_df = pd.read_csv(fault_detection_metadata_path, index_col=["ID", "IDpix"])
  fault_detection_metadata_df["change_start"] = pd.to_datetime(fault_detection_metadata_df["change_start"])

  megadrought_date = pd.Timestamp(megadrought_threshold)

  pre_megadrought_fault_detection_df = fault_detection_df.loc[:, fault_detection_df.columns < megadrought_date]

  labels_to_update = fault_detection_metadata_df[(fault_detection_metadata_df["change_start"] > megadrought_date) & (fault_detection_metadata_df["label"] == 1)].index

  pre_megadrought_fault_detection_metadata_df = fault_detection_metadata_df.copy()

  pre_megadrought_fault_detection_metadata_df.loc[labels_to_update, "label"] = 0
  pre_megadrought_fault_detection_metadata_df.loc[labels_to_update, "change_type"] = "stable"
  pre_megadrought_fault_detection_metadata_df.loc[labels_to_update, "change_start"] = pd.Timestamp(non_change_placeholder_date)


  out_paths = [
      pre_megadrought_fault_detection_dataset_path,
      pre_megadrought_fault_detection_metadata_path,
  ]

  create_output_paths(out_paths)

  pre_megadrought_fault_detection_df.to_csv(pre_megadrought_fault_detection_dataset_path)
  pre_megadrought_fault_detection_metadata_df.to_csv(pre_megadrought_fault_detection_metadata_path)
