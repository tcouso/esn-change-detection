#!/bin/bash

cd "$(dirname "$0")"

# python3 src/data/make_dataset.py
# python3 src/features/build_features.py
# python3 src/training/train_model.py
python3 src/inference/predict_model.py
