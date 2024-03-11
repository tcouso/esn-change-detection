#!/bin/bash

cd "$(dirname "$0")"

python3 src/data/make_dataset.py
python3 src/features/build_features.py
python3 src/models/train_model.py
python3 src/models/predict_model.py
