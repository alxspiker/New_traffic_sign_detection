#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Train on full MTSD dataset
# python3 train_traffic_sign_model.py --data-dir data_extracted --epochs 100

# Train focused speed limit model
python3 train_traffic_sign_model.py --speed-limit-only --force-cpu --data-dir speed_signs_extracted

# Custom training with GPU
# python3 train_traffic_sign_model.py --data-dir data_extracted --epochs 200 --batch-size 16 --model-size m
