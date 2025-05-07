#!/bin/bash

# Simplified script to run the pipeline

echo "Starting simplified pipeline..."

# 1. Install necessary dependencies
echo "Installing dependencies..."
pip3 install pandas numpy scikit-learn lightgbm matplotlib seaborn prophet statsmodels jupyter mlflow 

# 2. Data preprocessing
echo "Preprocessing data..."
python3 src/data/preprocess.py

# 3. Model training
echo "Training model..."
python3 src/train_model.py

echo "Pipeline complete!" 