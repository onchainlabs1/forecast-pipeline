#!/bin/bash

# Run the complete pipeline for store sales forecasting

echo "Starting Store Sales Forecasting Pipeline"
echo "----------------------------------------"

# Create necessary directories
mkdir -p data/raw data/processed models reports mlruns

# Step 1: Download the data
echo "Step 1: Downloading data from Kaggle"
python src/data/load_data.py

# Step 2: Preprocess the data
echo "Step 2: Preprocessing data"
python src/data/preprocess.py

# Step 3: Train the model
echo "Step 3: Training model"
python src/train_model.py

# Step 4: Start the API (optional)
if [ "$1" = "--with-api" ]; then
    echo "Step 4: Starting API"
    python src/api/main.py
fi

echo "Pipeline completed successfully!"
echo "Check the 'reports' directory for metrics and the 'models' directory for the trained model."