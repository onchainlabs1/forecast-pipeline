#!/bin/bash

echo "Starting Sales Forecasting API tests"
echo "============================================="

# Prediction test
echo "Running prediction test..."
python test_prediction.py > prediction_results.txt 2>&1
echo "Results saved in prediction_results.txt"

# Metrics test
echo "Running metrics test..."
python test_metrics.py > metrics_results.txt 2>&1
echo "Results saved in metrics_results.txt"

echo "Tests completed."
echo "Check prediction_results.txt and metrics_results.txt files for results." 