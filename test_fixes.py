#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to validate the fixes made to the API endpoints.
"""

import requests
import json
import sys
from datetime import datetime

# API URL
API_URL = "http://localhost:8000"

def get_token():
    """Get authentication token from API."""
    response = requests.post(
        f"{API_URL}/token",
        data={"username": "johndoe", "password": "secret"}
    )
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data["access_token"]
    else:
        print(f"Failed to get token: {response.status_code} - {response.text}")
        sys.exit(1)

def test_predict_single():
    """Test the predict_single endpoint with the fixed type checking."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test prediction with date parameter
    response = requests.get(
        f"{API_URL}/predict_single",
        params={
            "store_nbr": 1,
            "family": "PRODUCE",
            "onpromotion": False,
            "date": "2025-05-09"
        },
        headers=headers
    )
    
    print("\n1. Testing predict_single endpoint (fixes datetime type checking):")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success! Prediction: {result['prediction']:.2f}")
        print(f"✓ Saved to DB: {result['saved_to_db']}")
        if not result.get("is_fallback", False):
            print("✓ Successfully used model for prediction (not fallback)")
        else:
            print("✗ Used fallback prediction, model error still exists")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")

def test_feature_count():
    """Test the feature count returned by the API."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get feature names
    response = requests.get(
        f"{API_URL}/diagnostics",
        headers=headers
    )
    
    print("\n2. Testing feature count (should be exactly 81):")
    if response.status_code == 200:
        features = response.json().get("feature_names", [])
        print(f"✓ Feature count: {len(features)}")
        if len(features) == 81:
            print("✓ Correct feature count (81)")
        else:
            print(f"✗ Wrong feature count: expected 81, got {len(features)}")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")

def test_metrics_accuracy():
    """Test the metrics_accuracy_check endpoint."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test metrics accuracy check
    response = requests.get(
        f"{API_URL}/metrics_accuracy_check",
        headers=headers
    )
    
    print("\n3. Testing metrics_accuracy_check endpoint:")
    if response.status_code == 200:
        result = response.json()
        summary = result.get("summary", {})
        print(f"✓ Success! Count: {summary.get('count')}")
        print(f"✓ MAPE: {summary.get('mape', 0):.2f}%")
        print(f"✓ Forecast Accuracy: {summary.get('forecast_accuracy', 0):.2f}%")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")

def test_metrics_summary():
    """Test the metrics_summary endpoint with the ModelMetric fixes."""
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test metrics summary
    response = requests.get(
        f"{API_URL}/metrics_summary",
        headers=headers
    )
    
    print("\n4. Testing metrics_summary endpoint (fixes ModelMetric errors):")
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Total stores: {result.get('total_stores')}")
        print(f"✓ Total families: {result.get('total_families')}")
        print(f"✓ Avg sales: ${result.get('avg_sales'):.2f}")
        print(f"✓ Forecast accuracy: {result.get('forecast_accuracy'):.2f}%")
    else:
        print(f"✗ Failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("Testing fixed API endpoints...")
    print(f"API URL: {API_URL}")
    print(f"Current time: {datetime.now()}")
    
    # Run tests
    test_predict_single()
    test_feature_count()
    test_metrics_accuracy()
    test_metrics_summary()
    
    print("\nAll tests completed!") 