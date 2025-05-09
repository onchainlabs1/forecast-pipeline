#!/usr/bin/env python3

import requests
import json

# API URL
API_URL = "http://localhost:8000"

# Get token first
print("Getting token...")
response = requests.post(
    f"{API_URL}/token",
    data={"username": "johndoe", "password": "secret"}
)

if response.status_code != 200:
    print(f"Failed to get token: {response.status_code} - {response.text}")
    exit(1)

token = response.json()["access_token"]
print(f"Token obtained successfully")

# Set headers with authorization
headers = {"Authorization": f"Bearer {token}"}

# Test 1: metrics_summary endpoint
print("\n1. Testing metrics_summary endpoint...")
response = requests.get(
    f"{API_URL}/metrics_summary",
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"Success! Status code: {response.status_code}")
    print(f"Total stores: {result.get('total_stores', 'N/A')}")
    print(f"Total families: {result.get('total_families', 'N/A')}")
    print(f"Avg sales: ${result.get('avg_sales', 'N/A'):.2f}")
    print(f"Forecast accuracy: {result.get('forecast_accuracy', 'N/A'):.2f}%")
    print(f"Is mock data: {result.get('is_mock_data', 'N/A')}")
else:
    print(f"Failed: {response.status_code}")
    print(response.text)

# Test 2: metrics_accuracy_check endpoint
print("\n2. Testing metrics_accuracy_check endpoint...")
response = requests.get(
    f"{API_URL}/metrics_accuracy_check",
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    summary = result.get("summary", {})
    print(f"Success! Status code: {response.status_code}")
    print(f"Data points: {summary.get('count', 'N/A')}")
    print(f"MAPE: {summary.get('mape', 'N/A'):.2f}%")
    print(f"Forecast accuracy: {summary.get('forecast_accuracy', 'N/A'):.2f}%")
    print(f"MAE: {summary.get('mae', 'N/A'):.2f}")
    print(f"RMSE: {summary.get('rmse', 'N/A'):.2f}")
    print(f"Mean error: {summary.get('mean_error', 'N/A'):.2f}")
    print(f"Calculation method: {summary.get('calculation_method', 'N/A')}")
    
    # Print first few data points for verification
    detailed_results = result.get("detailed_results", [])
    if detailed_results:
        print("\nSample data points:")
        for i, point in enumerate(detailed_results[:3]):
            print(f"  {i+1}. Store: {point.get('store')}, Family: {point.get('family')}")
            print(f"     Predicted: {point.get('predicted'):.2f}, Actual: {point.get('actual'):.2f}")
            print(f"     Error: {point.get('error'):.2f}, Percentage error: {point.get('percentage_error'):.2f}%")
else:
    print(f"Failed: {response.status_code}")
    print(response.text) 