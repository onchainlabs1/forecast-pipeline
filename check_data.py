#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to check if the test data is available in the API.
"""

import requests
import json

# API URL
API_URL = "http://localhost:8000"

# Get token
print("Getting token...")
response = requests.post(
    f"{API_URL}/token",
    data={"username": "johndoe", "password": "secret"}
)

if response.status_code == 200:
    token_data = response.json()
    token = token_data["access_token"]
    print("Token obtained successfully.")
    
    # Check metrics accuracy
    print("\nChecking metrics accuracy...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/metrics_accuracy_check", headers=headers)
    
    if response.status_code == 200:
        try:
            data = response.json()
            summary = data.get("summary", {})
            
            print(f"Data points: {summary.get('count', 0)}")
            print(f"MAPE: {summary.get('mape', 0):.2f}%")
            print(f"Forecast Accuracy: {summary.get('forecast_accuracy', 0):.2f}%")
            
            print("\nAPI is working correctly!")
            
            # Check if we have the expected accuracy
            if summary.get("forecast_accuracy", 0) >= 75.0:
                print("\nSUCCESS: The test data is showing in the API with the expected accuracy.")
            else:
                print("\nWARNING: The accuracy is not at the expected level (should be around 80%).")
        except json.JSONDecodeError:
            print("Error decoding JSON response.")
    else:
        print(f"Error: {response.status_code} - {response.text}")
else:
    print(f"Error getting token: {response.status_code} - {response.text}") 