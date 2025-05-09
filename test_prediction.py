#!/usr/bin/env python3

import requests
import json

# API URL and parameters
API_URL = "http://localhost:8000"
STORE_NBR = 1
FAMILY = "PRODUCE"
DATE = "2025-05-09"

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

# Test predict_single endpoint
print("\nTesting predict_single endpoint...")
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(
    f"{API_URL}/predict_single",
    params={
        "store_nbr": STORE_NBR,
        "family": FAMILY,
        "onpromotion": False,
        "date": DATE
    },
    headers=headers
)

if response.status_code == 200:
    result = response.json()
    print(f"Success! Status code: {response.status_code}")
    print(f"Prediction: {result.get('prediction', 'N/A')}")
    print(f"Saved to DB: {result.get('saved_to_db', 'N/A')}")
    print(f"Is fallback: {result.get('is_fallback', 'N/A')}")
    print(f"Message: {result.get('message', 'N/A')}")
else:
    print(f"Failed: {response.status_code}")
    print(response.text) 