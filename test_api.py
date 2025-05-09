#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json

def print_json(data):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=2))

def get_token(base_url):
    """Get an authentication token."""
    try:
        auth_data = {
            "username": "johndoe",
            "password": "secret"
        }
        response = requests.post(f"{base_url}/token", data=auth_data)
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get("access_token")
        else:
            print(f"Authentication failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error obtaining token: {e}")
        return None

def main():
    print("Testing the Sales Forecasting API...")
    print("-----------------------------------------")
    
    base_url = "http://localhost:8000"
    
    # Get token
    token = get_token(base_url)
    headers = {}
    if token:
        print(f"Token obtained: {token[:10]}...")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print("Token not obtained, trying requests without authentication...")
    
    # 1. Health check test
    print("\n1. Health check test:")
    response = requests.get(f"{base_url}/health")
    health_data = response.json()
    print_json(health_data)
    print("-----------------------------------------")
    
    # 2. Individual prediction test
    print("\n2. Individual prediction test:")
    prediction_params = {
        "store_nbr": 1,
        "family": "PRODUCE",
        "onpromotion": False,
        "date": "2025-05-10"
    }
    response = requests.get(f"{base_url}/predict_single", params=prediction_params, headers=headers)
    try:
        prediction_data = response.json()
        print_json(prediction_data)
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error processing response: {e}")
        print(f"Response: {response.text}")
    print("-----------------------------------------")
    
    # 3. Explanation test
    print("\n3. Explanation test:")
    prediction_id = "1-PRODUCE-2025-05-10"
    explanation_params = {
        "store_nbr": 1,
        "family": "PRODUCE",
        "onpromotion": False,
        "date": "2025-05-10"
    }
    response = requests.get(f"{base_url}/explain/{prediction_id}", params=explanation_params, headers=headers)
    try:
        explanation_data = response.json()
        print_json(explanation_data)
        print(f"Status: {response.status_code}")
        
        # Check if explanation contains feature_contributions
        if "feature_contributions" in explanation_data:
            print(f"\nNumber of features explained: {len(explanation_data['feature_contributions'])}")
            print("\nTop contributions:")
            top_features = explanation_data["feature_contributions"][:5]
            for feature in top_features:
                print(f"  - {feature['feature']}: {feature['contribution']:.4f}")
    except Exception as e:
        print(f"Error processing response: {e}")
        print(f"Response: {response.text}")
    print("-----------------------------------------")
    
    print("Tests completed.")

if __name__ == "__main__":
    main() 