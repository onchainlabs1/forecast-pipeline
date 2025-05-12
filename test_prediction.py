#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the prediction API to ensure it handles all edge cases correctly.
"""

import os
import sys
import requests
import json
from datetime import datetime, timedelta
import pandas as pd

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Constants
API_URL = "http://localhost:8000"  # Local API URL
USERNAME = "admin"
PASSWORD = "admin"

def get_auth_token():
    """Get JWT token from API."""
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": USERNAME,
                "password": PASSWORD
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            print(f"Error getting token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception during authentication: {str(e)}")
        return None

def test_health():
    """Test API health endpoint."""
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {str(e)}")
        return False

def test_diagnostics(token):
    """Test API diagnostics endpoint."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_URL}/diagnostics", headers=headers)
        print(f"Diagnostics: {response.status_code}")
        
        if response.status_code == 200:
            diagnostics = response.json()
            print(f"API Status: {diagnostics.get('status')}")
            print(f"Model loaded: {diagnostics.get('model', {}).get('model_loaded')}")
            print(f"Features count: {diagnostics.get('test_prediction', {}).get('features_count')}")
            print(f"Can make real predictions: {diagnostics.get('can_make_real_predictions')}")
            return True
        else:
            print(f"Diagnostics failed: {response.text}")
            return False
    except Exception as e:
        print(f"Diagnostics exception: {str(e)}")
        return False

def test_predict_single(token, store_nbr, family, onpromotion, date_str):
    """Test single prediction with specific parameters."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        params = {
            "store_nbr": store_nbr,
            "family": family,
            "onpromotion": onpromotion,
            "date": date_str
        }
        
        print(f"\nTesting prediction for store {store_nbr}, family {family}, promo {onpromotion}, date {date_str}")
        response = requests.get(f"{API_URL}/predict_single", params=params, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result.get('prediction', result.get('predicted_sales', 'N/A'))}")
            print(f"Is fallback: {result.get('is_fallback', False)}")
            
            # Test explanation
            pred_id = result.get("prediction_id", f"{store_nbr}-{family}-{date_str}")
            exp_response = requests.get(
                f"{API_URL}/explain/{pred_id}",
                params=params,
                headers=headers
            )
            
            if exp_response.status_code == 200:
                explanation = exp_response.json()
                if "feature_contributions" in explanation:
                    print(f"Explanation features: {len(explanation['feature_contributions'])}")
                else:
                    print("No feature contributions in explanation")
            else:
                print(f"Explanation failed: {exp_response.status_code} - {exp_response.text}")
                
            return True
        else:
            print(f"Prediction failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Prediction exception: {str(e)}")
        return False

def test_edge_cases(token):
    """Test various edge cases for the prediction API."""
    # Test with all store numbers
    for store in [1, 25, 54]:  # Low, middle, high store numbers
        test_predict_single(token, store, "PRODUCE", False, datetime.now().strftime("%Y-%m-%d"))
    
    # Test with all families
    for family in ["PRODUCE", "AUTOMOTIVE", "SEAFOOD", "INVALID_FAMILY"]:
        test_predict_single(token, 1, family, False, datetime.now().strftime("%Y-%m-%d"))
    
    # Test with promotion
    test_predict_single(token, 1, "PRODUCE", True, datetime.now().strftime("%Y-%m-%d"))
    
    # Test with different dates
    today = datetime.now()
    test_predict_single(token, 1, "PRODUCE", False, today.strftime("%Y-%m-%d"))
    test_predict_single(token, 1, "PRODUCE", False, (today + timedelta(days=30)).strftime("%Y-%m-%d"))
    test_predict_single(token, 1, "PRODUCE", False, (today - timedelta(days=30)).strftime("%Y-%m-%d"))
    
    # Test with invalid date
    test_predict_single(token, 1, "PRODUCE", False, "invalid-date")
    
    # Test with extreme values
    test_predict_single(token, 0, "PRODUCE", False, today.strftime("%Y-%m-%d"))  # Invalid store
    test_predict_single(token, 55, "PRODUCE", False, today.strftime("%Y-%m-%d"))  # Store out of range
    test_predict_single(token, 9999, "PRODUCE", False, today.strftime("%Y-%m-%d"))  # Very high store number

def run_all_tests():
    """Run all tests."""
    print("Starting test suite...")
    
    if not test_health():
        print("Health check failed. API may not be running.")
        return False
    
    token = get_auth_token()
    if not token:
        print("Authentication failed. Check credentials.")
        return False
    
    print(f"Successfully authenticated with token: {token[:20]}...")
    
    if not test_diagnostics(token):
        print("Diagnostics test failed.")
        return False
    
    print("\nTesting edge cases...")
    test_edge_cases(token)
    
    print("\nAll tests completed!")
    return True

if __name__ == "__main__":
    run_all_tests() 