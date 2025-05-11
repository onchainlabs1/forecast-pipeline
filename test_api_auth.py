#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test script to verify API authentication directly."""

import requests
import json

API_URL = "http://localhost:8000"

def test_login():
    """Test login and token retrieval from API."""
    print("Testing API authentication...")
    
    # Try login with standard credentials
    try:
        print(f"Sending login request to {API_URL}/token")
        
        # The token endpoint expects form data, not JSON
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": "johndoe",
                "password": "secret"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # Print response info
        print(f"Login response status: {response.status_code}")
        print(f"Response headers: {response.headers}")
        print(f"Response content: {response.text[:200]}")
        
        if response.status_code == 200:
            # Get token
            token_data = response.json()
            token = token_data.get("access_token")
            token_type = token_data.get("token_type", "bearer")
            
            print(f"Login successful!")
            print(f"Token type: {token_type}")
            print(f"Token: {token[:20]}...{token[-20:] if token else ''}")
            
            # Test an authenticated endpoint
            print("\nTesting authenticated endpoint...")
            auth_headers = {"Authorization": f"{token_type} {token}"}
            print(f"Using auth headers: {auth_headers}")
            
            metrics_response = requests.get(f"{API_URL}/metrics_summary", headers=auth_headers)
            print(f"Metrics response status: {metrics_response.status_code}")
            print(f"Metrics response: {metrics_response.text[:200]}")
            
            if metrics_response.status_code == 200:
                print("Authentication works correctly!")
                return True
            else:
                print("Authentication failed on metrics endpoint")
                return False
        else:
            print(f"Login failed: {response.text}")
            return False
    except Exception as e:
        print(f"Error during API test: {str(e)}")
        return False

if __name__ == "__main__":
    test_login() 