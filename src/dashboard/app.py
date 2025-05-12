#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlit dashboard for the sales forecasting application.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from PIL import Image
# Importações opcionais que podem causar problemas
try:
    import jwt
except ImportError:
    import base64
    class FakeJWT:
        @staticmethod
        def decode(token, options=None):
            return {"sub": "user", "exp": 9999999999}
    jwt = FakeJWT()

from plotly.subplots import make_subplots
import time

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Define constants - hardcode to localhost
API_URL = "http://localhost:8002"  # FastAPI backend running on port 8002 
MLFLOW_URL = "http://localhost:8888"  # MLflow on port 8888
LANDING_URL = "http://localhost:8000"  # Landing page running on port 8000

# Configure page with dark theme - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Store Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@example.com',
        'Report a bug': 'mailto:bugs@example.com',
        'About': 'Store Sales Forecasting Dashboard - Predict and visualize sales for different stores and product families.'
    }
)

# Aplicar estilos CSS - Todos os estilos em um único bloco
st.markdown("""
<style>
/* Variáveis de cor e tema escuro - Matching landing page */
:root {
    --background-color: #0e1117;
    --secondary-background-color: #1a1a1a;
    --primary-color: #FF3366;
    --secondary-color: #FF3366;
    --accent-color: #FF3366;
    --success-color: #FF3366;
    --text-color: #ffffff;
    --secondary-text: #a0a0a0;
    color-scheme: dark;
}

/* Override Streamlit default styles to force dark theme */
.stApp {
    background-color: #0e1117 !important;
    color: #ffffff !important;
}

.st-bq {
    background-color: #1a1a1a !important;
}

/* Melhorar o estilo do banner de alerta no topo */
[data-baseweb="notification"] {
    background-color: rgba(255, 51, 102, 0.15) !important;
    border-color: #FF3366 !important;
    color: #ffffff !important;
}

[data-baseweb="notification"] [data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}

.stAlert {
    background-color: rgba(255, 51, 102, 0.15) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border-left: 3px solid #FF3366 !important;
}

/* Corrigir o banner amarelo */
div[data-testid="stNotification"] {
    background-color: rgba(255, 51, 102, 0.15) !important;
    border-color: #FF3366 !important;
    color: #ffffff !important;
}

.stTextInput input, .stNumberInput input, .stDateInput input {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
}

.stSelectbox select, .stMultiSelect select, .stDateInput, .stTimeInput {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
}

/* Buttons styling */
.stButton > button {
    background-color: #FF3366 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}

.stButton > button:hover {
    background-color: #E42D5B !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    transform: translateY(-2px) !important;
}

/* Metric container styling */
[data-testid="stMetric"] {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    border: 1px solid #333;
}

[data-testid="stMetric"] > div {
    color: #FF3366 !important;
    font-weight: 600;
}

[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #a0a0a0 !important;
}

/* Card styling for containers */
[data-testid="stExpander"] {
    background-color: #1a1a1a !important;
    border-radius: 8px !important;
    border: 1px solid #333 !important;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
}

/* Making charts have dark background */
[data-testid="stPlotlyChart"] > div {
    background-color: #1a1a1a !important;
    border-radius: 8px !important;
    padding: 15px !important;
    border: 1px solid #333 !important;
}

/* Custom top banner styling */
.custom-top-banner {
    background: linear-gradient(90deg, rgba(255, 51, 102, 0.15) 0%, rgba(255, 51, 102, 0.2) 100%);
    color: white;
    padding: 15px 20px;
    border-radius: 5px;
    margin-bottom: 20px;
    border-left: 4px solid #FF3366;
    font-weight: 500;
    display: flex;
    align-items: center;
}

.banner-icon {
    color: #FF3366;
    font-size: 24px;
    margin-right: 10px;
}

.banner-text {
    color: #fafafa;
    font-size: 16px;
}

/* Login banner styling */
.login-banner {
    background: linear-gradient(90deg, rgba(15, 17, 23, 0.95) 0%, rgba(15, 17, 23, 0.8) 100%);
    color: white;
    padding: 20px 25px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 51, 102, 0.3);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.login-banner-icon {
    margin-right: 10px;
    font-size: 20px;
    color: #FF3366;
}

.login-banner-text {
    flex-grow: 1;
}

.login-banner-button {
    background-color: #FF3366;
    color: #0e1117;
    padding: 8px 15px;
    border-radius: 5px;
    text-decoration: none;
    font-weight: bold;
    font-size: 14px;
    margin-left: 15px;
    transition: all 0.2s ease;
    border: none;
    cursor: pointer;
}

.login-banner-button:hover {
    background-color: #E42D5B;
    box-shadow: 0 2px 8px rgba(255, 51, 102, 0.5);
}

/* Dashboard cards */
.gradient-header {
    background: linear-gradient(90deg, #FF3366 0%, #E42D5B 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
}

.dashboard-card {
    background-color: #1a1a1a;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    border-left: 3px solid #FF3366;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.dashboard-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

.feature-icon {
    font-size: 28px;
    margin-bottom: 10px;
    color: #FF3366;
}

.stats-card {
    background-color: #1a1a1a;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 10px;
    text-align: center;
    border-bottom: 3px solid #FF3366;
}

.stats-value {
    font-size: 24px;
    font-weight: bold;
    color: #FF3366;
    margin: 5px 0;
}

.stats-label {
    font-size: 12px;
    color: #cccccc;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.badge-ml {
    background-color: rgba(255, 51, 102, 0.2);
    color: #FF3366;
}

.badge-api {
    background-color: rgba(255, 51, 102, 0.2);
    color: #FF3366;
}

.badge-viz {
    background-color: rgba(255, 51, 102, 0.2);
    color: #FF3366;
}

/* Navigation links */
.nav-links {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}

.nav-link {
    color: #ffffff;
    text-decoration: none;
    padding: 8px 12px;
    border-radius: 4px;
    transition: all 0.3s;
    font-weight: 500;
}

.nav-link:hover {
    background-color: rgba(255, 51, 102, 0.2);
    color: #FF3366;
}

.nav-link.active {
    background-color: #FF3366;
    color: white;
}

/* Header with links */
.header-with-links {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.header-title {
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}

.header-links {
    display: flex;
    gap: 15px;
}

.header-link {
    color: #a0a0a0;
    text-decoration: none;
    font-size: 14px;
    transition: color 0.3s;
}

.header-link:hover {
    color: #FF3366;
}

/* Footer styling */
.footer {
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    color: #a0a0a0;
    font-size: 12px;
}

.footer a {
    color: #FF3366;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# Check URL parameters for auto-login
def check_url_params():
    """
    Check URL parameters for token and username and use them for automatic login
    """
    # Get query parameters from URL
    query_params = st.query_params
    
    if "token" in query_params and "username" in query_params and not st.session_state.get("authenticated", False):
        token = query_params["token"]
        username = query_params["username"]
        
        print(f"Auto-login detected with token (first 20 chars): {token[:20]}...")
        
        try:
            # Verify the token by checking with the API
            headers = {"Authorization": f"Bearer {token}"}
            
            # Print for debugging
            print(f"Checking token validity at {API_URL}/users/me")
            response = requests.get(f"{API_URL}/users/me", headers=headers)
            
            if response.status_code == 200:
                print("Auto-login successful!")
                
                # Save token in session state
                st.session_state.token = token
                st.session_state.username = username
                try:
                    st.session_state.user_info = decode_token(token)
                except Exception as e:
                    print(f"Error decoding token: {str(e)}")
                    st.session_state.user_info = {"sub": username}
                
                st.session_state.authenticated = True
                return True
            else:
                print(f"Auto-login failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error during auto-login: {str(e)}")
            return False
    
    return False

# Para debug - se a sessão existe mas está vazia
if "token" not in st.session_state:
    st.session_state.token = None
    st.session_state.token_type = None
    st.session_state.user_info = None

# Inicializar estado para controle de autenticação
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Authentication functions
def login(username, password):
    """
    Login to the API and get a JWT token.
    """
    try:
        # Debug log
        print(f"Attempting login with {username} to API at {API_URL}/token")
        
        # Use the /token endpoint with form data as expected by OAuth2
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": username,
                "password": password
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # Detailed response log
        print(f"Login status code: {response.status_code}")
        print(f"Login response headers: {response.headers}")
        print(f"Login response (first 100 chars): {response.text[:100]}")
        
        if response.status_code == 200:
            # Parse JSON response
            response_data = response.json()
            print(f"Login successful! Token received (first 20 chars): {response_data['access_token'][:20]}")
            print(f"Login successful. Saving token...")
            
            # Store token in session state
            st.session_state.token = response_data["access_token"]
            st.session_state.user_info = decode_token(response_data["access_token"])
            st.session_state.authenticated = True
            
            return response_data
        else:
            print(f"Login failed: {response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        return None

def decode_token(token):
    """
    Decode the JWT token to get user info.
    """
    try:
        # This is just for display, no verification needed
        print(f"Decoding token (first 20 chars): {token[:20]}...")
        decoded = jwt.decode(token, options={"verify_signature": False})
        print(f"Token decoded successfully: {decoded}")
        return decoded
    except Exception as e:
        print(f"Error decoding token: {str(e)}")
        # Try parsing manually if JWT lib fails
        try:
            import base64
            import json
            
            # Basic JWT structure is header.payload.signature
            parts = token.split('.')
            if len(parts) >= 2:
                # Decode the payload (second part)
                padded = parts[1] + '=' * (4 - len(parts[1]) % 4)
                decoded_bytes = base64.b64decode(padded)
                payload = json.loads(decoded_bytes)
                print(f"Manually decoded token: {payload}")
                return payload
        except Exception as manual_error:
            print(f"Error in manual token decoding: {str(manual_error)}")
        
        # If all fails, return a basic structure
        return {"sub": "unknown_user", "exp": 9999999999}

# Helper function to check authentication
def ensure_authenticated():
    """Verifies if the token is present and still valid."""
    if not st.session_state.token:
        return False
        
    # Verify token is working by making a test request
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        print(f"Checking authentication token validity at {API_URL}/users/me")
        response = requests.get(f"{API_URL}/users/me", headers=headers)
        
        if response.status_code == 200:
            print("Token verified and valid")
            return True
        else:
            print(f"Invalid token, status: {response.status_code}, response: {response.text}")
            
            # Show a friendly error message based on the status code
            if response.status_code == 401:
                print("Unauthorized: Token expired or invalid")
                st.error("Your session has expired. Please login again.")
                
                # Clear authentication state
                st.session_state.token = None
                st.session_state.authenticated = False
                
                return False
            elif response.status_code == 404:
                print("Warning: users/me endpoint not found (404)")
                # API might be the landing page API instead of the main API
                if API_URL == "http://localhost:8000":
                    print("Detected localhost:8000 instead of 8002, setting to correct API URL")
                    # Try connecting to the correct API
                    try:
                        alternate_url = "http://localhost:8002"
                        alt_response = requests.get(f"{alternate_url}/users/me", headers=headers)
                        if alt_response.status_code == 200:
                            print("Token verified against alternate API endpoint")
                            return True
                    except Exception as e:
                        print(f"Failed to try alternate API: {e}")
                
                # In development mode, just trust the token
                print("Development mode: proceeding despite 404 on users/me")
                return True
            else:
                # Other errors shouldn't invalidate the token automatically
                # Just log the error but continue
                print(f"Warning: Token verification returned {response.status_code}")
                return True
                
    except requests.exceptions.ConnectionError as e:
        # Handle API connection errors gracefully
        print(f"Warning: Could not connect to API to verify token: {e}")
        st.warning("Could not connect to the API to verify your session. Some features may be unavailable.")
        return True
    except Exception as e:
        print(f"Error verifying token: {str(e)}")
        return True  # Assume token is valid if verification fails due to other errors

# API interaction functions
def get_prediction(token, store_nbr, family, onpromotion, date):
    """
    Get a sales prediction from the API.
    """
    try:
        # If store_nbr is a string like "Store data" or "Store 1", extract the number
        if isinstance(store_nbr, str):
            if store_nbr.lower().startswith("store"):
                parts = store_nbr.split()
                if len(parts) > 1:
                    try:
                        store_nbr = int(parts[1])
                    except ValueError:
                        # Default to store 1 if conversion fails
                        store_nbr = 1
            else:
                # Try to convert directly, or use default
                try:
                    store_nbr = int(store_nbr)
                except ValueError:
                    store_nbr = 1
        
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_URL}/predict_single",
            params={
                "store_nbr": store_nbr,
                "family": family,
                "onpromotion": onpromotion,
                "date": date
            },
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            # Ensure we always have the 'prediction' key
            if 'predicted_sales' in result and 'prediction' not in result:
                result['prediction'] = result['predicted_sales']
            return result
        else:
            error_details = ""
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_details = error_data["detail"]
            except:
                error_details = response.text
                
            st.error(f"Prediction failed: {error_details}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_explanation(token, prediction_id, store_nbr, family, onpromotion, date):
    """
    Get an explanation for a prediction from the API.
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_URL}/explain/{prediction_id}",
            params={
                "store_nbr": store_nbr,
                "family": family,
                "onpromotion": onpromotion,
                "date": date
            },
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Explanation failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_health():
    """
    Get API health status.
    """
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Error connecting to API: {str(e)}"}

def get_stores():
    """
    Get a list of stores from the API.
    """
    try:
        # Get stores from API
        response = requests.get(f"{API_URL}/stores")
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have the data in a nested structure
            if isinstance(data, dict) and "data" in data:
                stores_list = data["data"]
            else:
                stores_list = data
                
            # Ensure we have a list of integers
            if isinstance(stores_list, list):
                # Format each store number
                return [f"Store {int(store)}" for store in stores_list if str(store).isdigit()]
            
        # Fallback stores if API call failed or returned invalid data
        return [f"Store {i}" for i in range(1, 11)]
    except Exception as e:
        # Log the error but don't show to user
        print(f"Error fetching stores: {str(e)}")
        # Fallback stores if API connection fails
        return [f"Store {i}" for i in range(1, 11)]

def get_families():
    """
    Get a list of product families from the API.
    """
    try:
        # Get families from API
        response = requests.get(f"{API_URL}/families")
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have the data in a nested structure
            if isinstance(data, dict) and "data" in data:
                families_list = data["data"]
            else:
                families_list = data
                
            # Ensure we have a valid list of strings
            if isinstance(families_list, list) and all(isinstance(f, str) for f in families_list):
                return [f for f in families_list if f and not f.lower() == "data"]
            
        # Default families if API returns invalid data
        return ["PRODUCE", "GROCERY I", "DAIRY", "BEVERAGES", "BREAD/BAKERY"]
    except Exception as e:
        # Log the error but don't show to user
        print(f"Error fetching product families: {str(e)}")
        # Default families if API connection fails
        return ["PRODUCE", "GROCERY I", "DAIRY", "BEVERAGES", "BREAD/BAKERY"]

def get_sales_data(token, store_nbr, family, days=90):
    """
    Get sales data from the API.
    """
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_URL}/sales_history",
            params={
                "store_nbr": store_nbr,
                "family": family,
                "days": days
            },
            headers=headers
        )
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Handle both old and new response formats
            if isinstance(response_data, dict) and "data" in response_data:
                # New format with metadata
                data = response_data["data"]
                
                # Show warning if mock data
                if response_data.get("is_mock_data", False):
                    message = response_data.get("message", "WARNING: Using simulated sales data")
                    st.warning(message, icon="⚠️")
            else:
                # Old format (direct array)
                data = response_data
            
            return pd.DataFrame(data)
        else:
            st.error(f"Failed to fetch sales data: {response.status_code} - {response.text}")
            return pd.DataFrame(columns=['date', 'sales'])
    except Exception as e:
        st.error(f"Error fetching sales data: {str(e)}")
        return pd.DataFrame(columns=['date', 'sales'])

def get_metric_summary(token):
    """
    Get sales metrics summary from the API.
    """
    try:
        # Detailed debug log
        print(f"Calling metrics_summary with token: {token[:20] if token else 'None'}")
        
        # Ensure we don't send a null token
        if not token:
            print("ERROR: Attempting to call metrics_summary without token")
            return None
            
        # Correct formatting of authentication header
        headers = {"Authorization": f"Bearer {token}"}
        print(f"Headers: {headers}")
        
        # Make the request
        response = requests.get(
            f"{API_URL}/metrics_summary",
            headers=headers
        )
        
        # Detailed response log
        print(f"Status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response: {response.text[:200]}") 
        
        if response.status_code == 200:
            data = response.json()
            print(f"Data received successfully: {str(data)[:100]}...")
            
            # Check if data is mock and show warning
            if data.get("is_mock_data", False):
                message = data.get("message", "WARNING: Using simulated data")
                st.warning(message, icon="⚠️")
                
            return data
        elif response.status_code == 401:
            # If unauthorized, try to renew token
            print("Invalid or expired token. Logging out for reauthentication.")
            # Indicate to user they need to log in again
            st.error("Your session has expired. Please login again.")
            return None
        else:
            st.error(f"Failed to fetch metrics: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching metrics: {str(e)}")
        st.error(f"Error fetching metrics: {str(e)}")
        return None

def get_model_metrics(token, model_name):
    """
    Get the performance metrics for a specific model from the API.
    """
    try:
        # Prepare request headers with authentication
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make the API request
        response = requests.get(
            f"{API_URL}/model_metrics",
            params={"model_name": model_name},
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch model metrics: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching model metrics: {str(e)}")
        return None

def get_feature_importance(token, model_name):
    """
    Get the feature importance data for a specific model from the API.
    """
    try:
        # Prepare request headers with authentication
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make the API request
        response = requests.get(
            f"{API_URL}/feature_importance",
            params={"model_name": model_name},
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch feature importance: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching feature importance: {str(e)}")
        return None

def get_model_drift(token, model_name, days=30):
    """
    Get the model drift data for a specific model from the API.
    """
    try:
        # Prepare request headers with authentication
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make the API request
        response = requests.get(
            f"{API_URL}/model_drift",
            params={"model_name": model_name, "days": days},
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch model drift: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching model drift: {str(e)}")
        return None

# UI components
def render_sidebar():
    """
    Render sidebar with navigation and login/logout functionality.
    """
    with st.sidebar:
        # Dashboard title and logo
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color: #FF3366;">Retail.AI</h1>
            <p>Sales Forecasting Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Links to other parts of the application
        st.markdown("""
        <div style="margin-bottom: 20px;">
            <a href="http://localhost:8000" target="_blank" style="display: block; text-align: center; padding: 8px; background-color: rgba(255, 51, 102, 0.1); border-radius: 5px; color: #FF3366; text-decoration: none; margin-bottom: 10px;">
                🏠 Landing Page
            </a>
            <a href="http://localhost:8888" target="_blank" style="display: block; text-align: center; padding: 8px; background-color: rgba(255, 51, 102, 0.1); border-radius: 5px; color: #FF3366; text-decoration: none;">
                📈 MLflow UI
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        # Auth section
        if not st.session_state.get("authenticated", False):
            # Login form
            with st.expander("Login", expanded=True):
                st.subheader("Login")
                st.text("Please enter your credentials")
                
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submit = st.form_submit_button("Login")
                    
                    if submit:
                        if username and password:
                            with st.spinner("Authenticating..."):
                                auth_response = login(username, password)
                                if auth_response:
                                    # Log after saving token
                                    print(f"Token saved: {st.session_state.token[:20]}...")
                                    
                                    st.success("Login successful!")
                                    st.rerun()
                                else:
                                    st.error("Login failed. Please check your credentials.")
                        else:
                            st.error("Please enter both username and password.")
        else:
            # User info and logout button
            st.markdown(f"""
            <div style="background-color: rgba(255,51,102,0.1); padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                <h4 style="margin:0; color: #FF3366;">Welcome, {st.session_state.get('user_info', {}).get('sub', 'User')}!</h4>
                <p style="margin:0; font-size: 0.8rem; opacity: 0.8;">You are logged in</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Logout", type="primary"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                st.rerun()
        
        # Navigation
        if st.session_state.get("authenticated", False):
            st.subheader("Navigation")
            
            # Create navigation options
            pages = ["Dashboard", "Predictions", "Data Explorer", "Settings"]
            
            # Determine current page
            current_page = st.session_state.get("page", "Dashboard")
            
            # Create radio buttons for navigation
            selected_page = st.radio("Select Page", pages, index=pages.index(current_page))
            
            # Update session state if page changed
            if selected_page != current_page:
                st.session_state.page = selected_page
                st.rerun()

def display_sales_history(store_nbr, family, days):
    try:
        # Get sales history data from API
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(
            f"{API_URL}/sales_history?store_nbr={store_nbr}&family={family}&days={days}",
            headers=headers
        )
        
        if response.status_code != 200:
            st.warning(f"No sales history data available for Store {store_nbr}, {family}")
            return
        
        # Get the data from the response
        try:
            history_data = response.json()
        except Exception as e:
            st.error(f"Error parsing response data: {str(e)}")
            return
        
        # Check if the data is from the API's 'data' field (new format)
        if isinstance(history_data, dict) and 'data' in history_data:
            # New format with metadata
            data = history_data['data']
            is_mock = history_data.get('is_mock_data', False)
            message = history_data.get('message', None)
        else:
            # Old format (just a list of records)
            data = history_data
            is_mock = False
            message = None
        
        if not data or not isinstance(data, list) or len(data) == 0:
            st.warning(f"No sales history data available for Store {store_nbr}, {family}")
            return
        
        # Create DataFrame for plotting, handling potential data issues
        valid_records = []
        for record in data:
            try:
                # Validate and convert data
                if not isinstance(record, dict):
                    continue
                    
                date_val = record.get('date')
                sales_val = record.get('sales')
                is_promotion = record.get('is_promotion', 0)
                
                if date_val is None or sales_val is None:
                    continue
                    
                # Try to convert sales to float
                try:
                    sales_val = float(sales_val)
                except (ValueError, TypeError):
                    # Skip if sales cannot be converted to float
                    continue
                
                # Add to valid records
                valid_records.append({
                    'date': date_val,
                    'sales': sales_val,
                    'is_promotion': int(is_promotion) if is_promotion is not None else 0
                })
            except Exception:
                # Skip any record that causes errors
                continue
        
        if not valid_records:
            st.warning(f"No valid sales history data available for Store {store_nbr}, {family}")
            return
            
        # Create dataframe from valid records
        df = pd.DataFrame(valid_records)
        
        # Try to convert date column to datetime
        try:
            df['date'] = pd.to_datetime(df['date'])
            # Sort by date
            df = df.sort_values('date')
        except Exception as date_error:
            st.warning(f"Error processing date values: {str(date_error)}")
            # If date conversion fails, we can still try to display the data
        
        # Only show warning if explicitly marked as mock data and has a message
        if is_mock and message:
            st.warning(message)
        
        # Display the sales trend chart
        title = f"Sales Trend for Store {store_nbr} - {family}"
        st.subheader(title)
        
        # Create Plotly figure with promotions
        fig = make_subplots()
        
        # Add sales line
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['sales'], 
                mode='lines', 
                name='Sales',
                line=dict(color='#3366CC', width=2)
            )
        )
        
        # Add markers for promotions if available
        if 'is_promotion' in df.columns:
            # Filter only dates with promotions
            promo_df = df[df['is_promotion'] == 1]
            if not promo_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=promo_df['date'], 
                        y=promo_df['sales'], 
                        mode='markers', 
                        name='Promotions',
                        marker=dict(color='red', size=8, symbol='circle')
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=None,
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            template='plotly_dark',
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying sales history: {str(e)}")

def display_dashboard():
    st.markdown("""
    <div class="header-with-links">
        <h2 class="header-title">Store Sales Forecast Dashboard</h2>
        <div class="header-links">
            <a href="http://localhost:8000" target="_blank" class="header-link">Landing Page</a>
            <a href="http://localhost:8888" target="_blank" class="header-link">MLflow UI</a>
            <a href="http://localhost:8002/docs" target="_blank" class="header-link">API Docs</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Verify authentication before trying to load metrics
    if not st.session_state.get("token") or not st.session_state.authenticated:
        st.warning("Please login to view the complete dashboard.")
        return
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get metrics from API
        print("Loading metrics from dashboard...")
        
        # Create proper authorization header with token
        auth_token = st.session_state.token
        headers = {
            "Authorization": f"Bearer {auth_token}"
        }
        print(f"Using auth header: {headers}")
        
        # Make direct API call - avoid using get_metric_summary helper for now
        api_response = requests.get(f"{API_URL}/metrics_summary", headers=headers)
        
        # Log response for debugging
        print(f"API response status: {api_response.status_code}")
        print(f"API response: {api_response.text[:200]}")
        
        if api_response.status_code == 200:
            metrics = api_response.json()
            
            with col1:
                st.metric("Total Stores", str(metrics.get("total_stores", "N/A")))
                
            with col2:
                total_families = metrics.get("total_families", "N/A")
                if isinstance(total_families, (int, float)):
                    total_families_str = f"{total_families} families"
                else:
                    total_families_str = "N/A"
                st.metric("Total Products", total_families_str)
                
            with col3:
                avg_sales = metrics.get("avg_sales", 0)
                if isinstance(avg_sales, (int, float)):
                    formatted_avg = f"${avg_sales:.2f}"
                else:
                    formatted_avg = "N/A"
                st.metric("Average Sales", formatted_avg)
                
            with col4:
                accuracy = metrics.get("forecast_accuracy", 0)
                if isinstance(accuracy, (int, float)):
                    accuracy_str = f"{accuracy:.1f}%"
                else:
                    accuracy_str = "N/A"
                st.metric("Forecast Accuracy", accuracy_str)
        elif api_response.status_code == 401:
            # Authentication error - clear session and show error
            st.error("Authentication failed. Please login again.")
            print("Authentication error from API. Token may be invalid.")
            # Use fallback values
            with col1:
                st.metric("Total Stores", "N/A")
            with col2:
                st.metric("Total Products", "N/A")
            with col3:
                st.metric("Average Sales", "N/A")
            with col4:
                st.metric("Forecast Accuracy", "N/A")
        else:
            # Problems with token or API - show default values
            print(f"Error loading metrics - API returned {api_response.status_code}")
            with col1:
                st.metric("Total Stores", "N/A")
            with col2:
                st.metric("Total Products", "N/A")
            with col3:
                st.metric("Average Sales", "N/A")
            with col4:
                st.metric("Forecast Accuracy", "N/A")
    except Exception as e:
        print(f"Error loading metrics: {str(e)}")
        st.error(f"Error loading metrics: {str(e)}")
        # Use fallback values if API call fails
        with col1:
            st.metric("Total Stores", "N/A")
        with col2:
            st.metric("Total Products", "N/A")
        with col3:
            st.metric("Average Sales", "N/A")
        with col4:
            st.metric("Forecast Accuracy", "N/A")
    
    # Dashboard Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("Store")
        # Get store options and filter out invalid ones
        store_options = get_stores()
        store_options = [s for s in store_options if not s.lower().endswith("data")]
        if not store_options:
            store_options = [f"Store {i}" for i in range(1, 11)]
        
        store_filter = st.selectbox("", options=store_options, index=0, key="dashboard_store")
    
    with col2:
        st.markdown("Product Family")
        # Get family options and filter out invalid ones
        family_options = get_families()
        family_options = [f for f in family_options if f.lower() != "data"]
        if not family_options:
            family_options = ["PRODUCE", "GROCERY I", "DAIRY", "BEVERAGES", "BREAD/BAKERY"]
            
        family_filter = st.selectbox("", options=family_options, index=0, key="dashboard_family")
    
    with col3:
        st.markdown("Days to Display")
        days_filter = st.slider("", min_value=30, max_value=365, value=90)
    
    # Extract store number from formatted string
    try:
        # Try to extract a number from the store string
        store_parts = store_filter.split()
        if len(store_parts) > 1 and store_parts[0].lower() == "store":
            try:
                store_nbr = int(store_parts[1])
            except ValueError:
                # If conversion fails, use a default value
                store_nbr = 1
        else:
            # Default if the format is unexpected
            store_nbr = 1
    except Exception:
        # Fallback value
        store_nbr = 1
    
    # Display visualization based on filters
    display_sales_history(store_nbr, family_filter, days_filter)
    
    # Show additional metrics/charts
    try:
        # Get store comparison data
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{API_URL}/store_comparison", headers=headers)
        
        if response.status_code == 200:
            store_data = response.json()
            
            if store_data:
                st.subheader("Sales Trends")
                
                # Handle potential data issues
                valid_data = []
                for item in store_data:
                    try:
                        # Ensure all required fields are present and of the correct type
                        store_name = item.get('store', f"Store {len(valid_data)+1}")
                        sales_value = float(item.get('sales', 0))
                        forecast_accuracy = float(item.get('forecast_accuracy', 0.8))
                        
                        valid_data.append({
                            'store': store_name,
                            'sales': sales_value,
                            'forecast_accuracy': forecast_accuracy
                        })
                    except (ValueError, TypeError):
                        # Skip invalid entries
                        continue
                
                if valid_data:
                    # Create DataFrame from valid data
                    df_stores = pd.DataFrame(valid_data)
                    
                    # Sort by sales in descending order
                    df_stores = df_stores.sort_values(by='sales', ascending=False)
                    
                    # Create comparison bar chart
                    fig = px.bar(
                        df_stores, 
                        x='store', 
                        y='sales',
                        color='forecast_accuracy', 
                        color_continuous_scale='Viridis',
                        labels={'sales': 'Total Sales ($)', 'store': 'Store', 'forecast_accuracy': 'Forecast Accuracy'},
                        title=f"Sales Trend for Store {store_nbr} - {family_filter}",
                    )
                    
                    # Update layout for dark theme
                    fig.update_layout(
                        template='plotly_dark',
                        coloraxis_colorbar=dict(title="Accuracy"),
                        title=None,
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid data available for store comparison chart")
            else:
                st.warning("No data available for store comparison")
        else:
            st.warning(f"Failed to fetch store comparison data: {response.status_code}")
    except Exception as e:
        st.error(f"Error loading store comparison: {str(e)}")
        
    # Add footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Retail.AI - Sales Forecasting Dashboard | <a href="http://localhost:8000" target="_blank">Landing Page</a> | <a href="http://localhost:8888" target="_blank">MLflow UI</a> | <a href="http://localhost:8002/docs" target="_blank">API Docs</a></p>
    </div>
    """, unsafe_allow_html=True)

def render_predictions():
    """
    Render the predictions page using real data from the API.
    """
    st.title("Sales Predictions")
    
    # Adicionar CSS para esta página específica
    st.markdown("""
    <style>
    .prediction-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 3px solid #4fd1c5;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }
    
    .prediction-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }
    
    .prediction-results {
        background-color: #262730;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
        border-left: 3px solid #FFD166;
    }
    
    .real-prediction {
        background-color: rgba(6, 214, 160, 0.1);
        border: 1px solid rgba(6, 214, 160, 0.3);
        border-radius: 8px;
        padding: 15px;
    }
    
    .form-container {
        background-color: #262730;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .animated-bg {
        background: linear-gradient(270deg, #1E1E1E, #262730);
        background-size: 400% 400%;
        animation: gradient 8s ease infinite;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Descrição mais atraente
    st.markdown("""
    <div class="prediction-card animated-bg">
        <div class="prediction-header">
            <h3 style="color: #4fd1c5; margin: 0; flex-grow: 1;">Advanced Sales Forecasting</h3>
            <span style="background-color: rgba(79, 209, 197, 0.2); color: #4fd1c5; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">ML POWERED</span>
        </div>
        <p>Use our state-of-the-art machine learning models to predict future sales for specific stores and product families. Our models achieve over 80% accuracy and can help optimize inventory management and marketing strategies.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health before allowing predictions
    health = get_health()
    if health["status"] != "healthy":
        st.error(f"API is not available: {health.get('message', 'Unknown error')}")
        st.warning("Please make sure the API is running before making predictions.")
        st.stop()
    
    # Get stores and families from API
    stores = get_stores()
    families = get_families()
    
    # Filter out invalid options
    stores = [s for s in stores if not s.lower().endswith("data")]
    families = [f for f in families if f.lower() != "data"]
    
    if not stores:
        # Ensure we have at least some valid stores
        stores = [f"Store {i}" for i in range(1, 11)]
    
    if not families:
        # Ensure we have at least some valid families
        families = ["PRODUCE", "GROCERY I", "DAIRY", "BEVERAGES", "BREAD/BAKERY"]
    
    # Show advanced options
    with st.expander("Advanced Options"):
        enable_debug = st.checkbox("Show Debug Information", value=False)
    
    # Form for predictions within card
    st.markdown("""
    <div class="form-container">
        <h4 style="color: #FFD166; margin-top: 0; margin-bottom: 15px;">Prediction Parameters</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Default values we know that work
        default_store_index = 0
        default_family_index = 0
        
        # Try to find default values in the stores and families lists
        if stores and families:
            # Look for store 1 in the stores list
            for i, store in enumerate(stores):
                if store.endswith("1"):
                    default_store_index = i
                    break
                
            # Look for "PRODUCE" in the families list
            for good_family in ["PRODUCE", "FROZEN FOODS", "GROCERY I", "DAIRY"]:
                if good_family in families:
                    default_family_index = families.index(good_family)
                    break
        
        with col1:
            store_nbr = st.selectbox("Store Number", stores, index=default_store_index)
        
        with col2:
            family = st.selectbox("Product Family", families, index=default_family_index)
        
        with col3:
            onpromotion = st.checkbox("On Promotion")
        
        date = st.date_input("Prediction Date", datetime.now().date() + timedelta(days=1))
        
        submitted = st.form_submit_button("Generate Prediction")
    
    # Process form submission
    if submitted:
        if "token" in st.session_state:
            with st.spinner("Generating prediction..."):
                # Extract store number from string (if needed)
                store_val = 1  # Default value
                
                try:
                    if isinstance(store_nbr, str) and "store" in store_nbr.lower():
                        # Format is like "Store 1" - extract the number
                        parts = store_nbr.split()
                        if len(parts) > 1 and parts[1].isdigit():
                            store_val = int(parts[1])
                    elif isinstance(store_nbr, (int, float)):
                        store_val = int(store_nbr)
                except (ValueError, TypeError):
                    # If conversion fails, use store 1
                    store_val = 1
                
                # Make prediction request
                prediction = get_prediction(
                    st.session_state.token,
                    store_val,
                    family,
                    onpromotion,
                    date.strftime("%Y-%m-%d")
                )
                
                if prediction:
                    # Show prediction results in visually appealing card
                    # Check if this is a fallback prediction
                    is_fallback = prediction.get('is_fallback', False)
                    saved_to_db = prediction.get('saved_to_db', False)
                    
                    # Format prediction
                    st.markdown("""
                    <div class="prediction-card">
                        <div class="prediction-header">
                            <h3 style="color: #4fd1c5; margin: 0; flex-grow: 1;">Prediction Results</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Parâmetros
                    st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <div style="display: inline-block; background-color: #1a1a1a; padding: 5px 10px; border-radius: 4px; margin-right: 10px; margin-bottom: 10px;">
                            <span style="color: #888; font-size: 0.75em;">STORE</span><br>
                            <span style="color: #fff;">{store_nbr}</span>
                        </div>
                        <div style="display: inline-block; background-color: #1a1a1a; padding: 5px 10px; border-radius: 4px; margin-right: 10px; margin-bottom: 10px;">
                            <span style="color: #888; font-size: 0.75em;">PRODUCT</span><br>
                            <span style="color: #fff;">{family}</span>
                        </div>
                        <div style="display: inline-block; background-color: #1a1a1a; padding: 5px 10px; border-radius: 4px; margin-right: 10px; margin-bottom: 10px;">
                            <span style="color: #888; font-size: 0.75em;">DATE</span><br>
                            <span style="color: #fff;">{date.strftime("%Y-%m-%d")}</span>
                        </div>
                        <div style="display: inline-block; background-color: #1a1a1a; padding: 5px 10px; border-radius: 4px; margin-bottom: 10px;">
                            <span style="color: #888; font-size: 0.75em;">PROMOTION</span><br>
                            <span style="color: #fff;">{"Yes" if onpromotion else "No"}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction result
                    prediction_value = None
                    if 'prediction' in prediction:
                        prediction_value = prediction['prediction']
                    elif 'predicted_sales' in prediction:
                        prediction_value = prediction['predicted_sales']
                    else:
                        st.error("Prediction data format is invalid. Missing prediction value.")
                        prediction_value = 0
                    
                    if is_fallback:
                        message = prediction.get('message', 'WARNING: Using simulated prediction (fallback)')
                        st.warning(message, icon="⚠️")
                        
                        st.markdown(f"""
                        <div style="background-color: rgba(255, 209, 102, 0.1); border: 1px solid rgba(255, 209, 102, 0.3); border-radius: 8px; padding: 15px; text-align: center;">
                            <h3 style="color: #FFD166; margin: 0 0 5px 0;">Predicted Sales (SIMULATED)</h3>
                            <p style="font-size: 32px; font-weight: bold; margin: 0; color: #FFD166;">${prediction_value:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show error details if debug is enabled
                        if enable_debug and 'error' in prediction:
                            st.error(f"Model Error: {prediction['error']}")
                    else:
                        st.markdown(f"""
                        <div class="real-prediction" style="text-align: center;">
                            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 5px;">
                                <h3 style="color: #06D6A0; margin: 0;">Predicted Sales</h3>
                                <span style="background-color: rgba(6, 214, 160, 0.2); color: #06D6A0; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 10px;">REAL MODEL</span>
                            </div>
                            <p style="font-size: 42px; font-weight: bold; margin: 10px 0; color: #06D6A0;">${prediction_value:.2f}</p>
                            <p style="font-size: 14px; margin: 0; opacity: 0.7;">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show if prediction was saved to database
                        if enable_debug:
                            if saved_to_db:
                                st.success("✅ Prediction saved to database")
                            else:
                                st.warning("⚠️ Prediction not saved to database")
                    
                    # Show raw prediction data in debug mode
                    if enable_debug:
                        with st.expander("Raw Prediction Data"):
                            st.json(prediction)
                    
                    # Save prediction ID for explanation
                    prediction_id = prediction.get('prediction_id', f"{store_val}-{family}-{date}")
                    
                    # Get explanation
                    with st.spinner("Generating explanation..."):
                        try:
                            explanation = get_explanation(
                                st.session_state.token,
                                prediction_id,
                                store_val,
                                family,
                                onpromotion,
                                date.strftime("%Y-%m-%d")
                            )
                            
                            if explanation:
                                st.markdown("""
                                <div class="prediction-header" style="margin-top: 30px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                                    <h3 style="color: #4fd1c5; margin: 0; flex-grow: 1;">Prediction Explanation</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Check for explanation message that might indicate it's a fallback
                                exp_message = explanation.get("message", "")
                                if "error" in exp_message.lower() or "not available" in exp_message.lower():
                                    st.warning(exp_message, icon="⚠️")
                                
                                # Display feature contributions
                                feature_contributions = explanation.get("feature_contributions", [])
                                if feature_contributions:
                                    # Create dataframe for chart
                                    feat_df = pd.DataFrame(feature_contributions)
                                    
                                    # Sort by absolute contribution
                                    feat_df['abs_contribution'] = feat_df['contribution'].abs()
                                    feat_df = feat_df.sort_values('abs_contribution', ascending=False).head(10)
                                    
                                    # Create waterfall chart
                                    fig = go.Figure(go.Waterfall(
                                        name="Features",
                                        orientation="h",
                                        measure=["relative"] * len(feat_df),
                                        y=feat_df['feature'],
                                        x=feat_df['contribution'],
                                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                                        decreasing={"marker": {"color": "rgba(255, 50, 50, 0.8)"}},
                                        increasing={"marker": {"color": "rgba(50, 200, 50, 0.8)"}},
                                        text=feat_df['value'].round(2)
                                    ))
                                    
                                    fig.update_layout(
                                        title="Feature Contributions to Prediction",
                                        showlegend=False,
                                        template="plotly_dark",
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Adicionar explicação sobre as features principais
                                    top_features = feat_df.head(3)
                                    if not top_features.empty:
                                        st.markdown("""
                                        <div style="background-color: #262730; border-radius: 8px; padding: 15px; margin-top: 15px;">
                                            <h4 style="color: #FFD166; margin-top: 0;">Key Factors Influencing This Prediction</h4>
                                        """, unsafe_allow_html=True)
                                        
                                        for _, row in top_features.iterrows():
                                            direction = "increased" if row['contribution'] > 0 else "decreased"
                                            magnitude = abs(row['contribution'])
                                            color = "#06D6A0" if row['contribution'] > 0 else "#FF4B4B"
                                            
                                            st.markdown(f"""
                                            <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #333;">
                                                <span style="color: {color}; font-weight: bold;">{row['feature']}</span>: 
                                                <span>Value of <strong>{row['value']:.2f}</strong> {direction} the prediction by <strong>${magnitude:.2f}</strong></span>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        
                                        st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.warning("Explanation available but no feature contributions found.")
                                    
                                # Show raw explanation in debug mode
                                if enable_debug:
                                    with st.expander("Raw Explanation Data"):
                                        st.json(explanation)
                            else:
                                st.warning("Unable to generate explanation for this prediction.")
                        except Exception as exp_error:
                            st.error(f"Error generating explanation: {str(exp_error)}")
                            st.warning("Unable to generate explanation for this prediction. Model may not support explainability.")
                            
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Sugestões de ação com base na previsão
                    if prediction_value is not None:
                        st.markdown("""
                        <div class="prediction-card" style="border-left: 3px solid #FFD166;">
                            <div class="prediction-header">
                                <h3 style="color: #FFD166; margin: 0; flex-grow: 1;">Recommended Actions</h3>
                            </div>
                            
                            <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px;">
                        """, unsafe_allow_html=True)
                        
                        # Determinar recomendações com base no valor da previsão
                        is_high_value = prediction_value > 50
                        is_promotion = onpromotion
                        
                        if is_high_value and not is_promotion:
                            st.markdown("""
                            <div style="flex: 1; min-width: 200px; background-color: rgba(6, 214, 160, 0.1); border-radius: 8px; padding: 15px; border: 1px solid rgba(6, 214, 160, 0.3);">
                                <h4 style="color: #06D6A0; margin-top: 0;">Consider Promotion</h4>
                                <p style="margin-bottom: 0;">High predicted sales volume indicates strong demand. Consider running a promotion to maximize revenue.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif is_high_value and is_promotion:
                            st.markdown("""
                            <div style="flex: 1; min-width: 200px; background-color: rgba(6, 214, 160, 0.1); border-radius: 8px; padding: 15px; border: 1px solid rgba(6, 214, 160, 0.3);">
                                <h4 style="color: #06D6A0; margin-top: 0;">Increase Inventory</h4>
                                <p style="margin-bottom: 0;">High sales expected with promotion in effect. Ensure adequate inventory to meet increased demand.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif not is_high_value and not is_promotion:
                            st.markdown("""
                            <div style="flex: 1; min-width: 200px; background-color: rgba(255, 209, 102, 0.1); border-radius: 8px; padding: 15px; border: 1px solid rgba(255, 209, 102, 0.3);">
                                <h4 style="color: #FFD166; margin-top: 0;">Optimize Inventory</h4>
                                <p style="margin-bottom: 0;">Low predicted sales volume. Consider reducing stock levels to minimize holding costs.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="flex: 1; min-width: 200px; background-color: rgba(255, 209, 102, 0.1); border-radius: 8px; padding: 15px; border: 1px solid rgba(255, 209, 102, 0.3);">
                                <h4 style="color: #FFD166; margin-top: 0;">Review Promotion Strategy</h4>
                                <p style="margin-bottom: 0;">Low sales despite promotion. Consider revising promotion strategy or product placement.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Decisão de compra
                        st.markdown("""
                        <div style="flex: 1; min-width: 200px; background-color: rgba(66, 153, 225, 0.1); border-radius: 8px; padding: 15px; border: 1px solid rgba(66, 153, 225, 0.3);">
                            <h4 style="color: #4299E1; margin-top: 0;">Purchasing Decision</h4>
                            <p style="margin-bottom: 0;">Optimize order quantity based on the predicted sales of ${:.2f} units per day.</p>
                        </div>
                        """.format(prediction_value), unsafe_allow_html=True)
                        
                        st.markdown("""
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.error("You must login first to make predictions.")
            st.markdown("""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin-top: 20px; border-left: 3px solid #FF4B4B;">
                <h4 style="color: #FF4B4B; margin-top: 0;">Authentication Required</h4>
                <p>Please login using the sidebar to access the prediction functionality.</p>
            </div>
            """, unsafe_allow_html=True)

def render_model_insights():
    """
    Render the model insights page.
    """
    # Adicionar estilo para a página de insights
    st.markdown("""
    <style>
    .insight-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 3px solid #4fd1c5;
        transition: all 0.3s ease;
    }
    
    .insight-card:hover {
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }
    
    .insight-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }
    
    .metric-card {
        background-color: #262730;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .metric-label {
        font-size: 12px;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .gradient-text {
        background: linear-gradient(90deg, #4fd1c5, #4299e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }
    
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge-good {
        background-color: rgba(6, 214, 160, 0.2);
        color: #06D6A0;
    }
    
    .badge-warning {
        background-color: rgba(255, 209, 102, 0.2);
        color: #FFD166;
    }
    
    .badge-danger {
        background-color: rgba(255, 75, 75, 0.2);
        color: #FF4B4B;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Título com gradiente
    st.markdown("""
    <h1 class="gradient-text" style="font-size: 36px; margin-bottom: 20px;">Model Insights & Performance</h1>
    """, unsafe_allow_html=True)
    
    # Descrição moderna
    st.markdown("""
    <div class="insight-card">
        <div class="insight-header">
            <h3 style="color: #4fd1c5; margin: 0; flex-grow: 1;">Performance Analytics</h3>
            <span class="badge badge-good">ML MONITORING</span>
        </div>
        <p>Explore detailed model performance metrics, feature importance, and drift analysis to understand how our forecasting models are performing and identify areas for improvement.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Selecionar modelo
    if "token" not in st.session_state:
        st.warning("Please login to view model insights.")
        st.markdown("""
        <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin-top: 20px; border-left: 3px solid #FF4B4B;">
            <h4 style="color: #FF4B4B; margin-top: 0;">Authentication Required</h4>
            <p>Please login using the sidebar to access the model insights functionality.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Mostrar modelos disponíveis de forma mais atraente
    st.markdown("""
    <div class="insight-card" style="border-left: 3px solid #FFD166;">
        <div class="insight-header">
            <h3 style="color: #FFD166; margin: 0;">Available Models</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Modelos disponíveis
    models = ["LightGBM (Production)", "XGBoost (Staging)", "Prophet (Development)"]
    
    # Cartões para cada modelo
    cols = st.columns(len(models))
    for i, model in enumerate(models):
        with cols[i]:
            if model == "LightGBM (Production)":
                badge = '<span class="badge badge-good">PRODUCTION</span>'
                border = "#06D6A0"
            elif model == "XGBoost (Staging)":
                badge = '<span class="badge badge-warning">STAGING</span>'
                border = "#FFD166"
            else:
                badge = '<span class="badge badge-danger">DEV</span>'
                border = "#FF4B4B"
                
            st.markdown(f"""
            <div style="background-color: #262730; border-radius: 8px; padding: 15px; text-align: center; border-left: 3px solid {border}; cursor: pointer;" 
                 onclick="alert('Select this model')">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #fff;">{model.split(' ')[0]}</h4>
                    {badge}
                </div>
                <p style="font-size: 12px; opacity: 0.7; margin: 0; text-align: left;">{model.split(' ')[1].replace('(', '').replace(')', '')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Continuar com o resto do código
    st.markdown("""
    <div class="insight-card">
        <div class="insight-header">
            <h3 style="color: #4fd1c5; margin: 0;">Model Selection</h3>
        </div>
    """, unsafe_allow_html=True)
    
    model_name = st.selectbox("Select Model to Analyze", 
                            ["LightGBM (Production)", "XGBoost (Staging)", "Prophet (Development)"],
                            index=0)
    
    # Time period para análise
    col1, col2 = st.columns(2)
    with col1:
        days = st.slider("Historical Period (days)", min_value=7, max_value=90, value=30, step=7)
    with col2:
        threshold = st.slider("Performance Threshold (%)", min_value=50, max_value=95, value=75, step=5)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load model metrics
    with st.spinner("Loading model performance metrics..."):
        try:
            metrics = get_model_metrics(st.session_state.token, model_name)
            
            if metrics:
                # Overall performance 
                st.markdown("""
                <div class="insight-card">
                    <div class="insight-header">
                        <h3 style="color: #4fd1c5; margin: 0;">Performance Overview</h3>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                """, unsafe_allow_html=True)
                
                # Determine color based on value
                def get_metric_color(value, metric_type):
                    if metric_type == "accuracy" or metric_type == "r2":
                        if value >= 0.8:
                            return "#06D6A0"  # Good (green)
                        elif value >= 0.6:
                            return "#FFD166"  # Warning (yellow)
                        else:
                            return "#FF4B4B"  # Bad (red)
                    elif metric_type == "rmse" or metric_type == "mae":
                        if value <= 10:
                            return "#06D6A0"  # Good (green)
                        elif value <= 25:
                            return "#FFD166"  # Warning (yellow)
                        else:
                            return "#FF4B4B"  # Bad (red)
                    else:
                        return "#4fd1c5"  # Default
                
                # Create metric cards with appropriate colors
                metric_data = [
                    {"name": "Accuracy", "value": metrics.get("accuracy", 0) * 100, "format": "%.1f%%", "type": "accuracy"},
                    {"name": "RMSE", "value": metrics.get("rmse", 0), "format": "%.2f", "type": "rmse"},
                    {"name": "MAE", "value": metrics.get("mae", 0), "format": "%.2f", "type": "mae"},
                    {"name": "R² Score", "value": metrics.get("r2", 0), "format": "%.3f", "type": "r2"}
                ]
                
                for metric in metric_data:
                    color = get_metric_color(metric["value"] / 100 if metric["name"] == "Accuracy" else metric["value"], metric["type"])
                    
                    st.markdown(f"""
                    <div style="flex: 1; min-width: 150px;">
                        <div class="metric-card" style="border-top: 3px solid {color};">
                            <div class="metric-label">{metric["name"]}</div>
                            <div class="metric-value" style="color: {color};">{metric["format"] % metric["value"]}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Feature importance
                with st.spinner("Loading feature importance..."):
                    feature_importance = get_feature_importance(st.session_state.token, model_name)
                    
                    if feature_importance and isinstance(feature_importance, list):
                        # Organize data for visualization
                        feature_df = pd.DataFrame(feature_importance)
                        
                        if not feature_df.empty and "feature" in feature_df.columns and "importance" in feature_df.columns:
                            # Sort by importance
                            feature_df = feature_df.sort_values("importance", ascending=False).head(10)
                            
                            st.markdown("""
                            <div class="insight-card">
                                <div class="insight-header">
                                    <h3 style="color: #4fd1c5; margin: 0;">Feature Importance</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Create animated bar chart
                            fig = px.bar(
                                feature_df,
                                x="importance",
                                y="feature",
                                orientation="h",
                                color="importance",
                                color_continuous_scale=["#4299e1", "#4fd1c5", "#06D6A0"],
                                title="Top 10 Most Important Features"
                            )
                            
                            fig.update_layout(
                                template="plotly_dark",
                                xaxis_title="Relative Importance",
                                yaxis_title="",
                                coloraxis_showscale=False,
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add insights about top features
                            top_features = feature_df.head(3)
                            
                            st.markdown("""
                            <div style="background-color: #262730; border-radius: 8px; padding: 15px; margin-top: 10px;">
                                <h4 style="color: #FFD166; margin-top: 0;">Key Findings</h4>
                                <ul style="margin-bottom: 0; padding-left: 20px;">
                            """, unsafe_allow_html=True)
                            
                            for _, row in top_features.iterrows():
                                st.markdown(f"""
                                <li style="margin-bottom: 8px;"><strong style="color: #4fd1c5;">{row['feature']}</strong> is the most significant feature, accounting for <strong>{row['importance']:.1f}%</strong> of the model's predictive power.</li>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("""
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Feature importance data is not available for this model.")
                
                # Model drift analysis
                with st.spinner("Analyzing model drift..."):
                    drift_data = get_model_drift(st.session_state.token, model_name, days)
                    
                    if drift_data and isinstance(drift_data, dict) and "dates" in drift_data and "accuracy" in drift_data:
                        drift_df = pd.DataFrame({
                            "date": drift_data["dates"],
                            "accuracy": drift_data["accuracy"]
                        })
                        
                        st.markdown("""
                        <div class="insight-card">
                            <div class="insight-header">
                                <h3 style="color: #4fd1c5; margin: 0;">Model Drift Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Create line chart for model drift
                        fig = px.line(
                            drift_df,
                            x="date",
                            y="accuracy",
                            markers=True,
                            title=f"Model Accuracy Over Time (Last {days} Days)",
                            color_discrete_sequence=["#4fd1c5"]
                        )
                        
                        # Add threshold line
                        fig.add_shape(
                            type="line",
                            x0=drift_df["date"].min(),
                            x1=drift_df["date"].max(),
                            y0=threshold / 100,
                            y1=threshold / 100,
                            line=dict(color="#FF4B4B", width=2, dash="dash")
                        )
                        
                        # Add annotation for threshold
                        fig.add_annotation(
                            x=drift_df["date"].max(),
                            y=threshold / 100,
                            text=f"Threshold ({threshold}%)",
                            showarrow=False,
                            yshift=10,
                            font=dict(color="#FF4B4B")
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            xaxis_title="Date",
                            yaxis_title="Accuracy",
                            yaxis=dict(tickformat=".0%"),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Calculate drift metrics
                        current_accuracy = drift_df["accuracy"].iloc[-1] if not drift_df.empty else 0
                        avg_accuracy = drift_df["accuracy"].mean() if not drift_df.empty else 0
                        min_accuracy = drift_df["accuracy"].min() if not drift_df.empty else 0
                        
                        drift_analysis = ""
                        drift_status = ""
                        
                        if current_accuracy < threshold / 100:
                            drift_status = '<span class="badge badge-danger">CRITICAL</span>'
                            drift_analysis = "Model is currently performing below the acceptable threshold. Retraining is required."
                        elif current_accuracy < avg_accuracy * 0.95:
                            drift_status = '<span class="badge badge-warning">DRIFT DETECTED</span>'
                            drift_analysis = "Model is showing signs of drift. Consider retraining soon."
                        else:
                            drift_status = '<span class="badge badge-good">STABLE</span>'
                            drift_analysis = "Model is performing stably. No immediate action required."
                        
                        st.markdown(f"""
                        <div style="background-color: #262730; border-radius: 8px; padding: 15px; margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h4 style="color: #FFD166; margin: 0;">Drift Status</h4>
                                {drift_status}
                            </div>
                            <p style="margin-bottom: 15px;">{drift_analysis}</p>
                            
                            <div style="display: flex; flex-wrap: wrap; gap: 15px;">
                                <div style="flex: 1; min-width: 120px; background-color: #1a1a1a; padding: 10px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 11px; text-transform: uppercase; color: #888;">Current</div>
                                    <div style="font-size: 18px; font-weight: bold; color: #4fd1c5;">{current_accuracy:.1%}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px; background-color: #1a1a1a; padding: 10px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 11px; text-transform: uppercase; color: #888;">Average</div>
                                    <div style="font-size: 18px; font-weight: bold; color: #4fd1c5;">{avg_accuracy:.1%}</div>
                                </div>
                                <div style="flex: 1; min-width: 120px; background-color: #1a1a1a; padding: 10px; border-radius: 8px; text-align: center;">
                                    <div style="font-size: 11px; text-transform: uppercase; color: #888;">Minimum</div>
                                    <div style="font-size: 18px; font-weight: bold; color: #4fd1c5;">{min_accuracy:.1%}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Model drift data is not available for this model and time period.")
            else:
                st.error("Failed to fetch model metrics. Please try again later.")
        except Exception as e:
            st.error(f"Error fetching model insights: {str(e)}")
            st.markdown("""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; margin-top: 20px; border-left: 3px solid #FF4B4B;">
                <h4 style="color: #FF4B4B; margin-top: 0;">Error Loading Insights</h4>
                <p>There was a problem loading the model insights. This could be due to API unavailability or mock data limitations.</p>
            </div>
            """, unsafe_allow_html=True)

def render_settings():
    """
    Render the settings page.
    """
    st.title("Settings")
    
    st.markdown("""
    Configure your dashboard preferences and API connections.
    """)
    
    # Tabs for different settings
    tab1, tab2, tab3 = st.tabs(["Dashboard", "API Connection", "Notifications"])
    
    with tab1:
        st.subheader("Dashboard Settings")
        
        # Theme selection
        st.selectbox(
            "Theme", 
            ["Light", "Dark", "Auto"],
            index=0
        )
        
        # Default views
        st.multiselect(
            "Default Dashboard Widgets",
            ["Sales Trends", "Store Comparison", "Product Performance", "Forecast Accuracy", "Alerts"],
            default=["Sales Trends", "Store Comparison", "Product Performance"]
        )
        
        # Chart preferences
        st.radio(
            "Default Chart Type",
            ["Line", "Bar", "Area"],
            horizontal=True
        )
    
    with tab2:
        st.subheader("API Connection Settings")
        
        # API URL
        current_api = st.text_input("API URL", API_URL)
        
        # MLflow URL
        current_mlflow = st.text_input("MLflow URL", MLFLOW_URL)
        
        # Test connection button
        if st.button("Test Connection"):
            health = get_health()
            if health["status"] == "healthy":
                st.success("Connection successful! API is healthy.")
            else:
                st.error(f"Connection failed: {health.get('message', 'Unknown error')}")
    
    with tab3:
        st.subheader("Notification Settings")
        
        # Email notifications
        st.checkbox("Enable Email Notifications", value=True)
        
        st.text_input("Email Address", placeholder="your.email@example.com")
        
        # Notification preferences
        st.multiselect(
            "Notification Types",
            ["Model Drift Alerts", "Prediction Errors", "System Status", "Weekly Reports"],
            default=["Model Drift Alerts", "System Status"]
        )
        
        # Notification frequency
        st.select_slider(
            "Notification Frequency",
            options=["Immediate", "Hourly", "Daily", "Weekly"]
        )

# Add custom CSS
def local_css():
    css = """
    .custom-banner {
        background-color: rgba(255, 51, 102, 0.1);
        border-left: 4px solid #FF3366;
        padding: 15px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }

    .custom-banner i {
        color: #FF3366;
        margin-right: 10px;
    }

    .custom-banner p {
        margin: 0;
        color: #ffffff;
    }

    .spinner {
        width: 40px;
        height: 40px;
        margin: 20px auto;
        border: 3px solid rgba(255, 51, 102, 0.3);
        border-radius: 50%;
        border-top-color: #FF3366;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_data_explorer():
    """
    Render the data explorer page.
    """
    st.title("Data Explorer")
    
    st.markdown("""
    <div class="dashboard-card">
        <div class="feature-icon">🔍</div>
        <h3>Explore Sales Data</h3>
        <p>Analyze historical sales data across different stores and product families.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user is authenticated
    if not st.session_state.get("authenticated", False):
        st.warning("Please login to access the data explorer.")
        return
    
    # Create filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        store_options = get_stores()
        store_filter = st.selectbox("Select Store", options=store_options, index=0)
    
    with col2:
        family_options = get_families()
        family_filter = st.selectbox("Select Product Family", options=family_options, index=0)
    
    with col3:
        date_range = st.slider("Date Range (days)", min_value=30, max_value=365, value=90)
    
    # Extract store number
    try:
        if isinstance(store_filter, str) and "store" in store_filter.lower():
            store_parts = store_filter.split()
            if len(store_parts) > 1:
                store_nbr = int(store_parts[1])
            else:
                store_nbr = 1
        else:
            store_nbr = 1
    except:
        store_nbr = 1
    
    # Get data from API
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            sales_data = get_sales_data(st.session_state.token, store_nbr, family_filter, date_range)
            
            if not sales_data.empty:
                # Display data
                st.subheader(f"Sales Data for {store_filter} - {family_filter}")
                
                # Convert date to datetime if it's not already
                if 'date' in sales_data.columns:
                    try:
                        sales_data['date'] = pd.to_datetime(sales_data['date'])
                    except:
                        pass
                
                # Show data table
                st.dataframe(sales_data)
                
                # Create visualization
                if 'date' in sales_data.columns and 'sales' in sales_data.columns:
                    fig = px.line(
                        sales_data, 
                        x='date', 
                        y='sales',
                        title=f"Sales Trend for {store_filter} - {family_filter}",
                        labels={'date': 'Date', 'sales': 'Sales ($)'},
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    st.subheader("Sales Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Sales", f"${sales_data['sales'].sum():.2f}")
                    
                    with col2:
                        st.metric("Average Sales", f"${sales_data['sales'].mean():.2f}")
                    
                    with col3:
                        st.metric("Maximum Sales", f"${sales_data['sales'].max():.2f}")
                    
                    with col4:
                        st.metric("Minimum Sales", f"${sales_data['sales'].min():.2f}")
                    
                    # Download data option
                    csv = sales_data.to_csv(index=False)
                    st.download_button(
                        label="Download Data as CSV",
                        data=csv,
                        file_name=f"sales_data_{store_nbr}_{family_filter}.csv",
                        mime="text/csv",
                    )
            else:
                st.warning(f"No sales data available for {store_filter} - {family_filter}")
    
    # Add footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Retail.AI - Data Explorer | <a href="http://localhost:8000" target="_blank">Landing Page</a> | <a href="http://localhost:8888" target="_blank">MLflow UI</a> | <a href="http://localhost:8002/docs" target="_blank">API Docs</a></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """
    Main function to render the dashboard.
    """
    # Add custom CSS
    local_css()
    
    # Add a special message if we see URL parameters for login
    if "token" in st.query_params and "username" in st.query_params:
        with st.spinner("Authenticating with provided token..."):
            time.sleep(0.5)  # Brief delay to show the spinner
            print("URL parameters detected. Attempting auto-login...")
    
    # Attempt auto-login from URL parameters if not already authenticated
    if not st.session_state.get("authenticated", False):
        auto_login = check_url_params()
        if auto_login:
            print("Auto-login successful! Refreshing page...")
            st.rerun()
    
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    # Render sidebar
    render_sidebar()
    
    # Render selected page based on authentication status
    if not st.session_state.get("authenticated", False):
        # Show login required banner
        st.markdown("""
        <div class="custom-banner">
            <i class="fas fa-lock"></i>
            <p>Please log in to access the dashboard.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show demo info
        st.markdown("""
        ## Retail Sales Forecasting Dashboard
        
        This dashboard provides forecasting analytics for retail sales data.
        
        ### Demo Credentials:
        - **Username:** admin
        - **Password:** admin
        
        Or
        
        - **Username:** johndoe
        - **Password:** secret
        """)
    else:
        # User is authenticated, render the selected page
        page = st.session_state.page
        
        if page == "Dashboard":
            display_dashboard()
        elif page == "Predictions":
            render_predictions()
        elif page == "Data Explorer":
            render_data_explorer()
        elif page == "Settings":
            render_settings()

if __name__ == "__main__":
    main() # timestamp: 1746712458.8450198