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
import jwt
from plotly.subplots import make_subplots

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Define constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# Configure page with dark theme
st.set_page_config(
    page_title="Store Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force dark mode
st.markdown("""
<style>
:root {
    --background-color: #0e1117;
    --secondary-background-color: #262730;
    --primary-color: #ff4b4b;
    --text-color: #fafafa;
    color-scheme: dark;
}

/* Override Streamlit default styles to force dark theme */
.stApp {
    background-color: #0e1117 !important;
    color: #fafafa !important;
}

.st-bq {
    background-color: #262730 !important;
}

.stTextInput input, .stNumberInput input, .stDateInput input {
    background-color: #262730 !important;
    color: #fafafa !important;
}

.stSelectbox select, .stMultiSelect select {
    background-color: #262730 !important;
    color: #fafafa !important;
}

/* Make metric values more visible in dark mode */
[data-testid="stMetricValue"] {
    color: #fafafa !important;
    font-weight: bold !important;
    font-size: 1.5rem !important;
}

/* Make warning messages more visible */
.stAlert {
    background-color: #473a11 !important;
    color: #fbcf61 !important;
}
</style>
""", unsafe_allow_html=True)

# Authentication functions
def login(username, password):
    """
    Login to the API and get a JWT token.
    """
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={
                "username": username,
                "password": password,
                "scope": "predictions:read"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def decode_token(token):
    """
    Decode the JWT token to get user info.
    """
    try:
        # This is just for display, no verification needed
        return jwt.decode(token, options={"verify_signature": False})
    except Exception as e:
        st.error(f"Error decoding token: {str(e)}")
        return {}

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
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{API_URL}/metrics_summary",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if data is mock and show warning
            if data.get("is_mock_data", False):
                message = data.get("message", "WARNING: Using simulated data")
                st.warning(message, icon="⚠️")
                
            return data
        else:
            st.error(f"Failed to fetch metrics: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error fetching metrics: {str(e)}")
        return None

# UI components
def render_sidebar():
    """
    Render the sidebar elements.
    """
    st.sidebar.title("Navigation")
    
    # Check for authentication
    if "token" not in st.session_state:
        st.sidebar.subheader("Login")
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username and password:
                    auth_response = login(username, password)
                    if auth_response:
                        st.session_state.token = auth_response["access_token"]
                        st.session_state.token_type = auth_response["token_type"]
                        st.session_state.user_info = decode_token(auth_response["access_token"])
                        st.success("Login successful!")
                        st.rerun()
                else:
                    st.error("Please enter both username and password.")
        
        # Demo credentials
        st.sidebar.markdown("---")
        st.sidebar.subheader("Demo Credentials")
        st.sidebar.markdown("Username: johndoe")
        st.sidebar.markdown("Password: secret")
        st.sidebar.markdown("---")
        st.sidebar.markdown("Username: admin")
        st.sidebar.markdown("Password: admin")
        
    else:
        # User info
        st.sidebar.subheader("User Info")
        user_info = st.session_state.user_info
        st.sidebar.markdown(f"**Username:** {user_info.get('sub', 'Unknown')}")
        scopes = user_info.get("scopes", [])
        if scopes:
            st.sidebar.markdown("**Permissions:**")
            for scope in scopes:
                st.sidebar.markdown(f"- {scope}")
        
        # Page navigation
        st.sidebar.markdown("---")
        st.sidebar.subheader("Pages")
        page = st.sidebar.radio(
            "Select Page",
            ["Dashboard", "Predictions", "Model Insights", "Settings"]
        )
        st.session_state.page = page
        
        # Logout
        st.sidebar.markdown("---")
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # API Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Status")
    health = get_health()
    if health["status"] == "healthy":
        st.sidebar.success("API is healthy")
    else:
        st.sidebar.error(f"API is not available: {health.get('message', 'Unknown error')}")
    
    # Links
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resources")
    st.sidebar.markdown("[MLflow Dashboard](%s)" % MLFLOW_URL)
    st.sidebar.markdown("[API Documentation](%s/docs)" % API_URL)

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
    st.header("Store Sales Forecast Dashboard")
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # Get metrics from API
        metrics = get_metric_summary(st.session_state.token)
        
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
    except Exception as e:
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

def render_predictions():
    """
    Render the predictions page using real data from the API.
    """
    st.title("Sales Predictions")
    
    st.markdown("""
    Use this page to get sales predictions for specific stores, product families, and dates.
    """)
    
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
    with st.expander("Prediction Options"):
        enable_debug = st.checkbox("Show Debug Information", value=True)
    
    # Form for predictions
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
        
        submitted = st.form_submit_button("Get Prediction")
    
    # Process form submission
    if submitted:
        if "token" in st.session_state:
            with st.spinner("Getting prediction..."):
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
                    # Show prediction results
                    st.success("Prediction request successful!")
                    
                    # Check if this is a fallback prediction
                    is_fallback = prediction.get('is_fallback', False)
                    saved_to_db = prediction.get('saved_to_db', False)
                    
                    # Format prediction
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Verifica qual chave está disponível
                        prediction_value = None
                        if 'prediction' in prediction:
                            prediction_value = prediction['prediction']
                        elif 'predicted_sales' in prediction:
                            prediction_value = prediction['predicted_sales']
                        else:
                            st.error("Prediction data format is invalid. Missing prediction value.")
                            prediction_value = 0
                            
                        # Estiliza de forma diferente caso seja fallback
                        if is_fallback:
                            message = prediction.get('message', 'WARNING: Using simulated prediction (fallback)')
                            st.warning(message, icon="⚠️")
                            
                            st.markdown(f"""
                            <div style="padding: 10px; background-color: #fff3cd; border-radius: 5px; border: 1px solid #ffeeba;">
                                <h3 style="color: #856404; margin: 0;">Predicted Sales (SIMULATED)</h3>
                                <p style="font-size: 24px; font-weight: bold; margin: 0; color: #856404;">${prediction_value:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show error details if debug is enabled
                            if enable_debug and 'error' in prediction:
                                st.error(f"Model Error: {prediction['error']}")
                        else:
                            st.success("REAL MODEL PREDICTION", icon="✅")
                            st.metric(
                                "Predicted Sales (REAL)", 
                                f"${prediction_value:.2f}",
                                delta=None
                            )
                            
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
                                st.subheader("Prediction Explanation")
                                
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
                                        height=500
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
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

def render_model_insights():
    """
    Render the model insights page.
    """
    st.title("Model Insights")
    
    st.markdown("""
    This page provides insights into the model performance and metrics.
    """)
    
    # Accuracy Verification Section
    st.header("Forecast Accuracy Verification")
    
    # Button to verify accuracy calculations
    if st.button("Verify Forecast Accuracy Calculation"):
        try:
            with st.spinner("Retrieving detailed accuracy metrics..."):
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.get(f"{API_URL}/metrics_accuracy_check", headers=headers)
                
                if response.status_code == 200:
                    accuracy_data = response.json()
                    summary = accuracy_data.get("summary", {})
                    details = accuracy_data.get("detailed_results", [])
                    
                    # Display the summary metrics
                    st.subheader("Accuracy Summary")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        data_points = summary.get("count", 0)
                        st.metric("Data Points Used", f"{data_points}")
                        st.metric("MAPE", f"{summary.get('mape', 0):.2f}%")
                        st.metric("Forecast Accuracy", f"{summary.get('forecast_accuracy', 0):.2f}%")
                        
                    with col2:
                        st.metric("Mean Error", f"{summary.get('mean_error', 0):.2f}")
                        st.metric("MAE", f"{summary.get('mae', 0):.2f}")
                        st.metric("RMSE", f"{summary.get('rmse', 0):.2f}")
                        
                    st.write(f"**Calculation Method:** {summary.get('calculation_method', 'N/A')}")
                    
                    # Display the detailed results
                    if details:
                        st.subheader("Sample Data Points")
                        df = pd.DataFrame(details[:10])  # Show only first 10 for simplicity
                        
                        # Reorder columns for better presentation
                        if not df.empty and set(['store', 'family', 'date', 'predicted', 'actual', 'error', 'percentage_error']).issubset(df.columns):
                            df = df[['store', 'family', 'date', 'predicted', 'actual', 'error', 'percentage_error']]
                        
                        st.dataframe(df)
                        
                        # Create error distribution chart
                        if len(details) >= 5:
                            st.subheader("Error Distribution")
                            error_df = pd.DataFrame(details)
                            
                            fig = px.histogram(
                                error_df, 
                                x="percentage_error",
                                nbins=20,
                                title="Percentage Error Distribution",
                                labels={"percentage_error": "Percentage Error (%)"}
                            )
                            
                            fig.update_layout(
                                template='plotly_dark',
                                height=400,
                                margin=dict(l=10, r=10, t=30, b=10)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create scatter plot of predicted vs actual
                            st.subheader("Predicted vs Actual")
                            fig = px.scatter(
                                error_df,
                                x="actual",
                                y="predicted",
                                hover_data=["store", "family", "date", "error", "percentage_error"],
                                title="Predicted vs Actual Values",
                                labels={"actual": "Actual Sales", "predicted": "Predicted Sales"}
                            )
                            
                            # Add perfect prediction line
                            max_val = max(error_df["actual"].max(), error_df["predicted"].max())
                            fig.add_shape(
                                type="line",
                                x0=0, y0=0,
                                x1=max_val, y1=max_val,
                                line=dict(color="green", width=2, dash="dash")
                            )
                            
                            fig.update_layout(
                                template='plotly_dark',
                                height=500,
                                margin=dict(l=10, r=10, t=30, b=10)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No detailed data points available for verification.")
                else:
                    st.error(f"Failed to fetch accuracy metrics: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error verifying accuracy: {str(e)}")

    # Recent Predictions Section
    st.header("Recent Predictions")
    
    # Button to show recent predictions with accuracy
    if st.button("Show Recent Predictions"):
        try:
            with st.spinner("Retrieving recent predictions..."):
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                response = requests.get(f"{API_URL}/recent_predictions", headers=headers)
                
                if response.status_code == 200:
                    predictions = response.json()
                    
                    if predictions:
                        # Convert to DataFrame for easier display
                        df = pd.DataFrame([
                            {
                                "Store": p['store_nbr'],
                                "Family": p['family'],
                                "Date": p['date'],
                                "Predicted": p['predicted_sales'],
                                "Actual": p['actual_sales'] if p['actual_sales'] is not None else "N/A",
                                "Error %": p['accuracy_metrics']['percentage_error'] if p['accuracy_metrics'] else "N/A",
                                "Accuracy": f"{p['accuracy_metrics']['accuracy']:.2f}%" if p['accuracy_metrics'] else "N/A"
                            } 
                            for p in predictions
                        ])
                        
                        st.dataframe(df)
                        
                        # Calculate overall accuracy from available data points
                        valid_predictions = [p for p in predictions if p['accuracy_metrics']]
                        if valid_predictions:
                            avg_accuracy = sum(p['accuracy_metrics']['accuracy'] for p in valid_predictions) / len(valid_predictions)
                            st.metric("Average Accuracy (Recent Predictions)", f"{avg_accuracy:.2f}%")
                        else:
                            st.warning("No validation data available for recent predictions.")
                    else:
                        st.warning("No recent predictions found.")
                else:
                    st.error(f"Failed to fetch recent predictions: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error retrieving recent predictions: {str(e)}")
            
    # Existing content
    st.header("Model Metrics")
    
    # Model selection
    models = ["LightGBM (Production)", "Prophet (Staging)", "ARIMA (Development)"]
    selected_model = st.selectbox("Select Model", models)
    
    # Load model metrics
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(
            f"{API_URL}/model_metrics?model_name={selected_model}",
            headers=headers
        )
        
        if response.status_code == 200:
            model_metrics = response.json()
            
            # Model performance metrics
            st.subheader("Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("RMSE", f"{model_metrics.get('rmse', 0):.2f}", model_metrics.get('rmse_change'))
            col2.metric("MAE", f"{model_metrics.get('mae', 0):.2f}", model_metrics.get('mae_change'))
            col3.metric("MAPE", f"{model_metrics.get('mape', 0):.1f}%", model_metrics.get('mape_change'))
            col4.metric("R²", f"{model_metrics.get('r2', 0):.2f}", model_metrics.get('r2_change'))
        else:
            st.warning("Unable to fetch model metrics from API.")
            # Don't show metrics if unable to fetch real ones
    except Exception as e:
        st.warning(f"Error fetching model metrics: {str(e)}")
    
    # Get real feature importance from API
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        feat_imp_response = requests.get(
            f"{API_URL}/feature_importance",
            params={"model_name": selected_model},
            headers=headers
        )
        
        if feat_imp_response.status_code == 200:
            feature_imp_data = feat_imp_response.json()
            
            # Feature importance
            st.subheader("Feature Importance")
            
            # Create dataframe for chart
            feat_imp = pd.DataFrame(feature_imp_data)
            
            # Create horizontal bar chart
            fig = px.bar(
                feat_imp.sort_values('importance', ascending=True), 
                y='feature', 
                x='importance',
                orientation='h',
                title="Feature Importance",
                height=500,
                color='importance',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to fetch feature importance data from API.")
    except Exception as e:
        st.warning(f"Error fetching feature importance: {str(e)}")
    
    # Show model drift
    st.subheader("Model Drift")
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        drift_response = requests.get(
            f"{API_URL}/model_drift",
            params={"model_name": selected_model, "days": 7},
            headers=headers
        )
        
        if drift_response.status_code == 200:
            drift_data = drift_response.json()
            
            # Check if we have data
            if drift_data and "dates" in drift_data and len(drift_data["dates"]) > 0:
                # Create a DataFrame for plotting
                drift_df = pd.DataFrame({
                    "date": drift_data["dates"],
                    "rmse": drift_data["rmse"],
                    "mae": drift_data["mae"],
                    "drift_score": drift_data["drift_score"]
                })
                
                # Plot drift metrics
                fig = make_subplots(rows=2, cols=1, 
                                    subplot_titles=("Model Metrics Over Time", "Drift Score"),
                                    vertical_spacing=0.1)
                
                # Add RMSE trace
                fig.add_trace(
                    go.Scatter(x=drift_df["date"], y=drift_df["rmse"], name="RMSE"),
                    row=1, col=1
                )
                
                # Add MAE trace
                fig.add_trace(
                    go.Scatter(x=drift_df["date"], y=drift_df["mae"], name="MAE"),
                    row=1, col=1
                )
                
                # Add Drift Score trace
                fig.add_trace(
                    go.Bar(x=drift_df["date"], y=drift_df["drift_score"], name="Drift Score (%)"),
                    row=2, col=1
                )
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No drift data available for the selected model.")
        else:
            st.warning(f"Failed to fetch model drift data: {drift_response.status_code} - {drift_response.text}")
    except Exception as e:
        st.warning(f"Error fetching model drift data: {str(e)}")
        st.info("Model drift monitoring is not available at this time.")

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

def main():
    """
    Main function to run the dashboard.
    """
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    
    # Render sidebar
    render_sidebar()
    
    # Render selected page
    if "token" not in st.session_state:
        # Show welcome screen if not logged in
        st.title("Welcome to Store Sales Forecast Dashboard")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Please login to access the dashboard
            
            This dashboard provides insights into store sales forecasts, allowing you to:
            
            - View sales trends across different stores and product families
            - Make predictions for specific stores and dates
            - Understand model performance and feature importance
            - Monitor for model drift and data quality issues
            
            Login using the form in the sidebar to get started.
            """)
        
        with col2:
            # Sample image or icon
            st.image("https://cdn-icons-png.flaticon.com/512/6295/6295417.png", width=200)
    
    else:
        # Render the appropriate page
        if st.session_state.page == "Dashboard":
            display_dashboard()
        elif st.session_state.page == "Predictions":
            render_predictions()
        elif st.session_state.page == "Model Insights":
            render_model_insights()
        elif st.session_state.page == "Settings":
            render_settings()

if __name__ == "__main__":
    main() # timestamp: 1746712458.8450198