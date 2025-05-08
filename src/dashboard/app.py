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

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Define constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

# Configure page
st.set_page_config(
    page_title="Store Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            return response.json()
        else:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
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
        response = requests.get(f"{API_URL}/stores")
        if response.status_code == 200:
            return response.json()
        else:
            st.warning("Failed to fetch stores from API, using cached values.")
            # Fallback
            return list(range(1, 55))
    except Exception as e:
        st.warning(f"Error fetching stores: {str(e)}")
        # Fallback
        return list(range(1, 55))

def get_families():
    """
    Get a list of product families from the API.
    """
    try:
        response = requests.get(f"{API_URL}/families")
        if response.status_code == 200:
            return response.json()
        else:
            st.warning("Failed to fetch product families from API, using cached values.")
            # Fallback
            return [
                "BEVERAGES", "BREAD/BAKERY", "CLEANING", "DAIRY", "DELI", 
                "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE", 
                "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES", 
                "LADIESWEAR", "LIQUOR,WINE,BEER", "MEATS", "PERSONAL CARE", 
                "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY", 
                "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"
            ]
    except Exception as e:
        st.warning(f"Error fetching product families: {str(e)}")
        # Fallback
        return [
            "BEVERAGES", "BREAD/BAKERY", "CLEANING", "DAIRY", "DELI", 
            "EGGS", "FROZEN FOODS", "GROCERY I", "GROCERY II", "HARDWARE", 
            "HOME AND KITCHEN I", "HOME AND KITCHEN II", "HOME APPLIANCES", 
            "LADIESWEAR", "LIQUOR,WINE,BEER", "MEATS", "PERSONAL CARE", 
            "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY", 
            "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD"
        ]

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
            data = response.json()
            return pd.DataFrame(data)
        else:
            st.warning("Failed to fetch sales data from API, using generated data.")
            # Fallback to generated data
            return get_demo_sales_data()
    except Exception as e:
        st.warning(f"Error fetching sales data: {str(e)}")
        # Fallback to generated data
        return get_demo_sales_data()

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
            return response.json()
        else:
            st.warning("Failed to fetch metrics from API, using demo data.")
            # Return fallback data
            return {
                "total_stores": 54,
                "total_families": 24,
                "avg_sales": 1234.56,
                "forecast_accuracy": 87.5
            }
    except Exception as e:
        st.warning(f"Error fetching metrics: {str(e)}")
        # Return fallback data
        return {
            "total_stores": 54,
            "total_families": 24,
            "avg_sales": 1234.56,
            "forecast_accuracy": 87.5
        }

# Demo data functions for development and fallback
def get_demo_sales_data():
    """
    Generate demo sales data when API is not available.
    """
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create seasonal patterns
    sales = np.sin(np.arange(len(dates)) * 0.1) * 20 + np.random.randn(len(dates)) * 5 + 100
    
    # Add weekly patterns
    for i in range(len(dates)):
        if dates[i].dayofweek >= 5:  # Weekend
            sales[i] *= 1.5
    
    # Add special promotions spikes
    promo_days = np.random.choice(len(dates), 30, replace=False)
    sales[promo_days] *= 2
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'is_promotion': np.zeros(len(dates))
    })
    
    df.loc[promo_days, 'is_promotion'] = 1
    
    return df

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

def render_dashboard():
    """
    Render the main dashboard.
    """
    st.title("Store Sales Forecast Dashboard")
    
    # Get real metrics if authenticated, otherwise use demo data
    if "token" in st.session_state:
        metrics = get_metric_summary(st.session_state.token)
    else:
        metrics = {
            "total_stores": 54,
            "total_families": 24,
            "avg_sales": 1234.56,
            "forecast_accuracy": 87.5
        }
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Stores", f"{metrics['total_stores']}", "+3")
    col2.metric("Total Products", f"{metrics['total_families']} families", "")
    col3.metric("Average Sales", f"${metrics['avg_sales']:.2f}", "+5.2%")
    col4.metric("Forecast Accuracy", f"{metrics['forecast_accuracy']:.1f}%", "+2.3%")
    
    # Filter controls
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        store = st.selectbox("Store", get_stores())
    with col2:
        family = st.selectbox("Product Family", get_families())
    with col3:
        days = st.slider("Days to Display", 30, 365, 90)
    
    # Get sales data - real if authenticated, demo otherwise
    if "token" in st.session_state:
        sales_data = get_sales_data(st.session_state.token, store, family, days)
    else:
        sales_data = get_demo_sales_data()
    
    # Time series chart
    st.subheader("Sales Trends")
    
    # Filter data based on user selection
    filtered_data = sales_data.iloc[-days:]
    
    # Create a time series chart
    fig = px.line(
        filtered_data, 
        x='date', 
        y='sales',
        title=f"Sales Trend for Store {store} - {family}",
        height=400
    )
    
    # Add markers for promotions
    if 'is_promotion' in filtered_data.columns:
        promo_data = filtered_data[filtered_data['is_promotion'] == 1]
        fig.add_trace(
            go.Scatter(
                x=promo_data['date'],
                y=promo_data['sales'],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Promotions'
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Store comparison
    st.subheader("Store Comparison")
    
    # Create demo comparison data
    stores = np.random.choice(get_stores(), 10, replace=False)
    store_perf = pd.DataFrame({
        'store': [f"Store {s}" for s in stores],
        'sales': np.random.randint(1000, 5000, size=10),
        'forecast_accuracy': np.random.uniform(0.7, 0.95, size=10)
    })
    
    # Create bar chart
    fig = px.bar(
        store_perf, 
        x='store', 
        y='sales',
        color='forecast_accuracy',
        color_continuous_scale='Viridis',
        title="Sales by Store",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Product family performance
    st.subheader("Product Family Performance")
    
    # Create demo product performance data
    families = np.random.choice(get_families(), 10, replace=False)
    family_perf = pd.DataFrame({
        'family': families,
        'sales': np.random.randint(1000, 8000, size=10),
        'growth': np.random.uniform(-0.2, 0.3, size=10)
    })
    
    # Create horizontal bar chart
    fig = px.bar(
        family_perf.sort_values('sales', ascending=True), 
        y='family', 
        x='sales',
        color='growth',
        color_continuous_scale='RdYlGn',
        title="Sales by Product Family",
        orientation='h',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_predictions():
    """
    Render the predictions page.
    """
    st.title("Sales Predictions")
    
    st.markdown("""
    Use this page to get sales predictions for specific stores, product families, and dates.
    """)
    
    # Form for predictions
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            store_nbr = st.selectbox("Store Number", get_stores())
        
        with col2:
            family = st.selectbox("Product Family", get_families())
        
        with col3:
            onpromotion = st.checkbox("On Promotion")
        
        date = st.date_input("Prediction Date", datetime.now().date() + timedelta(days=1))
        
        submitted = st.form_submit_button("Get Prediction")
    
    # Process form submission
    if submitted:
        if "token" in st.session_state:
            # Make prediction request
            prediction = get_prediction(
                st.session_state.token,
                store_nbr,
                family,
                onpromotion,
                date.strftime("%Y-%m-%d")
            )
            
            if prediction:
                # Show prediction results
                st.success("Prediction successful!")
                
                # Format prediction
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicted Sales", 
                        f"${prediction['predicted_sales']:.2f}",
                        delta=None
                    )
                
                # Save prediction ID for explanation
                prediction_id = f"{store_nbr}-{family}-{date}"
                
                # Get explanation
                explanation = get_explanation(
                    st.session_state.token,
                    prediction_id,
                    store_nbr,
                    family,
                    onpromotion,
                    date.strftime("%Y-%m-%d")
                )
                
                if explanation:
                    st.subheader("Prediction Explanation")
                    
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
                
                # Historical context
                st.subheader("Historical Context")
                st.markdown("Here's how this prediction compares to historical sales:")
                
                # Create demo historical data
                hist_dates = pd.date_range(end=date - timedelta(days=1), periods=60, freq='D')
                hist_sales = np.sin(np.arange(len(hist_dates)) * 0.3) * 10 + np.random.randn(len(hist_dates)) * 5 + 50
                
                # Add some randomness for weekends
                for i in range(len(hist_dates)):
                    if hist_dates[i].dayofweek >= 5:  # Weekend
                        hist_sales[i] *= 1.3
                
                hist_df = pd.DataFrame({
                    'date': hist_dates,
                    'sales': hist_sales
                })
                
                # Add the prediction point
                future_df = pd.DataFrame({
                    'date': [date],
                    'sales': [prediction['predicted_sales']]
                })
                
                # Plot
                fig = px.line(
                    hist_df, 
                    x='date', 
                    y='sales',
                    title=f"Historical Sales and Prediction for Store {store_nbr} - {family}",
                    height=400
                )
                
                # Add the prediction point
                fig.add_trace(
                    go.Scatter(
                        x=future_df['date'],
                        y=future_df['sales'],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name='Prediction'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.error("Please login to make predictions.")

def render_model_insights():
    """
    Render the model insights page.
    """
    st.title("Model Insights")
    
    st.markdown("""
    This page provides insights into the model performance and feature importance.
    """)
    
    # Model selection
    st.subheader("Model Selection")
    models = ["LightGBM (Production)", "Prophet (Staging)", "ARIMA (Development)"]
    selected_model = st.selectbox("Select Model", models)
    
    # Model performance metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Demo metrics based on selected model
    if selected_model == "LightGBM (Production)":
        col1.metric("RMSE", "245.32", "-12.5%")
        col2.metric("MAE", "187.44", "-8.2%")
        col3.metric("MAPE", "14.3%", "-5.1%")
        col4.metric("R²", "0.87", "+0.04")
    elif selected_model == "Prophet (Staging)":
        col1.metric("RMSE", "267.89", "+9.2%")
        col2.metric("MAE", "201.35", "+7.4%")
        col3.metric("MAPE", "16.2%", "+13.3%")
        col4.metric("R²", "0.82", "-0.06")
    else:  # ARIMA
        col1.metric("RMSE", "295.67", "+20.5%")
        col2.metric("MAE", "234.12", "+24.9%")
        col3.metric("MAPE", "18.7%", "+30.8%")
        col4.metric("R²", "0.75", "-0.14")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Demo feature importance
    features = [
        "onpromotion", "day_of_week", "store_nbr", "month", 
        "day_of_month", "is_weekend", "family_GROCERY I", 
        "family_BEVERAGES", "family_PRODUCE", "family_CLEANING"
    ]
    importance = np.random.uniform(0.01, 0.25, size=len(features))
    importance = importance / importance.sum()
    
    # Sort by importance
    feat_imp = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        feat_imp, 
        y='feature', 
        x='importance',
        orientation='h',
        title="Feature Importance",
        height=500,
        color='importance',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model drift
    st.subheader("Model Drift Monitoring")
    
    # Time period selection
    time_period = st.radio("Select Time Period", ["Last Week", "Last Month", "Last Quarter"], horizontal=True)
    
    # Demo drift data
    days = 7
    if time_period == "Last Month":
        days = 30
    elif time_period == "Last Quarter":
        days = 90
    
    dates = pd.date_range(end=datetime.now().date(), periods=days, freq='D')
    
    # Create drift metrics
    drift_df = pd.DataFrame({
        'date': dates,
        'performance_score': np.random.uniform(0.7, 0.95, size=days),
        'data_drift_score': np.random.uniform(0.0, 0.3, size=days)
    })
    
    # Add a trend
    t = np.arange(days) / days
    drift_df['performance_score'] = drift_df['performance_score'] - t * 0.1
    drift_df['data_drift_score'] = drift_df['data_drift_score'] + t * 0.15
    
    # Plot
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=drift_df['date'],
            y=drift_df['performance_score'],
            mode='lines',
            name='Performance',
            line=dict(color='blue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=drift_df['date'],
            y=drift_df['data_drift_score'],
            mode='lines',
            name='Data Drift',
            line=dict(color='red')
        )
    )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=drift_df['date'].min(),
        y0=0.2,
        x1=drift_df['date'].max(),
        y1=0.2,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        )
    )
    
    fig.update_layout(
        title="Model Drift Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert based on drift
    if drift_df['data_drift_score'].iloc[-1] > 0.2:
        st.warning("⚠️ Data drift detected! Model retraining recommended.")
    else:
        st.success("✅ No significant data drift detected. Model is performing well.")

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
            render_dashboard()
        elif st.session_state.page == "Predictions":
            render_predictions()
        elif st.session_state.page == "Model Insights":
            render_model_insights()
        elif st.session_state.page == "Settings":
            render_settings()

if __name__ == "__main__":
    main() 