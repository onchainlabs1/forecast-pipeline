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
            result = response.json()
            # Garantir que sempre temos a chave 'prediction'
            if 'predicted_sales' in result and 'prediction' not in result:
                result['prediction'] = result['predicted_sales']
            return result
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
            st.error(f"Failed to fetch stores: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching stores: {str(e)}")
        return []

def get_families():
    """
    Get a list of product families from the API.
    """
    try:
        response = requests.get(f"{API_URL}/families")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch product families: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching product families: {str(e)}")
        return []

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
            return response.json()
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

def render_dashboard():
    """
    Render the main dashboard with real data.
    """
    st.title("Store Sales Forecast Dashboard")
    
    # Check API health
    health = get_health()
    if health["status"] != "healthy":
        st.error(f"API is not available: {health.get('message', 'Unknown error')}")
        st.warning("Please make sure the API is running before using the dashboard.")
        st.stop()
    
    # Get real metrics if authenticated
    metrics = None
    if "token" in st.session_state:
        metrics = get_metric_summary(st.session_state.token)
    
    if not metrics:
        st.error("Unable to load metrics data. Please log in and ensure API connection is working.")
        st.stop()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Stores", f"{metrics['total_stores']}", None)
    col2.metric("Total Products", f"{metrics['total_families']} families", None)
    col3.metric("Average Sales", f"${metrics['avg_sales']:.2f}", None)
    col4.metric("Forecast Accuracy", f"{metrics['forecast_accuracy']:.1f}%", None)
    
    # Get stores and families from API
    stores = get_stores()
    families = get_families()
    
    if not stores or not families:
        st.error("Failed to load store or product family data from API.")
        st.stop()
    
    # Filter controls
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    # Default values we know that work
    default_store_index = 0
    default_family_index = 0
    
    # Stores and families we know have data
    good_combinations = [
        {"store": 1, "family": "PRODUCE"},
        {"store": 1, "family": "FROZEN FOODS"},
        {"store": 1, "family": "GROCERY II"},
        {"store": 1, "family": "LIQUOR,WINE,BEER"},
        {"store": 1, "family": "HOME APPLIANCES"}
    ]
    
    # Try to find default values in the stores and families lists
    if stores and families:
        # Look for store 1 in the stores list
        if 1 in stores:
            default_store_index = stores.index(1)
            
        # Look for "PRODUCE" in the families list
        for good_family in ["PRODUCE", "FROZEN FOODS", "GROCERY II", "LIQUOR,WINE,BEER", "HOME APPLIANCES"]:
            if good_family in families:
                default_family_index = families.index(good_family)
                break
    
    with col1:
        store = st.selectbox("Store", stores, index=default_store_index) if stores else st.text_input("Store Number")
    with col2:
        family = st.selectbox("Product Family", families, index=default_family_index) if families else st.text_input("Product Family")
    with col3:
        days = st.slider("Days to Display", 30, 365, 90)
    
    # Get sales data - must have auth token
    if "token" not in st.session_state:
        st.error("Please log in to view sales data.")
        st.stop()
    
    # Try to get data for the selected combination
    sales_data = get_sales_data(st.session_state.token, store, family, days)
    
    # If no data found, try the known good combinations
    if sales_data.empty:
        st.warning(f"No data found for Store {store}, Family {family}. Trying to find combinations with data...")
        
        found_data = False
        for combo in good_combinations:
            st.info(f"Trying Store {combo['store']}, Family {combo['family']}...")
            test_data = get_sales_data(st.session_state.token, combo['store'], combo['family'], days)
            
            if not test_data.empty:
                st.success(f"Data found for Store {combo['store']}, Family {combo['family']}!")
                sales_data = test_data
                store = combo['store']
                family = combo['family']
                found_data = True
                break
        
        # If still no data found, show error
        if not found_data:
            st.error("Could not find data for any known combination.")
            
            # Suggest combinations that should work
            st.markdown("### Suggested Combinations with Data")
            st.markdown("The following combinations should have data. Verify that the families are exactly as shown below:")
            
            for combo in good_combinations:
                st.markdown(f"- Store **{combo['store']}** with family **{combo['family']}**")
            
            st.stop()
    
    # Time series chart
    st.subheader("Sales Trends")
    
    # Create a time series chart
    fig = px.line(
        sales_data, 
        x='date', 
        y='sales',
        title=f"Sales Trend for Store {store} - {family}",
        height=400
    )
    
    # Add markers for promotions if available
    if 'is_promotion' in sales_data.columns:
        promo_data = sales_data[sales_data['is_promotion'] == 1]
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
    
    # Get real comparison data from API
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        compare_response = requests.get(
            f"{API_URL}/store_comparison",
            headers=headers
        )
        
        if compare_response.status_code == 200:
            store_perf = pd.DataFrame(compare_response.json())
            
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
            
            st.subheader("Store Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to load store comparison data.")
    except Exception as e:
        st.warning(f"Error loading store comparison: {str(e)}")
    
    # Get real product family performance from API  
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        family_response = requests.get(
            f"{API_URL}/family_performance",
            headers=headers
        )
        
        if family_response.status_code == 200:
            family_perf = pd.DataFrame(family_response.json())
            
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
            
            st.subheader("Product Family Performance")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to load product family performance data.")
    except Exception as e:
        st.warning(f"Error loading product family performance: {str(e)}")

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
    
    if not stores or not families:
        st.error("Failed to load store or product family data from API.")
        st.stop()
    
    # Form for predictions
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # Default values we know that work
        default_store_index = 0
        default_family_index = 0
        
        # Tenta encontrar os índices dos valores default nos stores e families
        if stores and families:
            # Procura store 1 na lista de lojas
            if 1 in stores:
                default_store_index = stores.index(1)
                
            # Procura "PRODUCE" na lista de famílias
            for good_family in ["PRODUCE", "FROZEN FOODS", "GROCERY II", "LIQUOR,WINE,BEER", "HOME APPLIANCES"]:
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
                        # Verifica qual chave está disponível
                        prediction_value = None
                        if 'prediction' in prediction:
                            prediction_value = prediction['prediction']
                        elif 'predicted_sales' in prediction:
                            prediction_value = prediction['predicted_sales']
                        else:
                            st.error("Prediction data format is invalid. Missing prediction value.")
                            prediction_value = 0
                            
                        st.metric(
                            "Predicted Sales", 
                            f"${prediction_value:.2f}",
                            delta=None
                        )
                    
                    # Save prediction ID for explanation
                    prediction_id = prediction.get('prediction_id', f"{store_nbr}-{family}-{date}")
                    
                    # Get explanation
                    with st.spinner("Generating explanation..."):
                        try:
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
                                else:
                                    st.warning("Explanation available but no feature contributions found.")
                            else:
                                st.warning("Unable to generate explanation for this prediction.")
                        except Exception as exp_error:
                            st.error(f"Error generating explanation: {str(exp_error)}")
                            st.warning("Unable to generate explanation for this prediction. Model may not support explainability.")
                    
                    # Historical context from real API data
                    st.subheader("Historical Context")
                    st.markdown("Here's how this prediction compares to historical sales:")
                    
                    # Get real historical data
                    try:
                        headers = {"Authorization": f"Bearer {st.session_state.token}"}
                        hist_response = requests.get(
                            f"{API_URL}/historical_sales",
                            params={
                                "store_nbr": store_nbr,
                                "family": family,
                                "days": 60
                            },
                            headers=headers
                        )
                        
                        if hist_response.status_code == 200:
                            hist_data = hist_response.json()
                            hist_df = pd.DataFrame(hist_data)
                            
                            # Add the prediction point
                            future_df = pd.DataFrame({
                                'date': [date],
                                'sales': [prediction['prediction']]  # Usa o mesmo valor verificado anteriormente
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
                            st.warning("Unable to load historical data for comparison.")
                    except Exception as e:
                        st.warning(f"Error loading historical data: {str(e)}")
        else:
            st.error("Please login to make predictions.")

def render_model_insights():
    """
    Render the model insights page with real data from the API.
    """
    st.title("Model Insights")
    
    st.markdown("""
    This page provides insights into the model performance and feature importance.
    """)
    
    # Check API health
    health = get_health()
    if health["status"] != "healthy":
        st.error(f"API is not available: {health.get('message', 'Unknown error')}")
        st.warning("Please make sure the API is running before viewing model insights.")
        st.stop()
    
    if "token" not in st.session_state:
        st.error("Please login to view model insights.")
        st.stop()
    
    # Get model list from API
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        models_response = requests.get(
            f"{API_URL}/models",
            headers=headers
        )
        
        if models_response.status_code == 200:
            models_data = models_response.json()
            models = [model["name"] for model in models_data]
        else:
            st.warning("Unable to fetch model list from API, using default values.")
            models = ["LightGBM (Production)", "Prophet (Staging)", "ARIMA (Development)"]
    except Exception as e:
        st.warning(f"Error fetching model list: {str(e)}")
        models = ["LightGBM (Production)", "Prophet (Staging)", "ARIMA (Development)"]
    
    # Model selection
    st.subheader("Model Selection")
    selected_model = st.selectbox("Select Model", models)
    
    # Get real model performance metrics from API
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        metrics_response = requests.get(
            f"{API_URL}/model_metrics",
            params={"model_name": selected_model},
            headers=headers
        )
        
        if metrics_response.status_code == 200:
            model_metrics = metrics_response.json()
            
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
            render_dashboard()
        elif st.session_state.page == "Predictions":
            render_predictions()
        elif st.session_state.page == "Model Insights":
            render_model_insights()
        elif st.session_state.page == "Settings":
            render_settings()

if __name__ == "__main__":
    main() # timestamp: 1746712458.8450198