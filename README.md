# RetailPro AI: Sales Forecasting Platform

<div align="center">
  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.21.2-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  <img src="reports/dashboard_screenshot.png" alt="RetailPro AI Dashboard" width="800"/>
</p>

## 🎯 Business Problem

Retail chains face significant challenges in inventory management, with both overstocking and understocking leading to substantial financial losses:

- **Overstocking** ties up capital and leads to markdowns and waste
- **Understocking** results in lost sales and reduced customer satisfaction
- **Seasonal variations** complicate manual forecasting approaches
- **Promotion planning** requires accurate sales predictions

RetailPro AI addresses these challenges by providing precise, store-level forecasts across all product families, enabling retailers to optimize inventory levels, plan promotions effectively, and maximize profitability.

## 💡 Solution

Our platform delivers an end-to-end forecasting solution that:

- **Predicts sales** with 80%+ accuracy for 54 stores across 34 product families
- **Visualizes trends** through an intuitive dashboard with customizable filters
- **Explains predictions** using explainable AI techniques to build trust
- **Monitors performance** with real-time drift detection and model metrics
- **Secures data** with robust JWT authentication and role-based access

## 🔍 Key Features

<table>
  <tr>
    <td width="33%">
      <h3 align="center">📊 Interactive Dashboard</h3>
      <p align="center">Real-time visualization of sales trends with powerful filtering and drill-down capabilities</p>
    </td>
    <td width="33%">
      <h3 align="center">🔮 ML Predictions</h3>
      <p align="center">Generate accurate forecasts using advanced machine learning models with proven accuracy</p>
    </td>
    <td width="33%">
      <h3 align="center">📈 Performance Analysis</h3>
      <p align="center">Track forecast accuracy and model drift with automated monitoring and alerts</p>
    </td>
  </tr>
  <tr>
    <td width="33%">
      <h3 align="center">🔍 Model Insights</h3>
      <p align="center">Understand predictions with explainable AI and feature importance visualization</p>
    </td>
    <td width="33%">
      <h3 align="center">🔒 Enterprise Security</h3>
      <p align="center">JWT-based authentication with role-based access control and data encryption</p>
    </td>
    <td width="33%">
      <h3 align="center">⚙️ MLOps Integration</h3>
      <p align="center">Complete integration with MLflow for experiment tracking and model versioning</p>
    </td>
  </tr>
</table>

## 📊 Performance Metrics

<div align="center">
  
| Metric | Value | Description |
|--------|-------|-------------|
| **Forecast Accuracy** | 80.17% | Overall accuracy of predictions |
| **MAPE** | 19.83% | Mean Absolute Percentage Error |
| **MAE** | 46.16 | Mean Absolute Error |
| **RMSE** | 50.64 | Root Mean Square Error |

</div>

## 🛠️ Technical Architecture

Our platform follows a modern microservices architecture with three main components:

<p align="center">
  <img src="reports/architecture_diagram.png" alt="System Architecture" width="700"/>
</p>

```
┌───────────────────┐     ┌──────────────────┐     ┌────────────────┐
│                   │     │                  │     │                │
│ Streamlit         │─────▶ FastAPI          │─────▶ ML Models      │
│ Dashboard         │     │ Backend          │     │ & Predictions  │
│                   │     │                  │     │                │
└───────────────────┘     └──────────────────┘     └────────────────┘
                                   │                        │
                                   ▼                        ▼
                          ┌──────────────────┐     ┌────────────────┐
                          │                  │     │                │
                          │ Authentication   │     │ Feature        │
                          │ & Security       │     │ Engineering    │
                          │                  │     │                │
                          └──────────────────┘     └────────────────┘
                                                            │
                                                            ▼
                                                   ┌────────────────┐
                                                   │                │
                                                   │ Monitoring     │
                                                   │ & Metrics      │
                                                   │                │
                                                   └────────────────┘
```

## 🧰 Tech Stack

<div align="center">
  
### Frontend
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-00A4EF?style=for-the-badge&logoColor=white)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

### ML & Data Science
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-38cc77?style=for-the-badge&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-00B7EB?style=for-the-badge&logoColor=white)

### DevOps & Monitoring
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)

</div>

## 📂 Project Structure

```
mlproject/
├── src/
│   ├── api/              # FastAPI application for model serving
│   ├── dashboard/        # Streamlit interface with visualizations
│   ├── database/         # Database models and connection utilities
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # ML model definition, training and evaluation
│   ├── security/         # Authentication and authorization
│   └── utils/            # Shared utility functions
├── models/               # Serialized model artifacts
├── tests/                # Automated test suite
├── airflow/              # Airflow DAGs for scheduled tasks
├── monitoring/           # Monitoring and alerting components
├── docker-compose.yml    # Docker configuration
└── requirements.txt      # Python dependencies
```

## 🚀 Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mlproject.git
cd mlproject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow ui --port 8888 --host 0.0.0.0

# Run the API
python -m src.api.main

# In another terminal, start the dashboard
streamlit run src/dashboard/app.py
```

## 🔐 Authentication

The system uses JWT-based authentication with the following demo credentials:
- **Username**: johndoe
- **Password**: secret

Or admin access:
- **Username**: admin
- **Password**: admin

## 📸 Screenshots

<div align="center">
  <p float="left">
    <img src="reports/metrics_overview.png" width="400" />
    <img src="reports/store_comparison.png" width="400" /> 
  </p>
  <p float="left">
    <img src="reports/model_explanations.png" width="400" />
    <img src="reports/mlflow_tracking.png" width="400" />
  </p>
</div>

## 📄 License

MIT

## 📧 Contact

Your Name - your.email@example.com

## Running with Attractive Landing Page

This project now offers a modern and attractive landing page as the entry point to the sales forecasting dashboard. The landing page was designed with a user experience similar to professional websites, including animations, smooth transitions, and modern design with a dark theme.

### Landing Page Features

- **Modern Design**: Dark theme interface with smooth gradients and animations
- **Visual Presentation**: Displays key metrics (54 stores, 34 product families, etc.)
- **Seamless Integration**: Automatically connects to the Streamlit dashboard
- **Professional Experience**: Provides an impactful first impression

### How to Run

To start the complete system with the landing page and dashboard:

```bash
python run_with_landing.py
```

This command will:
1. Start the landing page server at `http://localhost:8000`
2. Automatically start the Streamlit dashboard at `http://localhost:8501`
3. Open the browser on the landing page

The landing page offers "Login Dashboard" and "Access Dashboard" buttons that redirect to Streamlit, where all the main analysis functionality is available.

### Screenshots

The landing page displays information about the project, including the number of stores (54), product families (34), average sales ($55.92), and forecast accuracy (80.2%).