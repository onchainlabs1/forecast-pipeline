# RetailPro AI: Sales Forecasting Platform

<div align="center">
  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.21.2-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

## 🎬 Project Overview Video

<div align="center">
  <a href="https://www.youtube.com/watch?v=Tye1Xjep3ss" target="_blank">
    <img src="docs/images/thumb_youtube.png" alt="Watch the project video on YouTube" width="500"/>
  </a>
  <br/>
  <b>Click the image above to watch a full video walkthrough of the project on YouTube!</b>
</div>

## 🖼️ Platform Previews

<div align="center">
  <img src="docs/images/landing.png" alt="Landing Page Screenshot" width="700"/>
  <br/>
  <i>Landing Page: Welcome screen for users, project context, and navigation.</i>
</div>

<div align="center">
  <img src="docs/images/dashboard_main.png" alt="Dashboard Main Screenshot" width="700"/>
  <br/>
  <i>Main Dashboard: Interactive sales forecasting and analytics.</i>
</div>

<div align="center">
  <img src="docs/images/predictions.png" alt="Predictions Screenshot" width="700"/>
  <br/>
  <i>Predictions: Visualizing model outputs and forecast results.</i>
</div>

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

Our platform follows a modern microservices architecture with four main components:

- **Landing Page:** The entry point for users, providing project context and navigation.
- **Streamlit Dashboard:** Interactive visualization and forecasting interface.
- **FastAPI Backend:** Handles API requests, authentication, and model inference.
- **ML Models & Monitoring:** Handles predictions, feature engineering, and model monitoring.

Below is a high-level architecture diagram:

```
┌────────────────────┐
│   Landing Page     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Streamlit Dashboard│
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│   FastAPI Backend  │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ ML Models &        │
│ Predictions        │
├────────────────────┤
│ Feature Engineering│
├────────────────────┤
│ Monitoring &       │
│ Metrics            │
└────────────────────┘
```

## 📝 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/onchainlabs1/forecast-pipeline.git
   cd forecast-pipeline
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```
   
3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   For more information on dependency management, see [Dependency Management Guide](DEPENDENCIES.md).

## 🚀 Running the Application

### Option 1: All-in-One Startup

Run everything with a single command:
```bash
python run_with_landing.py
```

This will start:
- The FastAPI backend on port 8000
- The Lovable landing page on port 8002
- The Streamlit dashboard on port 8501

For custom port configuration, see [Port Configuration Guide](PORT_CONFIGURATION.md).

### Option 2: Individual Component Startup

Start each component separately:

1. **API Server (Backend)**
   ```bash
   python src/api/main.py
   ```
   Access at: http://localhost:8000

2. **Landing Page (Lovable)**
   ```bash
   python src/landing/server.py
   ```
   Access at: http://localhost:8002

3. **Dashboard (Streamlit)**
   ```bash
   streamlit run src/dashboard/app.py
   ```
   Access at: http://localhost:8501

## 🔐 Authentication

Use the following credentials to log in:
- **Username:** admin
- **Password:** admin

## 🧪 Testing

Run the test suite:
```bash
pytest
```

## 🔧 Code Standards

This project uses pre-commit hooks to maintain code quality and consistency. To set up:

```bash
pip install pre-commit
pre-commit install
```

The pre-commit configuration enforces:
- Trailing whitespace removal
- Adding newline at end of files
- Code formatting with Black
- Linting with Flake8

An EditorConfig file is also provided to help maintain consistent coding styles.

## 📄 Documentation

- [Port Configuration Guide](PORT_CONFIGURATION.md): Learn how to configure ports for different components
- [Dependency Management Guide](DEPENDENCIES.md): Understanding the project's dependency structure

## 📊 MLflow Integration

View model metrics and experiments:
```bash
mlflow ui
```
Access the MLflow UI at: http://localhost:5000

## 🌐 Airflow Integration (Optional)

For automated model retraining and data pipeline orchestration:
```bash
cd airflow
docker-compose up -d
```
Access the Airflow UI at: http://localhost:8080

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Environment Setup

### 1. Security Configuration

For security reasons, this project uses environment variables for sensitive configurations. Before running the application, you need to set up your environment:

1. Generate a secure `.env` file:
```bash
python scripts/generate_env.py
```

This will create a `.env` file with secure random values for:
- `SECRET_KEY`: Used for JWT token encryption
- `DEMO_ADMIN_PASSWORD`: Password for the admin user
- `DEMO_USER_PASSWORD`: Password for the demo user
- Other configuration variables

### 2. Environment Variables

The following environment variables are required:

```bash
# Security Configuration
SECRET_KEY=your-secret-key-here

# Demo Credentials (change in production)
DEMO_ADMIN_PASSWORD=admin
DEMO_USER_PASSWORD=secret

# Server Ports
PORT=8002
API_PORT=8000
DASHBOARD_PORT=8501
MLFLOW_PORT=8888

# Database Configuration
DATABASE_URL=sqlite:///data/store_sales.db

# MLflow Configuration
MLFLOW_TRACKING_URI=mlruns

# Token Configuration
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

⚠️ **IMPORTANT**: 
- Never commit the `.env` file to version control
- In production, always use strong, randomly generated passwords
- Change the demo credentials in production
