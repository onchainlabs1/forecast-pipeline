# Sales Forecasting MLOps Project

> **Status: In Development** - This project is currently under active development. Some features may not be fully implemented and changes are expected.

## Overview

This project implements a complete MLOps pipeline for store sales forecasting, including:

- **Machine Learning Model**: Using LightGBM for sales prediction
- **REST API**: Implemented with FastAPI to serve predictions
- **Dashboard**: Interactive user interface implemented with Streamlit
- **Experiment Tracking**: Using MLflow to manage model versions
- **Data Versioning**: Using DVC for data version control
- **Authentication**: JWT system for secure user authentication
- **Monitoring**: Monitoring for model drift and prediction quality
- **Database**: Full database integration for storing historical data and predictions

## Project Structure

```
mlproject/
├── data/                # Raw and processed data
├── models/              # Trained models
├── mlruns/              # MLflow records
├── notebooks/           # Exploratory data analysis notebooks
├── src/
│   ├── api/             # FastAPI API
│   ├── dashboard/       # Streamlit Dashboard
│   ├── features/        # Feature engineering
│   ├── models/          # Model definitions
│   ├── security/        # Security and authentication
│   ├── repositories/    # Database repositories
│   ├── database/        # Database models and setup
│   └── utils/           # Utilities
├── tests/               # Automated tests
├── .dvc/                # DVC configuration
├── .github/             # GitHub Actions workflows
├── requirements.txt     # Project dependencies
├── run_dashboard.sh     # Script to start the dashboard application
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/mlproject.git
cd mlproject
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the `python-jose` library for JWT authentication:
```bash
pip install python-jose
```

## Running the Application

### All-in-one Startup

Use the provided script to start both API and dashboard:

```bash
# Start both API and dashboard
bash run_dashboard.sh
```

### Running Components Separately

You can also start the components individually:

```bash
# Start the API
python src/api/main.py

# Start the Streamlit dashboard (in another terminal)
streamlit run src/dashboard/app.py
```

The API will be available at `http://localhost:8000`. API documentation: `http://localhost:8000/docs`
The dashboard will be available at `http://localhost:8501`

## Dashboard Features

The dashboard provides the following features:

- **Sales Trends**: View historical sales data for stores and product families
- **Store Comparison**: Compare performance across different stores
- **Product Family Performance**: Analyze sales and growth by product family
- **Predictions**: Generate and visualize sales predictions for specific stores and dates
- **Model Insights**: View model performance metrics, feature importance, and drift
- **User Authentication**: Secure login system with role-based access

## Database Integration

The project includes a full database integration for storing and retrieving:

- Historical sales data
- Model predictions
- Store and product family information
- Model metrics and feature importance
- Model drift detection

### Database Structure

The database includes the following tables:

- `stores`: Store information
- `product_families`: Product family information
- `historical_sales`: Historical sales data
- `predictions`: Model predictions
- `model_metrics`: Model performance metrics
- `feature_importance`: Feature importance for model explainability
- `model_drift`: Model drift metrics for monitoring

### Using the Database

The API connects to the database to provide real data for the dashboard. If a specific combination of store and product family has no data, the system will suggest combinations with available data.

## Deployment Configuration

### Environment Variables

Create a `.env` file in the project root:

```
API_URL=https://your-api-url.com
MLFLOW_URL=https://your-mlflow-url.com
DATABASE_URL=sqlite:///./app.db  # Default SQLite database
```

### Streamlit Cloud Deployment

The dashboard portion of this application can be easily deployed to Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Make sure your fork is up to date with the latest changes
3. Go to [Streamlit Cloud](https://streamlit.io/cloud)
4. Click "New app" and connect your GitHub account
5. Select your forked repository
6. Set the main file path to `src/dashboard/app.py`
7. Add the following secrets in the Streamlit Cloud settings:
   - `API_URL`: URL where your FastAPI backend is deployed
   - `MLFLOW_URL`: URL where your MLflow tracking server is hosted (optional)
8. Deploy the application

For the complete solution, you'll also need to deploy the FastAPI backend separately on a platform like Heroku, AWS, GCP, or Azure.

> **Note**: When deploying to Streamlit Cloud, make sure your API endpoint supports CORS to allow requests from your Streamlit app's domain.

## Demo Users

- Username: johndoe / Password: secret
- Username: admin / Password: admin

## Development

### Updating Dependencies

If you add new dependencies, update the requirements.txt file:

```bash
pip freeze > requirements.txt
```

### Running Tests

```bash
pytest
```

### Troubleshooting

If you encounter the KeyError: 'predicted_sales' error in the dashboard:
1. Stop all running processes
2. Delete all __pycache__ directories: `find . -name "__pycache__" -type d -exec rm -rf {} +`
3. Restart the application using the script: `bash run_dashboard.sh`

The dashboard is now configured to handle both 'prediction' and 'predicted_sales' key formats from the API, making it more robust.

#### Interface Customization

The dashboard now uses a dark theme by default. This is configured in the `src/dashboard/app.py` file with custom CSS. You can modify the theme colors by editing the CSS variables in the file.

### Known Issues

- SHAP explanations for predictions may fail with non-tree-based models
- Some combination of stores and product families may not have historical data
- The dashboard may show error when the prediction functionality is used with specific parameters

## Contribution

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 