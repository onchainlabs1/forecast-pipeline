# Store Sales Forecasting System

This project implements a complete sales forecasting system for a retail chain, offering an interactive dashboard to visualize historical data, generate future forecasts, and analyze model performance metrics.

## Features

- **Interactive Dashboard**: User-friendly interface to explore sales data
- **Sales Forecasting**: Predictions by store and product family
- **Performance Analysis**: Transparent verification of forecast accuracy
- **Model Insights**: Visualization of feature importance and explainability
- **Monitoring**: Model drift detection and data quality alerts

## Performance Metrics

- **Forecast Accuracy**: 80.17% (calculated as 100 - MAPE)
- **MAPE**: 19.83% (Mean Absolute Percentage Error)
- **MAE**: 46.16 (Mean Absolute Error)
- **RMSE**: 50.64 (Root Mean Square Error)

## Project Structure

```
mlproject/
├── src/
│   ├── api/              # REST API for serving predictions
│   ├── dashboard/        # Streamlit interface
│   ├── database/         # Database models and connection
│   ├── features/         # Feature generation for the model
│   ├── models/           # Machine learning models
│   ├── security/         # Authentication and security
│   └── utils/            # Utility functions
├── models/               # Trained models
├── tests/                # Automated tests
├── run_dashboard.sh      # Script to run the dashboard
└── run_app_with_db.sh    # Script to run API and database
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mlproject.git
cd mlproject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the System

1. Start the API:
```bash
python -m src.api.main
```

2. In another terminal, start the dashboard:
```bash
streamlit run src/dashboard/app.py
```

3. Access the dashboard in your browser: http://localhost:8501

## Demo Credentials

- **Username**: johndoe
- **Password**: secret

## Known Issues

- Type checking error in `isinstance(date, datetime.date)` method
- Feature count mismatch (expected 81, generated 119)
- SHAP explainer error due to dimension mismatch
- Error saving predictions to database (invalid 'date' parameter)

## Next Steps

- Fix type checking bugs
- Resolve feature count incompatibility
- Improve model explainability
- Implement more automated tests