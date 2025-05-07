# Store Sales Forecasting

A complete sales forecasting pipeline for multiple stores and products, integrating modern MLOps practices.

## Objective

Develop a complete sales forecasting pipeline with time series for multiple stores and products, integrating modern MLOps practices. The project is reproducible, scalable, and easily demonstrable via GitHub.

## Data Source

- Dataset: [Favorita Grocery Sales Forecasting - Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- Description: Daily sales by product and store, with information on holidays, promotions, and product types.
- Size: ~125MB
- Format: CSV (train.csv, holidays_events.csv, oil.csv, stores.csv, transactions.csv)

## Technology Stack

- **ML**: LightGBM, Prophet, ARIMA (baseline)
- **Feature Engineering**: pandas, scikit-learn, tsfeatures
- **MLOps**:
  - MLflow (experiments + model registry)
  - DVC (data versioning and pipeline)
  - Docker (reproducible environment)
  - FastAPI (inference REST service)
  - GitHub Actions (CI/CD for tests and deployment)
  - Feature Store (metadata management + feature versioning)
  - Model Monitoring (drift detection + automated retraining)

## Advanced MLOps Features

### Feature Store
- Central repository for all feature definitions
- Feature versioning and metadata tracking
- Dependency management between features
- Transformation management for training and inference
- Feature grouping for better organization

### Model Monitoring
- Data drift detection using statistical methods
- Performance monitoring to detect model degradation
- Automated alerting when drift is detected
- Visual drift reports for better understanding
- Automated recommendation for retraining

### Automated Training Pipeline
- End-to-end training pipeline with proper workflow
- Integration with Feature Store and Model Monitoring
- Experiment tracking with MLflow
- Model registry integration
- Continuous evaluation and deployment

## Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Raw and processed data
├── feature_store/        # Feature definitions and metadata
├── models/               # Trained models
├── monitoring/           # Model monitoring data and alerts
├── notebooks/            # Exploratory data analysis
├── reports/              # Generated reports and visualizations
├── src/                  # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering and Feature Store
│   ├── models/           # Model training and evaluation
│   ├── monitoring/       # Monitoring and drift detection
│   ├── api/              # FastAPI service
│   └── utils/            # Utility functions
├── tests/                # Unit and integration tests
├── scheduled_training.sh # Script for automated training
└── logs/                 # Training logs
```

## Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/store-sales-forecasting.git
cd store-sales-forecasting

# Install dependencies
pip install -r requirements.txt

# Download the data
python src/data/load_data.py

# Run the training pipeline
python src/training_pipeline.py
```

## Running the API

Start the prediction API service:

```bash
python src/api/main.py
```

The API will be available at http://localhost:8000, with interactive documentation at http://localhost:8000/docs.

## Experiment Tracking with MLflow

This project uses MLflow for experiment tracking and model versioning:

```bash
# Start the MLflow UI
mlflow ui
```

Access the MLflow dashboard at http://localhost:5000 to:
- Compare different runs and models
- View performance metrics
- Manage model versions
- Track model lineage and parameters

## Model Monitoring

Monitor your model for drift and performance issues:

```bash
# Check for data drift in new data
python -c "from src.monitoring.model_monitoring import ModelMonitor; mm = ModelMonitor('store-sales-forecaster'); mm.check_data_drift(new_data)"

# View recent alerts
python -c "from src.monitoring.model_monitoring import ModelMonitor; mm = ModelMonitor('store-sales-forecaster'); print(mm.get_alerts())"
```

## Automated Training

The project includes a script for scheduled model training:

```bash
# Make the script executable
chmod +x scheduled_training.sh

# Run the script manually
./scheduled_training.sh
```

For automated training, set up a cron job (see `scheduled_training_instructions.md` for details).

## License

MIT 