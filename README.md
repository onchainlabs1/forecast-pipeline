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

## Project Structure

```
├── .github/workflows/    # CI/CD pipelines
├── data/                 # Raw and processed data
├── models/               # Trained models
├── notebooks/            # Exploratory data analysis
├── reports/              # Generated reports and visualizations
├── src/                  # Source code
│   ├── data/             # Data processing scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Model training and evaluation
│   └── utils/            # Utility functions
└── tests/                # Unit and integration tests
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
python src/train_model.py
```

## License

MIT 