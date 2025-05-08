# Sales Forecasting MLOps Project

## Overview

This project implements a complete MLOps pipeline for store sales forecasting, including:

- **Machine Learning Model**: Using LightGBM for sales prediction
- **REST API**: Implemented with FastAPI to serve predictions
- **Dashboard**: Interactive user interface implemented with Streamlit
- **Experiment Tracking**: Using MLflow to manage model versions
- **Data Versioning**: Using DVC for data version control
- **Authentication**: JWT system for secure user authentication
- **Monitoring**: Sentry integration for error detection

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
│   └── utils/           # Utilities
├── tests/               # Automated tests
├── .dvc/                # DVC configuration
├── .github/             # GitHub Actions workflows
├── requirements.txt     # Project dependencies
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

## Running the API

```bash
# Start the API
python src/api/main.py
```

The API will be available at `http://localhost:8000`. API documentation: `http://localhost:8000/docs`

## Running the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run src/dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## Deployment Configuration

### Environment Variables

Create a `.env` file in the project root:

```
API_URL=https://your-api-url.com
MLFLOW_URL=https://your-mlflow-url.com
SENTRY_DSN=your-sentry-dsn
```

### Streamlit Cloud Configuration

1. Push the project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select the repository and the file `src/dashboard/app.py`
5. Configure the necessary environment variables (API_URL, MLFLOW_URL)
6. Deploy the application

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

## Contribution

1. Fork the project
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 