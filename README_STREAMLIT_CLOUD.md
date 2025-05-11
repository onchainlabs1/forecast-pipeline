# Streamlit Cloud Deployment Guide

This guide provides instructions for deploying the sales forecasting dashboard on Streamlit Cloud.

## Deployment Instructions

1. **Log in to Streamlit Cloud** (https://streamlit.io/cloud)

2. **Configure your deployment:**
   - Repository: `forecast-pipeline`
   - Branch: `main`
   - Main file path: `streamlit_minimal.py` (use this minimal entry file)

3. **Advanced settings:**
   - Python version: 3.9 or 3.10 (not 3.12)
   - Package dependencies: The system will use the `requirements.txt` file

## Troubleshooting

If you encounter dependency issues:

1. Try using the `streamlit_minimal.py` file as the entry point
2. Check the logs for specific dependency errors
3. If problems persist, update the `requirements.txt` file with more conservative versions

## API Connection

The dashboard connects to the Render API endpoint at:
https://forecast-pipeline-2.onrender.com

If the API is not responding:
1. Check if the Render deployment is running
2. Verify API health at: https://forecast-pipeline-2.onrender.com/health
3. Restart the Render service if necessary

## Notes on Dependencies

The project has been configured with minimal dependencies for Streamlit Cloud compatibility:

- Streamlit 1.28.0
- Pandas 1.3.5
- Numpy 1.22.4
- No SqlAlchemy or other database dependencies
- No Apache Airflow dependencies
- Fixed pydantic version to avoid conflicts

This minimalist approach ensures maximum compatibility with Streamlit Cloud. 