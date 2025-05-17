# Dependency Management Guide

This document explains the dependency structure for the RetailPro AI project.

## Main Dependencies

The project uses a modular dependency structure to support different deployment scenarios.

### Core Dependencies

The main project dependencies are defined in `requirements.txt`. This file includes all the
necessary packages to run the full application (API, dashboard, and landing page).

```bash
pip install -r requirements.txt
```

### Streamlit-specific Dependencies

For Streamlit Cloud deployment or when running only the dashboard component:

```bash
pip install -r streamlit_requirements.txt
```

## Development Dependencies

For development and testing, additional packages are required:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

## Deployment Options

### Local Deployment

For standard local deployment, use the main requirements file:

```bash
pip install -r requirements.txt
```

### Streamlit Cloud Deployment

For deploying only the dashboard to Streamlit Cloud:

```bash
pip install -r streamlit-cloud-requirements.txt
```

### Docker Deployment

The Docker configuration uses the main requirements file automatically:

```bash
docker build -t retailpro-ai .
docker run -p 8000:8000 -p 8501:8501 -p 8002:8002 retailpro-ai
```

## Legacy Files

The following files are being consolidated and may be removed in future versions:
- `requirements-streamlit.txt` (use `streamlit_requirements.txt` instead)
- `streamlit-cloud-requirements.txt` (use `streamlit_requirements.txt` instead) 