# Port Configuration Guide

The RetailPro AI platform now supports configurable ports for all its components. This allows running multiple instances of the application simultaneously or resolving port conflicts with other applications.

## Default Ports

- **API Server**: 8000
- **Landing Page**: 8002
- **Dashboard**: 8501

## Setting Custom Ports

You can set custom ports using environment variables:

```bash
# Set custom ports and run the application
API_PORT=9000 LANDING_PORT=9002 DASHBOARD_PORT=9501 ./run_custom_ports.sh
```

Or set them individually before running:

```bash
# Set ports individually
export API_PORT=9000
export LANDING_PORT=9002
export DASHBOARD_PORT=9501

# Then run the application
./run_custom_ports.sh
```

## Running Individual Components

Each component can be run with custom ports:

### API Only

```bash
API_PORT=9000 python src/api/main.py
```

### Dashboard Only

```bash
API_PORT=9000 DASHBOARD_PORT=9501 python -m streamlit run src/dashboard/app.py --server.port $DASHBOARD_PORT
```

### Landing Page Only

```bash
LANDING_PORT=9002 python src/landing/server.py
```

## Running with the Standard Scripts

All provided scripts now support port configuration:

```bash
API_PORT=9000 DASHBOARD_PORT=9501 ./run_dashboard.sh
```

```bash
API_PORT=9000 DASHBOARD_PORT=9501 LANDING_PORT=9002 ./run_with_landing.py
``` 