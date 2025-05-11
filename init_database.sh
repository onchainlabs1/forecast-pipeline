#!/bin/bash

# Script to initialize the database and load initial data

echo "Initializing database..."

# Use the full path to Python from Anaconda
PYTHON_PATH="/Users/fabio/anaconda3/bin/python"

# Create the database directory if it doesn't exist
mkdir -p data/db

# Check if required packages are installed
echo "Checking required packages..."
REQUIRED_PACKAGES="sqlalchemy pandas numpy"
MISSING_PACKAGES=""

for package in $REQUIRED_PACKAGES; do
    $PYTHON_PATH -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

if [ ! -z "$MISSING_PACKAGES" ]; then
    echo "Installing missing packages:$MISSING_PACKAGES"
    $PYTHON_PATH -m pip install $MISSING_PACKAGES
fi

# Create data directories if they don't exist
mkdir -p data/raw

# Run the database initialization script
echo "Running database initialization script..."
$PYTHON_PATH -m src.database.init_db

# Make sure there is sample data in the database
echo "Checking if sample data is loaded properly..."
DB_STATUS=$($PYTHON_PATH -c "
import sys
sys.path.append('.')
from src.database.database import get_db
from src.database.models import Store, ProductFamily, HistoricalSales
from sqlalchemy import func, distinct

db = next(get_db())
stores_count = db.query(func.count(distinct(Store.id))).scalar() or 0
families_count = db.query(func.count(distinct(ProductFamily.id))).scalar() or 0
sales_count = db.query(func.count(HistoricalSales.id)).scalar() or 0

print(f'DB_STATUS: {stores_count} stores, {families_count} families, {sales_count} sales records')
" 2>/dev/null)

echo "Database status: $DB_STATUS"

# If there's not enough data, force loading the demo data
if [[ "$DB_STATUS" != *"DB_STATUS: "* ]] || [[ "$DB_STATUS" == *"0 stores"* ]] || [[ "$DB_STATUS" == *"0 families"* ]] || [[ "$DB_STATUS" == *"0 sales"* ]]; then
    echo "Insufficient data detected in the database. Loading sample data..."
    $PYTHON_PATH -c "
import sys
sys.path.append('.')
from src.database.data_loader import load_sample_data
load_sample_data(force=True)
print('Sample data loaded successfully.')
"
fi

# Check one more time to make sure data was loaded
DB_STATUS=$($PYTHON_PATH -c "
import sys
sys.path.append('.')
from src.database.database import get_db
from src.database.models import Store, ProductFamily, HistoricalSales
from sqlalchemy import func, distinct

db = next(get_db())
stores_count = db.query(func.count(distinct(Store.id))).scalar() or 0
families_count = db.query(func.count(distinct(ProductFamily.id))).scalar() or 0
sales_count = db.query(func.count(HistoricalSales.id)).scalar() or 0

print(f'Final DB Status: {stores_count} stores, {families_count} families, {sales_count} sales records')
" 2>/dev/null)

echo "Final database status: $DB_STATUS"

echo "Database initialization completed successfully."
echo "You can now start the API with:"
echo "  $PYTHON_PATH -m src.api.main"
echo ""
echo "Or run the dashboard with:"
echo "  $PYTHON_PATH -m src.dashboard.app"

# Create run_dashboard.sh script
cat > run_dashboard.sh << EOF
#!/bin/bash

# Script to start both API and dashboard
# First check if API is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "API already running on port 8000"
else
    echo "Starting API on port 8000..."
    $PYTHON_PATH -m src.api.main &
    API_PID=$!
    sleep 2
fi

echo "Starting the dashboard..."
$PYTHON_PATH -m streamlit run src/dashboard/app.py

# If we started the API, kill it when the dashboard closes
if [ ! -z "$API_PID" ]; then
    echo "Dashboard closed, stopping API server..."
    kill $API_PID
fi
EOF

chmod +x run_dashboard.sh

echo "Created run_dashboard.sh script to start both API and dashboard"
echo "To run the complete application, use:"
echo "  ./run_dashboard.sh" 