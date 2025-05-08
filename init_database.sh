#!/bin/bash

# Script to initialize the database and load initial data

echo "Initializing database..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3 before continuing."
    exit 1
fi

# Create the database directory if it doesn't exist
mkdir -p data/db

# Check if required packages are installed
echo "Checking required packages..."
REQUIRED_PACKAGES="sqlalchemy pandas"
MISSING_PACKAGES=""

for package in $REQUIRED_PACKAGES; do
    python3 -c "import $package" 2>/dev/null
    if [ $? -ne 0 ]; then
        MISSING_PACKAGES="$MISSING_PACKAGES $package"
    fi
done

if [ ! -z "$MISSING_PACKAGES" ]; then
    echo "Installing missing packages:$MISSING_PACKAGES"
    pip install $MISSING_PACKAGES
fi

# Run the database initialization script
echo "Running database initialization script..."
python3 -m src.database.init_db

# Check if initialization was successful
if [ $? -eq 0 ]; then
    echo "Database initialization completed successfully."
    echo "You can now start the API with:"
    echo "  python -m src.api.main"
    echo ""
    echo "Or run the dashboard with:"
    echo "  python -m src.dashboard.app"
else
    echo "Error: Database initialization failed."
    exit 1
fi 