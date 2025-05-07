#!/bin/bash

# Script to create the kaggle.json file

# Creates the ~/.kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Asks for user information
echo "Enter your Kaggle username:"
read username

echo "Enter your Kaggle API key:"
read key

# Creates the kaggle.json file
echo "{
  \"username\": \"$username\",
  \"key\": \"$key\"
}" > ~/.kaggle/kaggle.json

# Sets the correct permissions
chmod 600 ~/.kaggle/kaggle.json

echo "kaggle.json file successfully created at ~/.kaggle/kaggle.json"
echo "Testing the configuration..."

# Tests the configuration
kaggle competitions list | head -n 5

echo ""
echo "If you saw a list of competitions above, the configuration is correct!"
echo "Now you can run: python3 src/data/load_data.py" 