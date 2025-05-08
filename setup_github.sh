#!/bin/bash

# Script to prepare and upload the project to GitHub

echo "Configuring Git repository for the Sales Forecasting MLOps project..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Git not found. Please install Git first."
    exit 1
fi

# Install required dependencies
echo "Installing required dependencies..."
pip install python-multipart python-jose[cryptography] passlib[bcrypt] sentry_sdk shap
echo "Dependencies installed successfully."

# Initialize Git repository if .git doesn't exist
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

# Configure git to use English for commit messages
echo "Configuring git to use English for commit messages..."
git config --local i18n.commitEncoding utf-8
git config --local i18n.logOutputEncoding utf-8
git config --local core.quotepath false
git config --local commit.template /dev/null

# Create .gitignore file
echo "Creating .gitignore file..."
cat > .gitignore << EOL
# Python files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
.env

# Jupyter Notebook
.ipynb_checkpoints

# Project specific
mlruns/
*.pkl
logs/
*.log

# Data (managed by DVC, if used)
data/*.csv
data/*.parquet
data/*.json

# IDE configurations
.idea/
.vscode/
*.swp
*.swo

# OS specific
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOL

# Add all files (except those ignored by .gitignore)
echo "Adding files to the repository..."
git add .

# Make the initial commit
echo "Making initial commit..."
git commit -m "Initial commit: Sales Forecasting MLOps Project"

# Instructions to connect to GitHub
echo ""
echo "Now, create a repository on GitHub and connect it using the commands below:"
echo ""
echo "  git remote add origin https://github.com/YOUR-USERNAME/mlproject.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "Replace 'YOUR-USERNAME' with your GitHub username."
echo ""
echo "To create a repository on GitHub, go to: https://github.com/new"
echo ""

# Configure files for Streamlit Cloud
echo "Creating configuration file for Streamlit Cloud..."
mkdir -p .streamlit
cat > .streamlit/config.toml << EOL
[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
EOL

echo "Creating requirements file for Streamlit Cloud..."
cp requirements.txt requirements-streamlit.txt

echo "Configuration complete! You're ready to push your project to GitHub." 