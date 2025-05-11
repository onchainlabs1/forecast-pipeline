#!/bin/bash

# Script to push Streamlit Cloud deployment changes to GitHub
# Usage: ./deploy_to_streamlit.sh [branch-name]

set -e  # Exit on error

# Default to "fix/streamlit-cloud-compatibility" if no branch is specified
BRANCH=${1:-fix/streamlit-cloud-compatibility}

echo "Preparing Streamlit Cloud deployment..."

# Create a new branch if it doesn't exist
if ! git rev-parse --verify $BRANCH >/dev/null 2>&1; then
  echo "Creating new branch: $BRANCH"
  git checkout -b $BRANCH
else
  echo "Switching to existing branch: $BRANCH"
  git checkout $BRANCH
fi

# Add the new/modified files
git add requirements.txt
git add streamlit_minimal.py
git add .streamlit/config.toml
git add environment.yml
git add README_STREAMLIT_CLOUD.md

# Commit the changes
git commit -m "Fix Streamlit Cloud compatibility issues

- Updated requirements.txt with compatible dependency versions
- Created streamlit_minimal.py for Streamlit Cloud deployment
- Added .streamlit/config.toml for proper configuration
- Updated environment.yml for Conda users
- Added README_STREAMLIT_CLOUD.md with deployment instructions"

# Push to GitHub
echo "Pushing changes to GitHub..."
git push -u origin $BRANCH

echo "==============================================="
echo "Deployment files pushed to $BRANCH"
echo "Next steps:"
echo "1. Create a pull request on GitHub"
echo "2. Deploy to Streamlit Cloud using streamlit_minimal.py as the main file"
echo "3. Test the deployment to ensure it's working properly"
echo "===============================================" 