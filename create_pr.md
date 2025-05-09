# Instructions for Creating a Pull Request on GitHub

Follow these steps to create a PR for the `fix/api-model-errors` branch:

1. Visit the GitHub repository at: https://github.com/onchainlabs1/forecast-pipeline

2. Click on the "Pull requests" tab 

3. Click the green "New pull request" button

4. In the "compare" dropdown, select `fix/api-model-errors`

5. Click "Create pull request"

6. Title the PR: "Fix API errors: datetime type checking, model parameters, and metrics calculation"

7. In the description field, paste the contents of the `pr_description.md` file

8. Click "Create pull request"

## Summary of Changes

The `fix/api-model-errors` branch includes fixes for:

- Incorrect datetime type checking in `generate_features` function
- Invalid parameters in `Prediction` model creation 
- Non-existent fields in `ModelMetric` creation
- Feature count mismatch between `get_feature_names` and `generate_features`
- Improved MAPE calculation with better error handling

## Testing

The changes have been tested with a custom script that validates all the fixes and confirms that:

- Predictions can now be successfully generated and saved
- The feature count is properly handled (81 features)
- Metrics calculation works correctly (80.17% accuracy)
- API endpoints no longer throw errors 