# Manual Data Download

Since we're having issues with the Kaggle API credentials, let's download the data manually:

1. Go to the competition page on Kaggle: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

2. Log in to your Kaggle account

3. Click the "Download All" button to download all files

4. Extract the downloaded ZIP file

5. Move the following files to the `/Users/fabio/Desktop/mlproject/data/raw/` directory:
   - train.csv
   - test.csv
   - holidays_events.csv
   - oil.csv
   - stores.csv
   - transactions.csv

## After manual download

Once the files are in the correct directory, we can continue with the pipeline:

```bash
# Check if the files are in the correct location
ls -la data/raw/

# Run data preprocessing
python3 src/data/preprocess.py

# Train the model
python3 src/train_model.py

# Or run the entire pipeline (skip the download step)
bash run_pipeline.sh
``` 