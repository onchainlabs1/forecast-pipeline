# Instructions for Setting Up Kaggle Credentials

To download data from Kaggle via the API, follow these steps:

## 1. Create a Kaggle account
If you don't have an account yet, create one at [kaggle.com](https://www.kaggle.com/).

## 2. Generate your API Token
1. Log in to your Kaggle account
2. Go to your account: click on your profile picture in the upper right corner and select "Account"
3. Scroll down to the "API" section
4. Click on "Create New API Token"
5. This will download a file called `kaggle.json` with your credentials

## 3. Configure your credentials
1. If the `~/.kaggle` directory doesn't exist, create it:
   ```bash
   mkdir -p ~/.kaggle
   ```

2. Move the downloaded file to this directory:
   ```bash
   mv ~/Downloads/kaggle.json ~/.kaggle/
   ```

3. Set the correct permissions (to ensure that only you can read the file with your credentials):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 4. Test your credentials
You can verify if your credentials are configured correctly by running:
```bash
kaggle competitions list
```

## 5. Continue with the pipeline
Once the credentials are configured, you can continue with downloading the data:
```bash
python3 src/data/load_data.py
```

Or run the entire pipeline:
```bash
bash run_pipeline.sh
```

## Structure of the kaggle.json file
The `kaggle.json` file should have the following structure:
```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key"
}
```

## Important note
Never share your Kaggle credentials publicly or add them to version control! 