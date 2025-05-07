# Scheduled Training Instructions

This document explains how to set up automated training for the sales forecasting model.

## Prerequisites

- The MLOps project must be fully set up
- MLflow tracking server should be configured
- Required dependencies must be installed

## Scheduling Options

### Option 1: Using Cron (Linux/macOS)

To set up a recurring training job using cron:

1. Open the crontab editor:
   ```bash
   crontab -e
   ```

2. Add a schedule for the training script. For example, to run weekly on Monday at 2:00 AM:
   ```
   0 2 * * 1 cd /Users/fabio/Desktop/mlproject && ./scheduled_training.sh
   ```

3. Save and exit the editor.

### Option 2: Using Task Scheduler (Windows)

1. Open Task Scheduler
2. Create a new Basic Task
3. Set the trigger (e.g., weekly on Monday at 2:00 AM)
4. Set the action to run `scheduled_training.sh`
5. Set the start in directory to the project path

### Option 3: Using Cloud Services

For production environments, consider using cloud scheduling services:

- **AWS**: Use AWS EventBridge (CloudWatch Events) with Lambda or ECS
- **GCP**: Use Cloud Scheduler with Cloud Functions or Cloud Run
- **Azure**: Use Azure Functions with timer triggers

## Monitoring Training Jobs

1. Check the logs directory for training logs:
   ```bash
   ls -la logs/
   ```

2. View MLflow experiments to compare model versions:
   ```bash
   mlflow ui
   ```
   Then open http://localhost:5000 in your browser

## Advanced Configuration

You can customize the `scheduled_training.sh` script to:

1. Only retrain when new data is available
2. Send notifications upon completion or failure
3. Automatically promote models that meet performance criteria
4. Deploy the new model to production

## Troubleshooting

If scheduled training isn't working:

1. Check the logs in the `logs/` directory
2. Ensure the script has execution permissions: `chmod +x scheduled_training.sh`
3. Verify the cron service is running
4. Test the script manually: `./scheduled_training.sh` 