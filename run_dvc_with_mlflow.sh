#!/bin/bash

# Start MLflow server in the background
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts --host 0.0.0.0 --port 1234 &

# Wait for the MLflow server to start
sleep 10

# Execute DVC repro command
dvc repro

# Optionally, wait for any background process to complete
# wait
