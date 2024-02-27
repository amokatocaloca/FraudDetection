import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, auc, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import yaml  # Import YAML to read the parameters file
import argparse
from datetime import datetime  # Correct import for datetime
import json  # Add this import to your existing imports
import pickle


mlflow.set_tracking_uri("http://0.0.0.0:1234")  # Set the MLflow tracking URI
from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_name = "FraudDetectionModelExperiment"
experiment = client.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Experiment '{experiment_name}' not found. Creating...")
    client.create_experiment(experiment_name)
else:
    print(f"Experiment '{experiment_name}' found.")
    
def save_metrics_to_json(metrics, filename):
    with open(filename, 'w') as f:
        json.dump(metrics, f)

def read_params(params_path):
    with open(params_path) as file:
        return yaml.safe_load(file)
    
def logreg_and_register(x_train, y_train, x_val, y_val, x_test, y_test, file_base, train_params, experiment_name):
    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Scale data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        # Train logistic regression model
        lr = LogisticRegression(
            class_weight=train_params.get('class_weight', 'balanced'), 
            max_iter=train_params.get('max_iters', 500), 
            solver='lbfgs'
        )
        lr.fit(x_train_scaled, y_train)

        # Evaluate model
        y_val_pred = lr.predict(x_val_scaled)
        val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(y_val, y_val_pred, average='binary')

        # Log parameters, metrics, and model
        mlflow.log_params(train_params)
        mlflow.log_metrics({
            'val_precision': val_precision, 
            'val_recall': val_recall, 
            'val_f1_score': val_f1_score
        })
        mlflow.sklearn.log_model(lr, "model", registered_model_name=f"BestLRModel_{file_base}")

        print(f"Logged model with val_f1_score: {val_f1_score}")

        # Ensure the models directory exists
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)

        # Save model locally
        model_filename = os.path.join(models_dir, f'best_lr_{file_base}.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(lr, f)
        print(f"Model saved to {model_filename}")

        # Save metrics to JSON
        metrics_data = {
            'val_precision': val_precision, 
            'val_recall': val_recall, 
            'val_f1_score': val_f1_score
        }
        metrics_filename = f'metrics/train_metrics_{file_base}.json'
        save_metrics_to_json(metrics_data, metrics_filename)
        print(f"Metrics saved to {metrics_filename}")

# Function to scale data
def scale_data(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled

def plot_confusion_matrix(y_true, y_pred, file_base, iteration, session_id=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for Iteration {iteration}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    # Change this line to point to 'plots'
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    filename_suffix = f"_{session_id}" if session_id else ""
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{file_base}_{iteration}{filename_suffix}.png'))
    plt.close()

def plot_precision_recall_curve(model, x_val, y_val, file_base, iteration, session_id=None):
    y_scores = model.predict_proba(x_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_scores)
    auc_score = auc(recall, precision)
    
    plt.figure()
    plt.plot(recall, precision, label=f'Iteration {iteration} (AUC = {auc_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for Iteration {iteration}')
    plt.legend()
    
    # Change this line to point to 'plots'
    plt.savefig(os.path.join('plots', f'pr_curve_{file_base}_{iteration}{filename_suffix}.png'))
    plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description='Train and evaluate logistic regression model.')
    parser.add_argument('--data_filename', type=str, required=True, help='Path to the input CSV data file.')
    parser.add_argument('--params_path', type=str, default='params.yaml', help='Path to the parameters YAML file.')
    args = parser.parse_args()

    # Extract file_base here to ensure it's defined before use
    file_base = os.path.splitext(os.path.basename(args.data_filename))[0]
    params = read_params(args.params_path)
    print("Training parameters:", params)

    train_params = params['train']
    experiment_name = params['experiment_name']

    df = pd.read_csv(args.data_filename)
    if df.empty:
        print(f"Data file {args.data_filename} is empty.")
        sys.exit(1)

    X = df.drop(['isFraud', 'type'], axis=1).values
    y = df['isFraud'].values

    x_train, x_val_test, y_train, y_val_test = train_test_split(X, y, stratify=y, test_size=train_params['test_size'], random_state=train_params['random_state'])
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, stratify=y_val_test, test_size=0.5, random_state=train_params['random_state'])

    logreg_and_register(x_train, y_train, x_val, y_val, x_test, y_test, file_base, train_params, experiment_name)