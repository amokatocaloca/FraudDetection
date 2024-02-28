import mlflow
from mlflow.tracking import MlflowClient
import argparse
import os
import joblib  # For saving the model locally
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, auc, confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib
matplotlib.use('Agg')


# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:1234")

def select_and_register_best_model(experiment_name, metric_name, registered_model_name, local_model_path):
    """
    Select the best model based on a specified metric from MLflow, register it in the model registry,
    promote it to the production stage, and save it locally.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"No experiment found with name '{experiment_name}'.")
        return

    # Order runs by the specified metric in descending order to get the best run first
    runs = client.search_runs(experiment.experiment_id, order_by=[f"metrics.{metric_name} DESC"])

    if not runs:
        print(f"No runs found for experiment '{experiment_name}'.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id

    # Register the model from the best run in the MLflow model registry
    model_uri = f"runs:/{best_run_id}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    print(f"Model from run {best_run_id} registered in model registry under name '{registered_model_name}'.")
    print(f"Model version: {model_details.version}")

    # Transition the registered model to the "Production" stage
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_details.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model version {model_details.version} of '{registered_model_name}' has been promoted to production.")

    # Load the best model and save it locally
    model = mlflow.pyfunc.load_model(model_uri)
    if not os.path.exists(os.path.dirname(local_model_path)):
        os.makedirs(os.path.dirname(local_model_path))
    joblib.dump(model, local_model_path)
    print(f"Best model saved locally at: {local_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and register the best model based on metrics.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the MLflow experiment.")
    parser.add_argument("--metric_name", type=str, default="val_f1_score", help="Metric name to use for selecting the best model.")
    parser.add_argument("--registered_model_name", type=str, required=True, help="Name for the registered MLflow model.")
    parser.add_argument("--local_model_path", type=str, required=True, help="Local path to save the selected best model.")

    args = parser.parse_args()

    select_and_register_best_model(
        experiment_name=args.experiment_name,
        metric_name=args.metric_name,
        registered_model_name=args.registered_model_name,
        local_model_path=args.local_model_path
    )

