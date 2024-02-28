Fraud_Detection_Project
==============================

1.3 Task 3: Fraud detection in a government agency (spotlight: MLOps)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    │
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── preprocess.py  <- Scripts to turn raw data into features for modeling
    │   │
    |   ├──train_model.py  <- Scripts to train the model
    │   │
    │   │
    |   ├──analyze_process.py <- Scripts to analyze the training data
    │   │
    │   │
    |   ├──production_model_selection.py  <- Scripts to evaluate and select the best model
    │   │
    │  
    │   
    │
    └── DLBDSMTP01          <- the ML training folder
    │   
    │   
    ├── dvc.yaml  <- The DVC pipeline
    ├── params.yaml  <- The key params
    │   
    ├── .githib   <- The Github actions pipelines
    │  ├── mlops_workflow.yml
    │   
    ├── simulate_data_changes.py <- The script to simulate new datasets
    │   
    │   
    ├── Api-Flask <- The flask and swagger packaging for accessing the model
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── run_dvc_with_mlflow.sh <- The bash script for initializing the mlflow server

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Monthly Model Retraining Pipeline

## Overview

This MLOps pipeline is designed to automatically retrain a machine learning model on a monthly schedule, ensuring that the model stays up-to-date with the latest data. It leverages GitHub Actions to orchestrate the workflow, which includes steps for setting up the environment, installing dependencies, simulating data changes, running the DVC pipeline with MLflow tracking, and testing a Flask application.

## Schedule

The pipeline is configured to run at 00:00 UTC on the first day of every month. Additionally, it can be triggered manually via the GitHub Actions `workflow_dispatch` event.

## Workflow Steps

1. **Checkout Repository**: Clone the latest version of the repository to access the pipeline code.

2. **Set Script Execution Permissions**: Make the `run_dvc_with_mlflow.sh` script executable.

3. **Set up Python**: Configure the runner to use Python 3.9.

4. **Install Dependencies**: Install all required dependencies from `requirements.txt` along with DVC, MLflow, Flask, and Gunicorn.

5. **Configure Git**: Set the global Git email and username for commits.

6. **Simulate Data Changes**: Run a Python script to simulate changes in the data.

7. **Run DVC Pipeline with MLflow**: Execute the DVC pipeline, which integrates MLflow for experiment tracking. The MLFLOW_TRACKING_URI environment variable is set to track the experiments.

8. **Reproduce DVC Pipeline**: Reproduce the DVC pipeline stages and commit any changes to the repository.

9. **Commit DVC Changes**: Push the changes to the main branch of the repository.

10. **Start Flask Application**: Launch a Flask application using `deploy.py` in the background.

11. **Wait for Flask to Start**: Pause the workflow for 10 seconds to allow the Flask application to initialize.

12. **Test Flask Application**: Send a request to the Flask application's swagger UI to ensure it's running correctly.

13. **Kill Flask Application**: Terminate the Flask application process.

## Requirements

- GitHub repository with access to GitHub Actions.
- Python 3.9 environment.
- Required Python packages listed in `requirements.txt`.

## Setup Instructions

1. Fork or clone the repository containing the pipeline.
2. Ensure that GitHub Actions is enabled for your repository.
3. Customize the `.github/workflows/monthly_model_retraining.yml` file if needed to fit your project requirements.
4. Add any necessary secrets (e.g., for authentication) through the repository settings.

## Contributing

Contributions to the Monthly Model Retraining pipeline are welcome. Please feel free to submit issues and pull requests through GitHub.

## License

Specify your project's license here.

