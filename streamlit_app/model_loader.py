# model_loader.py

import os
import mlflow
import mlflow.sklearn

def load_model():
    """Load the best churn prediction model for Streamlit app"""
    
    # Connect to MLflow
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get the best model by recall
    exp = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.recall DESC"],
        max_results=1
    )
    
    # Load and return the model
    best_run = runs.iloc[0]
    model_uri = f"runs:/{best_run['run_id']}/{best_run['params.model']}"
    model = mlflow.sklearn.load_model(model_uri)
    
    return model