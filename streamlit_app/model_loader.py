import os
import requests
import pandas as pd
import mlflow
from dotenv import load_dotenv

def get_best_model_info():
    """Get information about the best performing model"""
    load_dotenv()
    
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    
    mlflow.set_tracking_uri(tracking_uri)
    
    exp = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.recall DESC"],
        max_results=1
    )
    
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment {experiment_name}")
    
    best_run = runs.iloc[0]
    return {
        "run_id": best_run['run_id'],
        "model_name": best_run['params.model'],
        "model_uri": f"runs:/{best_run['run_id']}/{best_run['params.model']}",
        "accuracy": best_run['metrics.accuracy'],
        "recall": best_run['metrics.recall'],
        "f1_score": best_run['metrics.f1_score']
    }

def predict_churn(features_df, serving_port=5000):
    """Make prediction via MLflow model serving endpoint"""
    try:
        # Format data for MLflow API
        input_data = {
            "inputs": features_df.to_dict(orient="records")
        }
        
        # Call MLflow serving endpoint
        response = requests.post(
            f"http://localhost:{serving_port}/invocations",
            json=input_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            predictions = response.json()
            return {
                "prediction": predictions[0],
                "success": True
            }
        else:
            return {
                "error": f"Server error: {response.status_code}",
                "success": False
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "error": "Cannot connect to MLflow model server. Is it running?",
            "success": False
        }
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "success": False
        }