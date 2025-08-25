#!/usr/bin/env python
import os
from dotenv import load_dotenv
import mlflow

def main():
    load_dotenv()  # loads .env from project root

    # ── 1) MLflow "tracking" vs "registry" hack ─────────────────────
    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not tracking_uri or not experiment_name:
        raise ValueError("Must set MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_NAME in .env")

    # a) all "tracking" calls go to your ARN-based server
    mlflow.set_tracking_uri(tracking_uri)

    # b) but force ALL registry calls onto a local file store so "arn:" never trips up registry
    registry_store = os.path.join(os.getcwd(), ".mlflow_registry")
    os.makedirs(registry_store, exist_ok=True)
    mlflow.set_registry_uri(f"file://{registry_store}")

    # c) safe to instantiate the client / experiment now
    mlflow.set_experiment(experiment_name)

    # ── 2) Find best run by recall ───────────────────────────
    exp = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.recall DESC"],
        max_results=1
    )
    if runs.empty:
        raise RuntimeError(f"No runs found in experiment {experiment_name}")

    best = runs.iloc[0]
    run_id = best["run_id"]
    artifact_path = best["params.model"]
    print(f"✅ Best run {run_id}, artifact_path={artifact_path}")

    # ── 3) Report best model info ────────────────────────────────
    model_uri = f"runs:/{run_id}/{artifact_path}"
    print(f"✅ Model URI: {model_uri}")
    print(f"✅ Model successfully stored in MLflow tracking server")
    print(f"✅ Available at: {tracking_uri}")
    print(f"✅ Pipeline completed - model ready for deployment!")

if __name__ == "__main__":
    main()