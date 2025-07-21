#!/usr/bin/env python

import os
from io import BytesIO

import mlflow
import mlflow.sklearn
import pandas as pd
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

def main():
    # ── 1) Load env vars ────────────────────────────────────────────────────
    load_dotenv()  # expects a .env at project root
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment   = os.getenv("MLFLOW_EXPERIMENT_NAME")
    bucket       = os.getenv("S3_BUCKET_NAME")
    test_path    = os.getenv("TEST_PARQUET_PATH")  # e.g. s3://…/processed/test_cleaned/

    if not all([tracking_uri, experiment, bucket, test_path]):
        raise ValueError(
            "Please set MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, "
            "S3_BUCKET_NAME, and TEST_PARQUET_PATH in your .env"
        )

    # ── 2) Connect to MLflow and pick best run by recall ────────────────────
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    exp = mlflow.get_experiment_by_name(experiment)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.recall DESC"],
        max_results=1
    )
    if runs.empty:
        raise RuntimeError(f"No runs found for experiment '{experiment}'")

    best = runs.iloc[0]
    run_id       = best["run_id"]
    model_art    = best["params.model"]
    print(f"✅ Best run_id={run_id}, artifact_path={model_art}")

    # ── 3) Load your full Pipeline from MLflow ──────────────────────────────
    model_uri = f"runs:/{run_id}/{model_art}"
    pipeline  = mlflow.sklearn.load_model(model_uri)

    # ── 4) Read cleaned test data from S3 ───────────────────────────────────
    print(f"⏬ Loading test data from {test_path}")
    df_test = pd.read_parquet(test_path)
    features = [
        'age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
        'total_spend', 'gender_index', 'last_interaction',
        'subscription_type_index', 'contract_length_index'
    ]
    X_test = df_test[features]

    # ── 5) Create a SHAP Explainer over the Pipeline’s predict_proba → positive class ───
    print("🧮 Initializing SHAP Explainer on pipeline.predict_proba[...]")
    # we explain the probability of the positive (1) class:
    f = lambda data: pipeline.predict_proba(data)[:, 1]
    explainer    = shap.Explainer(f, X_test)
    shap_values  = explainer(X_test)   # returns a shap.Explanation

    # ── 6) Plot and save a summary of per-feature impacts ─────────────────────
    print("📊 Generating SHAP summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_values.values,         # an (n_samples, n_features) array of shap values
        X_test,                     # your feature DataFrame
        feature_names=features,
        show=False
    )
    plt.tight_layout()
    out_path = "shap_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"✅ SHAP summary plotted to {out_path}")

if __name__ == "__main__":
    main()
