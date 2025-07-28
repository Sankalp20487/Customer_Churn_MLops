#!/usr/bin/env python

import os
import time
from dotenv import load_dotenv
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# ─────────────────────────────────────────────
# ✅ Load all config from .env into os.environ
# ─────────────────────────────────────────────
load_dotenv()  # make sure you have a .env file at your repo root

bucket_name = os.environ["S3_BUCKET_NAME"]
role_arn    = os.environ["TRAINING_ROLE_ARN"]

# ─────────────────────────────────────────────
# ✅ Build the “dummy” channel & job name
# ─────────────────────────────────────────────
input_data = {"dummy": f"s3://{bucket_name}/dummy/"}
job_name   = f"churn-training-{int(time.time())}"

# ─────────────────────────────────────────────
# ✅ Create the SageMaker SKLearn Estimator
# ─────────────────────────────────────────────
sklearn_estimator = SKLearn(
    entry_point="train.py",         # inside modeling/
    source_dir="modeling",          # train.py + requirements.txt live here
    role=role_arn,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.3-1",
    py_version="py3",
    job_name=job_name,
    base_job_name="churn-training",
    sagemaker_session=sagemaker.Session(),
    output_path=f"s3://{bucket_name}/output/",
    environment={
        "S3_BUCKET_NAME":        bucket_name,
        "TRAIN_PARQUET_PATH":    os.environ["TRAIN_PARQUET_PATH"],
        "TEST_PARQUET_PATH":     os.environ["TEST_PARQUET_PATH"],
        "MLFLOW_TRACKING_URI":   os.environ["MLFLOW_TRACKING_URI"],
        "MLFLOW_EXPERIMENT_NAME":os.environ["MLFLOW_EXPERIMENT_NAME"],
        "MONITORING_LOGS_DIR":   os.environ["MONITORING_LOGS_DIR"],
    }
)

# ─────────────────────────────────────────────
# ✅ Launch!
# ─────────────────────────────────────────────
sklearn_estimator.fit(input_data, logs="All")
print(f"✅ Training job submitted: {sklearn_estimator.latest_training_job.job_name}")
