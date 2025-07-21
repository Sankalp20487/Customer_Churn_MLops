# model_loader.py

import os
import boto3
import joblib
import tempfile

def load_model():
    # ── Read config from ENV ────────────────────────────────
    bucket = os.environ["S3_BUCKET_NAME"]
    key    = os.environ["FINAL_MODEL_PATH"]  # e.g. "models/Best_model.pkl"
    
    # ── Download artifact from S3 ───────────────────────────
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        s3.download_fileobj(bucket, key, tmp)
        tmp_path = tmp.name

    # ── Load & cleanup ───────────────────────────────────────
    model = joblib.load(tmp_path)
    os.remove(tmp_path)
    return model
