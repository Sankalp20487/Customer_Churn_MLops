# modeling/train.py

import os
import warnings
import boto3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from contextlib import nullcontext
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────────
# 👉 Make sure the intermediate output dir exists (so plt.savefig never crashes)
# ────────────────────────────────────────────────────────────────────────────────
INTERMEDIATE_DIR = os.environ.get("SM_OUTPUT_INTERMEDIATE_DIR", "/opt/ml/output/intermediate")
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# ✅ Load ENV Vars (from SageMaker env, not .env)
# ─────────────────────────────────────────────
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow_experiment  = os.environ.get("MLFLOW_EXPERIMENT_NAME")

if mlflow_tracking_uri and mlflow_experiment:
    import mlflow, mlflow.sklearn
    from mlflow.models.signature import infer_signature

    # send all tracking calls to your ARN-based server...
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # ...but force the model‐registry APIs onto a simple local file backend,
    # so that any registry calls (e.g. set_experiment()) won’t try to use "arn:" as a scheme.
    mlflow.set_registry_uri(f"file://{INTERMEDIATE_DIR}/mlflow_registry")

    # now it’s safe to set the experiment
    mlflow.set_experiment(mlflow_experiment)

bucket_name    = os.environ["S3_BUCKET_NAME"]
train_path     = os.environ["TRAIN_PARQUET_PATH"]
test_path      = os.environ["TEST_PARQUET_PATH"]
monitoring_dir = os.environ.get("MONITORING_LOGS_DIR", "monitoring_logs")

s3 = boto3.client("s3")

features = [
    'age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay',
    'total_spend', 'gender_index', 'last_interaction',
    'subscription_type_index', 'contract_length_index'
]

# ─────────────────────────────────────────────
# ✅ Load Data from S3
# ─────────────────────────────────────────────
train_df = pd.read_parquet(train_path)
test_df  = pd.read_parquet(test_path)

X_train, y_train = train_df[features], train_df["churn"]
X_test,  y_test  = test_df[features],  test_df["churn"]

# ─────────────────────────────────────────────
# ✅ Model Config
# ─────────────────────────────────────────────
models = {
    "LogisticRegression": {
        "estimator": LogisticRegression(),
        "params":    {"classifier__C": [0.1, 1, 10]}
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(),
        "params": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [5, 10]}
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(),
        "params": {"classifier__learning_rate": [0.05, 0.1], "classifier__n_estimators": [100, 200]}
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(),
        "params": {"classifier__n_estimators": [50, 100]}
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(),
        "params": {"classifier__max_depth": [5, 10, None]}
    },
    "XGBoost": {
        "estimator": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "params": {"classifier__learning_rate": [0.05, 0.1], "classifier__n_estimators": [100, 200]}
    },
    "NaiveBayes": {
        "estimator": GaussianNB(),
        "params": {}
    }
}

leaderboard = []

# ─────────────────────────────────────────────
# ✅ Training Loop
# ─────────────────────────────────────────────
for model_name, config in models.items():
    print(f"\n▶️  Training {model_name}")

    run_mlflow = bool(mlflow_tracking_uri and mlflow_experiment)
    ctx = mlflow.start_run(run_name=model_name) if run_mlflow else nullcontext()

    with ctx:
        # 1) fit
        pipe = Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", config["estimator"])
        ])
        clf = GridSearchCV(pipe, config["params"], cv=3, scoring="accuracy")
        clf.fit(X_train, y_train)

        # 2) predict & score
        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec  = recall_score(y_test, preds)
        f1   = f1_score(y_test, preds)

        print(f"✔️  {model_name} — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # 3) log params & metrics
        if run_mlflow:
            mlflow.log_param("model", model_name)
            mlflow.log_params(clf.best_params_)
            mlflow.log_metrics({
                "accuracy":  acc,
                "precision": prec,
                "recall":    rec,
                "f1_score":  f1
            })

            sig = infer_signature(X_train, clf.predict(X_train))
            mlflow.sklearn.log_model(
                clf.best_estimator_,
                artifact_path=model_name,        # still writes the pickle artifact
                signature=sig,
                input_example=X_train.iloc[:5]
                # <- NOTE no `registered_model_name=` argument, so registry is never invoked
            )

        # 4) confusion matrix
        cm      = confusion_matrix(y_test, preds)
        cm_fig, cm_ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=cm_ax)
        plt.title(f"Confusion Matrix – {model_name}")
        cm_path = os.path.join(INTERMEDIATE_DIR, f"conf_matrix_{model_name}.png")
        cm_fig.savefig(cm_path)
        plt.close(cm_fig)
        if run_mlflow:
            mlflow.log_artifact(cm_path)

        # 5) ROC curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc     = auc(fpr, tpr)
        roc_fig, roc_ax = plt.subplots()
        roc_ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        roc_ax.plot([0,1], [0,1], "k--")
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.set_title(f"ROC Curve – {model_name}")
        roc_ax.legend(loc="lower right")
        roc_path = os.path.join(INTERMEDIATE_DIR, f"roc_curve_{model_name}.png")
        roc_fig.savefig(roc_path)
        plt.close(roc_fig)
        if run_mlflow:
            mlflow.log_artifact(roc_path)

        # 6) feature importances (if available)
        try:
            importances = clf.best_estimator_.named_steps["classifier"].feature_importances_
            fi_fig, fi_ax = plt.subplots()
            sns.barplot(x=importances, y=features, ax=fi_ax)
            fi_ax.set_title(f"Feature Importance – {model_name}")
            fi_path = os.path.join(INTERMEDIATE_DIR, f"feat_importance_{model_name}.png")
            fi_fig.savefig(fi_path)
            plt.close(fi_fig)
            if run_mlflow:
                mlflow.log_artifact(fi_path)
        except Exception:
            pass

        # 7) monitoring logs → S3
        tn, fp, fn, tp = cm.ravel()
        run_date  = datetime.now().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        monitor_df = pd.DataFrame([{
            "date":            run_date,
            "model_name":      model_name,
            "accuracy":        acc,
            "precision":       prec,
            "recall":          rec,
            "f1_score":        f1,
            "tp":              tp,
            "fp":              fp,
            "tn":              tn,
            "fn":              fn,
            "actual_ratio":    y_test.sum() / len(y_test),
            "predicted_ratio": preds.sum() / len(preds),
            "test_sample_size":     len(y_test),
            "train_sample_size":     len(y_train)
        }])
        buf = BytesIO()
        monitor_df.to_parquet(buf, index=False)
        buf.seek(0)
        s3.upload_fileobj(buf, bucket_name, f"{monitoring_dir}/{model_name}_metrics_{timestamp}.parquet")

        leaderboard.append({
            "Model":     model_name,
            "Accuracy":  acc,
            "Precision": prec,
            "Recall":    rec,
            "F1 Score":  f1
        })

# ─────────────────────────────────────────────
# ✅ Show leaderboard
# ─────────────────────────────────────────────
results_df = pd.DataFrame(leaderboard).sort_values(by="F1 Score", ascending=False)
print(results_df)
