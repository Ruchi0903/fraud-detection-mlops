
import joblib
import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# ---------- Configuration ----------

DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "creditcard.csv",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
SMOTE_SAMPLING_STRATEGY = 0.1


def main():
    # ---------- Load data ----------

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {int(y.sum())}")

    # ---------- Split data BEFORE preprocessing ----------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------- Build training pipeline ----------

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), ["Time", "Amount"])
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "smote",
                SMOTE(
                    sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=15,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ---------- Train ----------

    print("Training model...")
    model.fit(X_train, y_train)

    # ---------- Save model for FastAPI ----------

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "fraud_model.joblib",
    )

    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # ---------- Evaluate on untouched test data ----------

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "fraud_precision": precision_score(y_test, y_pred),
        "fraud_recall": recall_score(y_test, y_pred),
        "fraud_f1": f1_score(y_test, y_pred),
        "average_precision": average_precision_score(y_test, y_prob),
    }

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # ---------- Log experiment to MLflow ----------

    # Use a local SQLite database for experiment metadata.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Fraud Detection")

    with mlflow.start_run(run_name="random_forest_baseline"):
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 50,
                "max_depth": 15,
                "smote_sampling_strategy": SMOTE_SAMPLING_STRATEGY,
                "random_state": RANDOM_STATE,
                "test_size": TEST_SIZE,
            }
        )

        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "project": "fraud-detection-mlops",
                "stage": "baseline",
                "framework": "scikit-learn",
            }
        )

        # Use pickle because this pipeline includes SMOTE.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="fraud_model",
            serialization_format="pickle",
        )

        print("\nMLflow run ID:", mlflow.active_run().info.run_id)

    print("\nTraining and experiment logging completed!")


if __name__ == "__main__":
    main()
