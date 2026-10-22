# Fraud Detection MLOps 

An end-to-end machine learning project for detecting fraudulent credit card transactions, with experiment tracking, a REST API, automated testing, containerization, and multi-platform Docker image publishing.

## Overview

This project demonstrates how to take a machine learning model beyond experimentation and prepare it for deployment using practical MLOps workflows.

The system trains a fraud classification model, tracks experiments with MLflow, serves predictions through FastAPI, tests API behavior with pytest, and uses GitHub Actions to automate testing and Docker image publishing.

### Key Features
- **Machine Learning:** Random Forest classifier with preprocessing and SMOTE for imbalanced data.
- **Data Leakage Prevention:** Train-test splitting before fitting preprocessing and applying resampling.
- **Experiment Tracking:** MLflow experiment tracking and metric logging.
- **REST API:** FastAPI endpoints for health checks and fraud predictions.
- **Input Validation:** Pydantic request validation.
- **Automated Testing:** Pytest tests for API endpoints and invalid inputs.
- **Containerization:** Docker image with a dedicated runtime dependency file.
- **CI/CD:** GitHub Actions workflow for automated tests, image builds, and publishing.
- **Multi-Platform Images:** Docker images for linux/amd64 and linux/arm64.
- **Container Registry:** Published images hosted on GitHub Container Registry (GHCR).


_Cloud deployment is a planned next step. The published container has been verified locally._

## Tech Stack
Category                                            Technologies
Language                                            Python 3.12
Machine Learning                                    scikit-learn, imbalanced-learn
Data Processing                                     pandas, NumPy, SciPy
Experiment Tracking                                 MLflow
API                                                 FastAPI, Uvicorn, Pydantic
Testing                                             pytest, FastAPI TestClient
Containerization                                    Docker, Docker Buildx
CI/CD                                               GitHub Actions
Container Registry                                  GitHub Container Registry
Serialization                                       joblib

## Project Structure
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   ├── train.py
│   └── api.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-api.txt
├── fraud_model.joblib
├── creditcard.csv
├── mlflow.db
└── README.md


The dataset, local MLflow database, and Python virtual environment are excluded from Git. The trained model artifact is included because the inference API requires it.


## Machine Learning Pipeline

The project uses the Credit Card Fraud Detection dataset, containing 284,807 transactions and 31 columns.

The target column is Class:
    - 0: Legitimate transaction
    - 1: Fraudulent transaction


### Training workflow
1. Load and inspect the dataset.
2. Separate features from the target.
3. Split the data into training and test sets.
4. Standardize Time and Amount using StandardScaler.
5. Apply SMOTE to the training data within an imbalanced-learn pipeline.
6. Train a Random Forest classifier.
7. Evaluate the model using precision, recall, F1-score, and average precision.
8. Log the experiment with MLflow.
9. Save the trained pipeline to fraud_model.joblib.

SMOTE and preprocessing are fitted within the training pipeline to help prevent information leakage from the test set.

### Model configuration
The current recorded model uses:
`RandomForestClassifier(`
    `n_estimators=50,`
   ` max_depth=15,`
  `  random_state=42,`
 `   n_jobs=-1`
`)`

### Recorded evaluation results
Metric                  Results
True negatives          56,849
False positives         15
False negatives         15
True positives          83
Fraud precision         84.69%
Fraud recall            84.69%
Fraud F1-score          84.69%
Average precision       88.25%


These are results from the recorded evaluation run, not a guarantee of real-world performance. Further validation is required before using the model for real financial decisions.

## MLflow Experiment Tracking

MLflow tracks the training experiment and evaluation metrics.

Current local configuration:
    - Experiment name: Fraud Detection
    - Tracking URI: sqlite:///mlflow.db
    - Model artifact: fraud_model.joblib

Start the MLflow UI from the project root:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`

Open http://localhost:5000 to inspect the experiment.

The tracking database is local and is not currently deployed as a shared MLflow tracking server.


## Getting Started

### Prerequisites
Install the following:
    - Python 3.12
    - Git
    - Docker Desktop, if you want to run the container
    - The credit card fraud dataset

### 1. Clone the repository
`git clone https://github.com/Ruchi0903/fraud-detection-mlops.git`
`cd fraud-detection-mlops`

### 2. Create a virtual environment
macOs and Linux:
`python3.12 -m venv .venv`
`source .venv/bin/activate`

Install the project dependencies:
`python -m pip install --upgrade pip`
`pip install -r requirements.txt`
`pip install pytest`

### 3. Download the dataset
Download creditcard.csv from the Kaggle Credit Card Fraud Detection dataset.
Place the file in the project root:
fraud-detection-mlops/
└── creditcard.csv

The dataset is not committed to the repository.

### 4. Train the model
From the project root, run:
python src/train.py

The script trains and evaluates the model, logs the experiment to MLflow, and saves the trained artifact as fraud_model.joblib.

### 5. Start the API locally
`uvicorn src.api:app --reload`

The API should be available at:
    - Root endpoint: http://localhost:8000/
    - Health check: http://localhost:8000/health
    - Interactive API documentation: http://localhost:8000/docs

The API requires an existing fraud_model.joblib file in the project root.


## API Reference

### GET /health
Checks whether the API is responding.

### POST /predict
Accepts a transaction containing Time, Amount, and features V1 through V28.
The response contains a binary prediction, a label (legitimate or fraud), and a fraud probability.

Invalid input, such as a negative Amount, is rejected by request validation.

## Automated Testing
Run the API test suite:
`python -m pytest -v`

The current tests cover:
1. Root endpoint behavior.
2. Health endpoint behavior.
3. Prediction response structure and probability range.
4. Rejection of a negative transaction amount.
5. These tests are executed by the GitHub Actions workflow.

## Docker

The Docker image packages the FastAPI application, runtime dependencies, and trained model artifact.

The Dockerfile uses requirements-api.txt to avoid installing the full training and experiment-tracking environment into the serving container.

### Build locally
After generating fraud_model.joblib:
`docker build --no-cache -t fraud-detection-api .`

### Run locally
`docker run --rm -p 8000:8000 fraud-detection-api`
If port 8000 is already occupied, map a different host port:
`docker run --rm -p 8001:8000 fraud-detection-api`

Then visit:
    - http://localhost:8001/health
    - http://localhost:8001/docs

### Published Docker image
The published image is available at:
`ghcr.io/ruchi0903/fraud-detection-mlops:latest`

Pull it:
`docker pull ghcr.io/ruchi0903/fraud-detection-mlops:latest`

Run it:
`docker run --rm -p 8001:8000 ghcr.io/ruchi0903/fraud-detection-mlops:latest`

The image includes variants for:
    - linux/amd64
    - linux/arm64

This supports typical Intel/AMD Linux servers and ARM64 systems, including Apple Silicon Macs.

Inspect the published image:
`docker buildx imagetools inspect ghcr.io/ruchi0903/fraud-detection-mlops:latest`

## CI/CD Pipeline

The workflow is defined in .github/workflows/tests.yml.

It is configured for:
    - Pushes to main
    - Pull requests targeting main
    - Manual execution through workflow_dispatch

### Workflow steps
1. Check out the source code.
2. Set up Python 3.12.
3. Install dependencies.
4. Run API tests.
5. Configure Docker Buildx.
6. Authenticate to GHCR for push-triggered runs.
7. Build Docker images for AMD64 and ARM64.
8. Publish images to GHCR on push events.

The workflow uses GitHub's automatically provided GITHUB_TOKEN with package write permissions. No hardcoded registry credentials are required.

Manual workflow runs execute the workflow but do not publish an image under the current push-only publishing condition.

### Image tags
The workflow publishes:
`ghcr.io/ruchi0903/fraud-detection-mlops:latest`
`ghcr.io/ruchi0903/fraud-detection-mlops:sha-<commit-sha>`

The latest tag is convenient for pulling the most recently published image. Commit-specific tags help identify and reproduce a particular build.

## Current Status

### Implemented
1. Model training and evaluation
2. Preprocessing and SMOTE pipeline
3. MLflow experiment tracking
4. FastAPI inference service
5. API input validation
6. Automated API tests
7. Docker containerization
8. Separate runtime dependencies
9. GitHub Actions CI/CD workflow
10. GHCR image publishing
11. Multi-platform Docker builds
12. Local execution of the published image on Apple Silicon
13. Verification of the API health and docs endpoints

### Planned
- Cloud deployment with a public HTTPS endpoint
- Automated deployment after successful CI/CD checks
- API metrics and observability
- Data and prediction drift monitoring
- Model registry and controlled model promotion
-Automated retraining with evaluation gates
- Infrastructure as code

### Future Improvements
The next development stages will focus on:
1. Deploying the published container to a managed cloud platform.
2. Verifying inference through the published image and deployed service.
3. Improving release and rollback workflows.
4. Adding monitoring for latency, failures, prediction volume, and data drift.
5. Introducing model versioning and controlled promotion.
6. Automating retraining with data validation and model evaluation safeguards.

## Disclaimer

This project is intended for learning and demonstration. The model has been evaluated on a benchmark dataset and is not validated for production financial decision-making. Do not use its predictions as the sole basis for blocking transactions or making financial decisions.