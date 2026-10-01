# Credit Score Classification on AWS (SageMaker + EC2)

An end-to-end machine learning project that classifies customers into **Poor**, **Standard**, or **Good** credit score categories and serves the model on AWS. The model is trained locally, packaged as `model.tar.gz`, deployed to a **Amazon SageMaker real-time endpoint**, and consumed by a **Streamlit** web app hosted on **Amazon EC2**.

## Architecture

```
┌──────────────┐   train    ┌───────────────┐  tar.gz   ┌──────────┐
│ pipeline_aws │ ─────────▶ │ best_model.pkl│ ────────▶ │ Amazon S3│
└──────────────┘            └───────────────┘           └────┬─────┘
                                                             │ model_data
                                                             ▼
┌──────────────┐  invoke_endpoint  ┌───────────────────────────────┐
│ Streamlit on │ ────────────────▶ │ SageMaker Endpoint            │
│ Amazon EC2   │ ◀──────────────── │ (inference.py, ml.m5.large)   │
└──────────────┘   JSON response   └───────────────────────────────┘
```

## Features

- **Modular training pipeline**: ingestion → preprocessing → training → evaluation
- **Leakage-safe preprocessing**: outlier bounds, medians, and loan vocabularies are fitted on the training split only
- **3 models compared**: Random Forest, XGBoost, CatBoost, all with class-imbalance handling
- **Automated best-model selection** by stratified 5-fold CV macro-F1
- **Deployment approval gate**: the model is approved only if it passes all metric thresholds
- **SageMaker inference script** with custom `model_fn`, `input_fn`, `predict_fn`, and `output_fn`
- **EC2 bootstrap script** that installs and runs the Streamlit app as a `systemd` service

## Project Structure

```
.
├── data_B.csv                 # Raw dataset
├── data_ingestion_aws.py      # Step 1: load & validate raw data
├── preprocessing_aws.py       # Step 2: cleaning, feature engineering, train/test split
├── train_aws.py               # Step 3: training + stratified 5-fold CV
├── evaluation_aws.py          # Step 4: test-set evaluation & best model selection
├── pipeline_aws.py            # Orchestrator: runs all steps + approval check
├── inference.py               # SageMaker inference handler
├── deploy_endpoint.ipynb      # Package model, upload to S3, deploy & delete endpoint
├── app_streamlit_aws.py       # Streamlit app that calls the SageMaker endpoint
├── user-data.sh               # EC2 bootstrap script (installs & starts the app)
├── exploration.ipynb          # Exploratory data analysis
├── requirements.txt
├── artifact/                  # best_model.pkl (generated)
└── ingested/                  # ingested data copy (generated)
```

## Pipeline Overview

| Step | Module | What it does |
|------|--------|--------------|
| 1. Ingestion | `data_ingestion_aws.py` | Reads the raw CSV, checks it is not empty and contains the `Credit_Score` target, then saves a copy to `ingested/` |
| 2. Preprocessing | `preprocessing_aws.py` | Cleans noisy values, engineers features, imputes, and performs a stratified 80/20 train/test split |
| 3. Training | `train_aws.py` | Trains 3 models inside a sklearn `Pipeline` with stratified 5-fold CV |
| 4. Evaluation | `evaluation_aws.py` | Evaluates each model on the held-out test set and selects the best one by CV macro-F1 |
| 5. Approval | `pipeline_aws.py` | Saves the best model to `artifacts/best_model.pkl` and runs the threshold checks |

### Preprocessing Highlights

- Strips junk characters from numeric columns (e.g. `5_` → `5`) and converts them to numbers
- Replaces placeholder noise values (`_______`, `_`, `!@9#%8`) with missing values
- Converts `Credit_History_Age` (e.g. "3 Years and 2 Months") to months
- Splits `Payment_Behaviour` into `Spent_Level` and `Payment_Value`
- Parses `Type_of_Loan` into per-loan-type frequency features
- Caps extreme outliers (Q3 + 3×IQR) and caps `Total_EMI_per_month` at the 99th percentile
- Imputes income-related columns using the median per `Occupation`
- Final transformer: median imputation + `RobustScaler` (numeric), one-hot (nominal), ordinal encoding (ordered categories)
- Drops identifiers and non-informative columns (`ID`, `Customer_ID`, `Name`, `SSN`, `Month`, `Num_of_Loan`)

### Models

| Model | Imbalance handling |
|-------|--------------------|
| Random Forest | `class_weight='balanced'` |
| XGBoost | balanced `sample_weight` |
| CatBoost | `auto_class_weights='Balanced'` |

### Deployment Approval Gate

The best model must meet **all** thresholds to be approved:

| Metric | Threshold |
|--------|:---------:|
| Macro F1 | ≥ 0.65 |
| Recall (Poor) | ≥ 0.70 |
| Precision (Good) | ≥ 0.58 |

> Recall on **Poor** is prioritized to catch risky customers, while precision on **Good** limits the chance of wrongly approving them.

## Getting Started

### Prerequisites

- Python 3.11+
- An AWS account with access to S3, SageMaker, and EC2 (the notebook uses an IAM role named `LabRole`)

### 1. Train the model locally

```bash
git clone https://github.com/aullya-nadine-kuswandi/CreditScorePredictionAWS.git
cd CreditScorePredictionAWS

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python pipeline_aws.py
```

This trains all three models, prints the evaluation report, saves the best one to `artifacts/best_model.pkl`, and prints the approval decision.

### 2. Deploy to a SageMaker endpoint

Open `deploy_endpoint.ipynb` in SageMaker Studio / a SageMaker notebook and run the cells in order:

1. Verify the execution role and list your S3 buckets
2. Package `artifacts/best_model.pkl` into `model.tar.gz`
3. Upload it to S3 (set `bucket_name` / `BUCKET` to your own bucket)
4. Test that `inference.py` can load the model
5. Deploy the endpoint (`ml.m5.large`, scikit-learn `1.4-2`, takes about 5–8 minutes) and run a smoke test

### 3. Host the Streamlit app on EC2

1. Launch an EC2 instance (Amazon Linux 2023) and attach an **IAM instance profile** that allows `sagemaker:InvokeEndpoint`
2. Open inbound port **8501** in the instance's security group
3. Paste `user-data.sh` into **Advanced details → User data** and edit these variables first:

| Variable | Description |
|----------|-------------|
| `GIT_REPO` | URL of this repository |
| `SUBFOLDER` | Subfolder containing the app (leave empty if it is in the repo root) |
| `APP_FILE` | `app_streamlit_aws.py` |
| `ENDPOINT_NAME` | Name of your SageMaker endpoint (default: `credit-score-endpoint`) |
| `REGION` | AWS region (default: `us-east-1`) |

4. Once the instance is running, open `http://<EC2-public-IP>:8501`

### Run the Streamlit app locally (optional)

With AWS credentials configured (`~/.aws/credentials`) and the endpoint running:

```bash
export ENDPOINT_NAME=credit-score-endpoint
export AWS_REGION=us-east-1
streamlit run app_streamlit_aws.py
```

## Endpoint API

**Request** (`POST`, `application/json`): one list of 30 feature values per instance, in the order defined by `FEATURE_NAMES` in `inference.py`.

```json
{
  "instances": [
    [18, "Entrepreneur", 62717.24, 5035.44, 6, 6, 29, 54, 22, 1.82, 7,
     "Bad", 2590.63, 31.9, 218, "Yes", 370.02, 42.6, 330.92, "High", "Large",
     0, 1, 1, 3, 2, 1, 1, 0, 0]
  ]
}
```

**Response**:

```json
{
  "probabilities": [[0.0, 0.0, 0.0]],
  "predictions": [0],
  "labels": ["Poor"]
}
```

`probabilities` follow the class order `[Poor, Standard, Good]`.

## Clean Up

SageMaker endpoints are billed while they are running. When you are done, delete the endpoint and its configuration with the last cell of `deploy_endpoint.ipynb`, and stop or terminate the EC2 instance.

## Tech Stack

- **Data & ML**: pandas, NumPy, scikit-learn, XGBoost, CatBoost
- **Cloud**: Amazon S3, Amazon SageMaker, Amazon EC2, boto3
- **App & visualization**: Streamlit, Plotly

## Notes

- All random seeds are fixed (`random_state=42`) for reproducibility.
- The training environment and the SageMaker container should use matching scikit-learn versions (`requirements.txt` pins `1.4.2` for the `1.4-2` container).
- `pipeline_aws.py` expects the raw dataset at `data_B.csv` in the project root.
