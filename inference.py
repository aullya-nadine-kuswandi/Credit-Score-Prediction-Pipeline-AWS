import json
import os
import joblib
import numpy as np
import pandas as pd


JSON_CONTENT_TYPE = "application/json"

CLASS_NAMES = ["Poor", "Standard", "Good"]

FEATURE_NAMES = [
    "Age",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Credit_History_Age",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Spent_Level",
    "Payment_Value",
    "personal loan_freq",
    "home equity loan_freq",
    "auto loan_freq",
    "mortgage loan_freq",
    "payday loan_freq",
    "not specified_freq",
    "credit-builder loan_freq",
    "student loan_freq",
    "debt consolidation loan_freq",
]


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "best_model.pkl"))


def input_fn(request_body, request_content_type):
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, pipeline):
    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction, accept_content_type):
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
