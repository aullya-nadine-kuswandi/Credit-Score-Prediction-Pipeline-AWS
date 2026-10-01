import json
import os
import boto3
import streamlit as st
import plotly.graph_objects as go
from botocore.exceptions import ClientError, NoCredentialsError

st.set_page_config(page_title="Credit Score Prediction", layout="wide")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

CLASS_NAMES = ["Poor", "Standard", "Good"]
COLORS = {'Poor': '#EF553B', 'Standard': '#FFA15A', 'Good': '#00CC96'}

FEATURE_NAMES = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Credit_History_Age", "Payment_of_Min_Amount",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance", "Spent_Level",
    "Payment_Value", "personal loan_freq", "home equity loan_freq", "auto loan_freq",
    "mortgage loan_freq", "payday loan_freq", "not specified_freq",
    "credit-builder loan_freq", "student loan_freq", "debt consolidation loan_freq",
]

OCCUPATIONS = ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer',
               'Entrepreneur', 'Journalist', 'Lawyer', 'Manager', 'Mechanic',
               'Media_Manager', 'Musician', 'Scientist', 'Teacher', 'Writer']
LOAN_FEATURES = [f for f in FEATURE_NAMES if f.endswith('_freq')]
LOAN_LABELS = [c.replace('_freq', '').title() for c in LOAN_FEATURES]


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features):
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


with st.sidebar:
    st.title("Credit Score Prediction")
    st.markdown(
        "This website can help you predict your credit score (Poor/Standard/Good)"
    )

st.header("Credit Score Prediction")

tab1, tab2, tab3 = st.tabs(["Applicant & Income", "Credit Behavior", "Loan Portfolio"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 0, 110, 30)
        occupation = st.selectbox("Occupation", OCCUPATIONS, index=4)
        annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0)
        num_bank = st.number_input("Number of Bank Accounts", 0, 50, 4)
    with c2:
        num_card = st.number_input("Number of Credit Cards", 0, 50, 5)
        outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1200.0, step=50.0)
        total_emi = st.number_input("Total EMI per Month", 0.0, 100000.0, 100.0, step=10.0)
        amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=200.0, step=10.0)
        monthly_balance = st.number_input("Monthly Balance", min_value=0.0, value=400.0, step=10.0)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        interest_rate = st.number_input("Interest Rate (%)", 0, 100, 14)
        delay_due = st.number_input("Delay from Due Date (days)", -30, 200, 15)
        num_delayed = st.number_input("Number of Delayed Payments", 0, 100, 10)
        credit_history = st.number_input("Credit History Age (months)", 0, 1000, 220)
        credit_util = st.number_input("Credit Utilization Ratio (%)", 0.0, 100.0, 20.0, step=0.01)
        credit_mix = st.selectbox("Credit Mix", ['Bad', 'Standard', 'Good'], index=1)
    with c2:
        changed_limit = st.number_input("Changed Credit Limit", min_value=-100.0, value=10.0, step=1.0)
        num_inquiries = st.number_input("Number of Credit Inquiries", 0, 100, 5)
        spent_level = st.radio("Spent Level", ['Low', 'High'], horizontal=True)
        payment_value = st.selectbox("Payment Value", ['Small', 'Medium', 'Large'])
        pay_min = st.selectbox("Payment of Minimum Amount", ['Yes', 'No', 'NM'])

with tab3:
    st.markdown("**Loan Portfolio** (number of each loan type held)")
    loan_values = {}
    cols = st.columns(3)
    for i, (feat, label) in enumerate(zip(LOAN_FEATURES, LOAN_LABELS)):
        with cols[i % 3]:
            loan_values[feat] = st.number_input(label, 0, 20, 0)

st.markdown("---")
predict_clicked = st.button("Predict Credit Score", type="primary", use_container_width=True)

if predict_clicked:
    values = {
        'Age': age, 'Occupation': occupation, 'Annual_Income': annual_income,
        'Monthly_Inhand_Salary': monthly_salary, 'Num_Bank_Accounts': num_bank,
        'Num_Credit_Card': num_card, 'Interest_Rate': interest_rate,
        'Delay_from_due_date': delay_due, 'Num_of_Delayed_Payment': num_delayed,
        'Changed_Credit_Limit': changed_limit, 'Num_Credit_Inquiries': num_inquiries,
        'Outstanding_Debt': outstanding_debt, 'Credit_Utilization_Ratio': credit_util,
        'Credit_History_Age': credit_history, 'Total_EMI_per_month': total_emi,
        'Amount_invested_monthly': amount_invested, 'Monthly_Balance': monthly_balance,
        'Payment_of_Min_Amount': pay_min, 'Credit_Mix': credit_mix,
        'Spent_Level': spent_level, 'Payment_Value': payment_value,
    }
    values.update(loan_values)

    features = [values[name] for name in FEATURE_NAMES]

    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error("AWS credentials tidak ditemukan. Di EC2: pastikan instance profile terpasang. "
                 "Di lokal: konfigurasi ~/.aws/credentials.")
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]              
        proba = {CLASS_NAMES[i]: probs[i] for i in range(len(CLASS_NAMES))}

        st.subheader("Prediction Result")
        col_left, col_right = st.columns(2)

        with col_left:
            banner = f"# {label}"
            if label == 'Good':
                st.success(banner)
            elif label == 'Standard':
                st.warning(banner)
            else:
                st.error(banner)
            st.metric("Confidence", f"{proba[label] * 100:.1f}%")

        with col_right:
            order = ['Poor', 'Standard', 'Good']
            fig = go.Figure(go.Bar(
                x=[proba[k] * 100 for k in order],
                y=order, orientation='h',
                marker_color=[COLORS[k] for k in order],
                text=[f"{proba[k] * 100:.1f}%" for k in order],
                textposition='outside',
            ))
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Probability (%)", xaxis_range=[0, 100],
                height=300, margin=dict(l=10, r=30, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
