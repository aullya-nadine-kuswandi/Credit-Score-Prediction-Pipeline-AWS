import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
import re
from pathlib import Path
from collections import Counter

class CreditPreprocessor:
    def __init__(self, test_size: float = 0.2):
        self.test_size = test_size
        self.random_state = 42

        self.loan_cols_ = None
        self.ub_bounds_: dict = {}
        self.emi_cap_ = None
        self.occ_median_: dict = {}
        self.global_median_: dict = {}
        self.num_cols_ = None

    @staticmethod
    def strip_numeric(series: pd.Series):
        return (series.astype(str)
                      .str.replace(r'[^0-9.\-]', '', regex=True)
                      .str.replace(r'(?<!^)-', '', regex=True))

    @staticmethod
    def parse_loan_types(x):
        if pd.isna(x):
            return Counter()
        x = x.replace(' and ', ',')
        loans = [l.strip().lower() for l in x.split(',')]
        loans = [l for l in loans if l]
        return Counter(loans)

    def make_loan_freq(self, s: pd.Series):
        return (s.apply(lambda x: pd.Series(self.parse_loan_types(x)))
                 .fillna(0).astype(int).add_suffix('_freq'))

    @staticmethod
    def credit_history_to_months(s):
        if pd.isna(s):
            return np.nan
        m = re.findall(r'(\d+)', str(s))
        if len(m) >= 2:
            return int(m[0]) * 12 + int(m[1])
        return np.nan

    def basic_clean(self, x: pd.DataFrame):
        x = x.copy()

        # Kolom numerik yang mengandung karakter sampah (contoh '5_')
        numeric_str_cols = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
                            'Changed_Credit_Limit', 'Outstanding_Debt',
                            'Amount_invested_monthly', 'Monthly_Balance']
        for col in numeric_str_cols:
            x[col] = pd.to_numeric(self.strip_numeric(x[col]), errors='coerce')

        # Unknown Category
        noise = {
            'Occupation': ['_______'],
            'Credit_Mix': ['_'],
            'Payment_Behaviour': ['!@9#%8'],
        }
        for col, vals in noise.items():
            x[col] = x[col].replace(vals, np.nan)

        # Convert Credit_History_Age ke bulan
        x['Credit_History_Age'] = x['Credit_History_Age'].apply(self.credit_history_to_months)

        # Pemisahan kolom Payment_Behaviour
        x['Spent_Level'] = x['Payment_Behaviour'].str.extract(r'(Low|High)_spent')[0]
        x['Payment_Value'] = x['Payment_Behaviour'].str.extract(r'spent_(Small|Medium|Large)_value')[0]
        x = x.drop(columns='Payment_Behaviour')

        # Invalid value berbasis logika (fixed bound)
        x['Age'] = x['Age'].where(x['Age'].between(0, 110), np.nan)
        x['Monthly_Balance'] = x['Monthly_Balance'].where(x['Monthly_Balance'] >= 0, np.nan)

        return x

    def apply_bounds(self, x: pd.DataFrame):
        x = x.copy()
        for c, (lo, hi) in self.ub_bounds_.items():
            x[c] = x[c].where(x[c].between(lo, hi), np.nan)
        x['Total_EMI_per_month'] = x['Total_EMI_per_month'].where(
            x['Total_EMI_per_month'] <= self.emi_cap_, np.nan)
        return x

    def group_impute(self, x: pd.DataFrame):
        x = x.copy()
        for c in self.occ_median_:          
            fill = x['Occupation'].map(self.occ_median_[c]).fillna(self.global_median_[c])
            x[c] = x[c].fillna(fill)
        return x

    def fit_transform_features(self, x: pd.DataFrame):
        x = self.basic_clean(x)

        # Loan frequency (fit vocabulary dari train)
        loan = self.make_loan_freq(x['Type_of_Loan'])
        self.loan_cols_ = loan.columns
        x = pd.concat([x.drop(columns='Type_of_Loan'), loan], axis=1)

        # Num_of_Loan redundan dengan total loan sehingga di drop
        x = x.drop(columns='Num_of_Loan')

        # Fit bound extreme outlier (Q3 + 3*IQR, lower bound 0)
        ub_cols = ['Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
                   'Interest_Rate', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']
        for c in ub_cols:
            q1, q3 = x[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.ub_bounds_[c] = (0, q3 + 3 * iqr)
        self.emi_cap_ = x['Total_EMI_per_month'].quantile(0.99)
        x = self.apply_bounds(x)

        # Month tidak relevan untuk prediksi Credit_Score sehingga didrop
        x = x.drop(columns='Month')

        # Fit median per Occupation untuk imputasi
        group_cols = ['Annual_Income', 'Monthly_Inhand_Salary']
        for c in group_cols:
            self.occ_median_[c] = x.groupby('Occupation')[c].median()
            self.global_median_[c] = x[c].median()
        x = self.group_impute(x)

        return x

    def transform_features(self, x: pd.DataFrame):
        x = self.basic_clean(x)

        loan = (self.make_loan_freq(x['Type_of_Loan'])
                .reindex(columns=self.loan_cols_, fill_value=0))
        x = pd.concat([x.drop(columns='Type_of_Loan'), loan], axis=1)

        x = x.drop(columns='Num_of_Loan')
        x = self.apply_bounds(x)
        x = x.drop(columns='Month')
        x = self.group_impute(x)

        return x

    def clean_and_split(self, data_path: str | Path):
        df = pd.read_csv(Path(data_path))

        # Drop kolom identifier yang tidak informatif untuk prediksi
        df = df.drop(columns=['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN'])

        # Encode target
        target_map = {'Poor': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Score'] = df['Credit_Score'].map(target_map).astype(int)

        X = df.drop(columns=['Credit_Score'])
        y = df['Credit_Score']

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        x_train = self.fit_transform_features(x_train)
        x_test = self.transform_features(x_test)

        return x_train, x_test, y_train, y_test

    def get_transformer(self, x_train: pd.DataFrame):
        num_cols = x_train.select_dtypes(include='number').columns.tolist()
        self.num_cols_ = num_cols

        numeric_preprocess = Pipeline([
            ('num_imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
        ])

        ohe_cols = ['Occupation', 'Payment_of_Min_Amount']
        ohe_preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('cat_onehot_encoder', OneHotEncoder(handle_unknown='ignore')),
        ])

        ordinal_cols = ['Credit_Mix', 'Spent_Level', 'Payment_Value']
        ordinal_categories = [
            ['Bad', 'Standard', 'Good'],
            ['Low', 'High'],
            ['Small', 'Medium', 'Large'],
        ]
        ordinal_preprocess = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('cat_ordinal_encoder', OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown='use_encoded_value', unknown_value=-1)),
        ])

        return ColumnTransformer([
            ('num', numeric_preprocess, num_cols),
            ('ohe', ohe_preprocess, ohe_cols),
            ('ordinal', ordinal_preprocess, ordinal_cols),
        ], remainder='drop')

    def run(self, data_path: str | Path):
        print("--- Step 2: Preprocessing ---")
        x_train, x_test, y_train, y_test = self.clean_and_split(data_path)
        transformer = self.get_transformer(x_train)
        print(f"✅ Preprocessing done | train={x_train.shape} test={x_test.shape} "
              f"| numeric={len(self.num_cols_)} features")
        return x_train, x_test, y_train, y_test, transformer
