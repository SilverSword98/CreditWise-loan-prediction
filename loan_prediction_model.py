import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

# ---------------------------------------------------------
# 1. DATA LOADING & CLEANING
# ---------------------------------------------------------
# Load the dataset
df = pd.read_csv('loan_approval_data.csv')

# Drop completely blank rows (as requested, there are 50 of them)
df.dropna(how='all', inplace=True)

# Map the target variable 'Loan_Approved' from Yes/No to 1/0
df['Loan_Approved'] = df['Loan_Approved'].map({'Yes': 1, 'No': 0})

# Drop rows where the target variable itself is missing (we can't train on these)
df.dropna(subset=['Loan_Approved'], inplace=True)

# Drop Applicant_ID as it has no predictive power
df.drop(columns=['Applicant_ID'], inplace=True, errors='ignore')


# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
# Feature 1: Total_Income 
# (Combining Applicant and Coapplicant income. Fill NaNs with 0 before adding)
df['Applicant_Income'] = df['Applicant_Income'].fillna(0)
df['Coapplicant_Income'] = df['Coapplicant_Income'].fillna(0)
df['Total_Income'] = df['Applicant_Income'] + df['Coapplicant_Income']

# Feature 2: Loan_to_Collateral Ratio
# (High ratio means the loan is risky because it isn't backed by enough collateral)
# Replace 0s in Collateral_Value with NaN to avoid division by zero errors
df['Collateral_Value'] = df['Collateral_Value'].replace(0, np.nan)
df['Loan_to_Collateral'] = df['Loan_Amount'] / df['Collateral_Value']

# Feature 3: Income_to_Loan Ratio
# (Measures the applicant's ability to pay back the loan based on total income)
df['Loan_Amount'] = df['Loan_Amount'].replace(0, np.nan)
df['Income_to_Loan'] = df['Total_Income'] / df['Loan_Amount']


# ---------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# ---------------------------------------------------------
# Separate features (X) and target (y)
X = df.drop(columns=['Loan_Approved'])
y = df['Loan_Approved']

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Create preprocessing steps for numerical data (Impute missing with median, then scale)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Create preprocessing steps for categorical data (Impute missing with mode, then one-hot encode)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a single preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])


# ---------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------
# Split the data (80% train, 20% test). Stratify ensures the 1/0 ratio is maintained.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model A: Logistic Regression (Baseline)
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)

# Model B: Random Forest (Tree-based)
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])
rf_pipeline.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. BUSINESS EVALUATION
# ---------------------------------------------------------
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"--- {name} Evaluation ---")
    
    # Confusion Matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives (Correctly Rejected): {cm[0][0]}")
    print(f"False Positives (High-Risk Approved - Financial Loss): {cm[0][1]}")
    print(f"False Negatives (Good Customers Rejected - Lost Business): {cm[1][0]}")
    print(f"True Positives (Correctly Approved): {cm[1][1]}")
    
    # Business Metrics
    print(f"\nPrecision: {precision_score(y_test, y_pred):.4f} (Higher means fewer bad loans approved)")
    print(f"Recall: {recall_score(y_test, y_pred):.4f} (Higher means fewer good customers rejected)")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f} (Overall ability to rank risk)\n")

evaluate_model("Logistic Regression (Baseline)", lr_pipeline, X_test, y_test)
evaluate_model("Random Forest", rf_pipeline, X_test, y_test)
