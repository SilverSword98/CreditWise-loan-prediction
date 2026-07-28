# CreditWise: Intelligent Loan Approval System 🏦📊

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)

## 📌 Overview
CreditWise is a machine learning pipeline designed to automatically predict whether a bank should approve or reject a customer's loan application. By analyzing past financial data and creating custom metrics like the loan-to-collateral ratio, the system learns to identify high-risk applicants without needing a human to check every document. 

This helps the bank save time on manual reviews while preventing financial losses from bad loans and ensuring good customers aren't unfairly rejected.

## 🏢 The Business Problem
SecureTrust Bank previously used a manual verification process for loan applications. This led to two major business challenges:
1. **Financial Loss (False Positives):** High-risk customers were sometimes approved, leading to defaults.
2. **Loss of Business (False Negatives):** Good customers were sometimes rejected, sending them to competitors.

The goal of this project is to automate the decision-making process while optimizing for **Precision** (minimizing bad loans) and **Recall** (maximizing good loans), rather than just looking at overall accuracy.

## 🧠 Methodology & Pipeline

### 1. Data Cleaning
* Dropped completely blank records from the dataset.
* Mapped the target variable (`Loan_Approved`) from categorical 'Yes'/'No' to binary `1`/`0`.
* Handled missing values using median imputation for numerical data and mode imputation for categorical data.

### 2. Feature Engineering
To help the model understand the financial health of the applicants better, I engineered three custom business features:
* **Total_Income:** Combined applicant and co-applicant income to get a full picture of household earning.
* **Loan_to_Collateral Ratio:** Measures how well the loan is backed by assets. A higher ratio indicates higher risk.
* **Income_to_Loan Ratio:** Measures the applicant's ability to repay the requested amount based on their total income.

### 3. Modeling
Built a `scikit-learn` pipeline to scale numerical features (`StandardScaler`) and encode categorical features (`OneHotEncoder`). Trained two models for comparison:
* **Logistic Regression** (Baseline Model)
* **Random Forest Classifier** (Tree-based Model)

## 📈 Key Results

The models were evaluated based on their ability to minimize financial loss (Precision) while maximizing approved business (Recall). 

| Metric | Logistic Regression (Baseline) | Random Forest (Final Model) |
| :--- | :--- | :--- |
| **ROC-AUC Score** | 0.76 | **0.85** |
| **Precision** | 0.71 | **0.83** |
| **Recall** | 0.88 | **0.81** |

*(Note: Random Forest was chosen as the final model because the 12% jump in Precision saves the bank significantly more money by avoiding bad loans, even with a slight drop in Recall).*

### Simulated Confusion Matrix (Random Forest)
| | Predicted: Reject (0) | Predicted: Approve (1) |
| :--- | :--- | :--- |
| **Actual: Reject (0)** | True Negatives (Saved Money) | False Positives (Financial Loss) |
| **Actual: Approve (1)** | False Negatives (Lost Business) | True Positives (Good Loan) |

* Achieved an **85% ROC-AUC score**, demonstrating a strong ability to distinguish between high-risk and low-risk applicants.
* The Random Forest model delivered a **12% increase in Precision** over the baseline model, directly reducing the estimated financial loss from bad loans.
