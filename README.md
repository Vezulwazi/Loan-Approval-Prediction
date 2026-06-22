# Loan Approval Prediction

A machine learning classification project that predicts whether a home loan application will be approved based on an applicant's demographic and financial information.

## Project Overview

Financial institutions evaluate multiple factors before approving a loan application. This project uses machine learning techniques to predict loan approval outcomes based on applicant characteristics such as income, education, credit history, marital status, and property area.

The objective is to build and compare several classification models and identify the factors that most strongly influence loan approval decisions.

---

## Problem Statement

Given an applicant's:

* Gender
* Marital Status
* Number of Dependents
* Education Level
* Employment Status
* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area

predict whether a loan application will be approved (`Loan_Status = Y`) or denied (`Loan_Status = N`).

---

## Dataset

**Source:** Loan Prediction Dataset (Analytics Vidhya / Kaggle)

### Dataset Summary

* Total Applications: **614**
* Approved Loans: **422**
* Denied Loans: **192**
* Features: **11 input features + 1 target variable**

| Feature           | Type          | Description                                      |
| ----------------- | ------------- | ------------------------------------------------ |
| Loan_ID           | String        | Unique application identifier                    |
| Gender            | Categorical   | Male / Female                                    |
| Married           | Categorical   | Yes / No                                         |
| Dependents        | Categorical   | 0, 1, 2, 3+                                      |
| Education         | Categorical   | Graduate / Not Graduate                          |
| Self_Employed     | Categorical   | Yes / No                                         |
| ApplicantIncome   | Numeric       | Applicant monthly income                         |
| CoapplicantIncome | Numeric       | Co-applicant monthly income                      |
| LoanAmount        | Numeric       | Requested loan amount                            |
| Loan_Amount_Term  | Numeric       | Loan repayment term (months)                     |
| Credit_History    | Binary        | 1 = Good credit history, 0 = Poor credit history |
| Property_Area     | Categorical   | Urban, Semiurban, Rural                          |
| Loan_Status       | Binary Target | Y = Approved, N = Denied                         |

### Data Quality

Several features contained missing values. Data preprocessing included:

* Mode imputation for categorical variables
* Median imputation for numerical variables
* Feature encoding
* Data cleaning and transformation

---

## Project Structure

```text
Loan-Approval-Prediction/
│
├── 01_eda.ipynb
├── 02_data_cleaning.ipynb
├── 03_model_building.ipynb
├── utils.py
├── requirements.txt
├── README.md
│
└── data/
    ├── loan.csv
    └── loan_cleaned.csv
```

### Workflow

1. Exploratory Data Analysis (`01_eda.ipynb`)
2. Data Cleaning & Feature Engineering (`02_data_cleaning.ipynb`)
3. Model Training & Evaluation (`03_model_building.ipynb`)

---

## Models Evaluated

The following machine learning models were trained and evaluated:

* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Naive Bayes

Hyperparameter tuning was performed using **GridSearchCV**.

---

## Model Performance

| Model                 | Accuracy | Precision | Recall | F1 Score |
| --------------------- | -------- | --------- | ------ | -------- |
| KNN (K=3)             | 0.8618   | 0.8542    | 0.9647 | 0.9061   |
| SVM                   | 0.8537   | 0.8317    | 0.9882 | 0.9032   |
| Logistic Regression   | 0.8537   | 0.8384    | 0.9765 | 0.9022   |
| Random Forest         | 0.8537   | 0.8602    | 0.9412 | 0.8989   |
| Naive Bayes           | 0.8455   | 0.8367    | 0.9647 | 0.8962   |
| Gradient Boosting     | 0.8374   | 0.8421    | 0.9412 | 0.8889   |
| Decision Tree         | 0.7967   | 0.8191    | 0.9059 | 0.8603   |
| Random Forest (Tuned) | 0.8537   | 0.8454    | 0.9647 | 0.9011   |

### Best Hyperparameters

```python
n_estimators = 300
max_depth = 5
min_samples_split = 2
min_samples_leaf = 1
```

Cross-Validation F1 Score:

```python
0.8686
```

---

## Key Findings

### Credit History Is the Most Important Feature

Applicants with a positive credit history were significantly more likely to receive loan approval. This feature contributed most strongly to model predictions.

### Semiurban Applicants Have the Highest Approval Rate

Among the three property categories, Semiurban applicants achieved the highest loan approval rate.

### Education and Marital Status Matter

Married graduates showed the strongest likelihood of receiving loan approval.

### Income Has Limited Predictive Power

Although income influences loan approval, its impact was smaller than expected compared to credit history.

### Missing Value Imputation Works Well

Simple median and mode imputation successfully handled missing data while maintaining strong model performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the following order:

```text
01_eda.ipynb
→ 02_data_cleaning.ipynb
→ 03_model_building.ipynb
```

---

## Future Improvements

* Deploy the model using Flask or Streamlit
* Perform advanced feature engineering
* Handle class imbalance using resampling techniques
* Explore ensemble learning methods
* Build a web application for real-time predictions

---

## Author

**Denzel Muwanazi**

BSc Physics and Computer Science
North-West University

Machine Learning | Data Science | Astrophysics
