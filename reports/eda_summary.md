# Exploratory Data Analysis (EDA) Summary

## Dataset Overview

- Number of rows: 7043
- Number of columns: 21
- Target variable: Churn

---

## Findings

### 1. Churn Distribution

- Around 26.5% of customers have churned.
- The dataset is slightly imbalanced.

---

### 2. Gender

- Male and female customers are nearly equally distributed.
- Churn rates between genders are similar.
- Gender does not appear to be a strong predictor.

---

### 3. Contract Type

- Customers with Month-to-month contracts have the highest churn rate.
- Two-year contracts have the lowest churn rate.
- Contract type is likely an important feature.

---

### 4. Tenure

- Customers with short tenure are more likely to churn.
- Long-term customers are generally more loyal.

---

### 5. Monthly Charges

- Customers paying higher monthly charges tend to churn more frequently.

---

## Potential Important Features

- Contract
- Tenure
- MonthlyCharges
- InternetService
- PaymentMethod

---

## Next Steps

- Handle missing values.
- Encode categorical features.
- Scale numerical features if necessary.
- Train baseline machine learning models.
