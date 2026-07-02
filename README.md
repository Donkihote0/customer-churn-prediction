# Dự đoán Khách hàng Rời bỏ Dịch vụ (Customer Churn Prediction)

## 1. Giới thiệu

Đây là dự án Machine Learning nhằm dự đoán khả năng khách hàng rời bỏ dịch vụ (Customer Churn) dựa trên thông tin cá nhân và lịch sử sử dụng dịch vụ.

Dự án được xây dựng theo quy trình chuẩn của một bài toán Machine Learning, từ phân tích dữ liệu, tiền xử lý, xây dựng mô hình đến dự đoán cho dữ liệu mới.

---

# 2. Mục tiêu

- Phân tích dữ liệu khách hàng.
- Tiền xử lý và làm sạch dữ liệu.
- Xây dựng các mô hình phân loại.
- So sánh hiệu quả giữa các mô hình.
- Tối ưu mô hình bằng Hyperparameter Tuning.
- Dự đoán khả năng rời bỏ của khách hàng mới.

---

# 3. Bộ dữ liệu

**Tên bộ dữ liệu**

IBM Telco Customer Churn Dataset

**Biến mục tiêu**

```
Churn Label
```

Ý nghĩa:

- Yes: Khách hàng rời bỏ dịch vụ.
- No: Khách hàng tiếp tục sử dụng dịch vụ.

---

# 4. Công nghệ sử dụng

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

---

# 5. Quy trình thực hiện

```
Thu thập dữ liệu
        │
        ▼
Phân tích dữ liệu (EDA)
        │
        ▼
Làm sạch dữ liệu
        │
        ▼
Feature Engineering
        │
        ▼
Huấn luyện mô hình
        │
        ▼
Đánh giá mô hình
        │
        ▼
So sánh mô hình
        │
        ▼
Hyperparameter Tuning
        │
        ▼
Dự đoán khách hàng mới
```

---

# 6. Các mô hình sử dụng

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

# 7. Các chỉ số đánh giá

Trong dự án sử dụng các chỉ số:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Trong đó **ROC-AUC** được sử dụng để lựa chọn mô hình tốt nhất.

---

# 8. Cấu trúc dự án

```
customer-churn-prediction/

│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample_customer.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   ├── best_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_DataCleaning.ipynb
│   ├── 03_FeatureEngineering.ipynb
│   ├── 04_ModelTraining.ipynb
│   ├── 05_ModelEvaluation.ipynb
│   ├── 06_ModelComparison.ipynb
│   ├── 07_HyperparameterTuning.ipynb
│   └── 08_Prediction.ipynb
│
├── reports/
│
├── src/
│
├── requirements.txt
│
└── README.md
```

---

# 9. Hướng dẫn cài đặt

## Bước 1

Clone hoặc tải project về máy.

## Bước 2

Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Bước 3

Mở Jupyter Notebook.

## Bước 4

Thực hiện lần lượt các notebook:

```
01_EDA.ipynb

↓

02_DataCleaning.ipynb

↓

03_FeatureEngineering.ipynb

↓

04_ModelTraining.ipynb

↓

05_ModelEvaluation.ipynb

↓

06_ModelComparison.ipynb

↓

07_HyperparameterTuning.ipynb

↓

08_Prediction.ipynb
```

---

# 10. Kết quả

Trong dự án, nhiều mô hình Machine Learning đã được xây dựng và so sánh.

Mô hình cuối cùng được lựa chọn dựa trên chỉ số **ROC-AUC** sau quá trình Hyperparameter Tuning.

Các kết quả đánh giá được lưu trong thư mục:

```
reports/
```

Bao gồm:

- model_comparison.csv
- tuning_results.csv
- best_params.json
- predictions.csv
  ...

---

# 11. Kiến thức áp dụng

Thông qua dự án, tôi đã thực hành:

- Phân tích dữ liệu (EDA)
- Làm sạch dữ liệu
- Feature Engineering
- Machine Learning Classification
- Đánh giá mô hình
- So sánh nhiều mô hình
- Hyperparameter Tuning
- Xây dựng Prediction Pipeline
- Tổ chức dự án Machine Learning theo cấu trúc module

---
