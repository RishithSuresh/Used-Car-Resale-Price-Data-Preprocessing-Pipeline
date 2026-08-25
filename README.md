# Used Car Resale Price – Data Preprocessing Pipeline

## 📌 Project Overview

This project performs a complete **data preprocessing workflow** on a Used Car Resale Dataset. The objective is to prepare raw automobile data for machine learning by identifying and handling data quality issues, analyzing outliers, encoding categorical variables, scaling numerical features, and separating the dataset into training and testing sets.

A major focus of this project is **preventing data leakage** by ensuring that preprocessing transformations are fitted only on the training dataset and subsequently applied to the testing dataset.

---

## 🎯 Objectives

The main objectives of this project are:

* Load and inspect the Used Car Resale Dataset.
* Understand the structure, data types, and statistical characteristics of the dataset.
* Identify missing values and duplicate records.
* Detect potential outliers using the **Interquartile Range (IQR)** method.
* Analyze outliers without unnecessarily removing legitimate vehicle observations.
* Separate features from the target variable.
* Remove the `Car_ID` identifier because it does not provide meaningful predictive information.
* Encode categorical variables using appropriate techniques.
* Apply **One-Hot Encoding** to nominal categorical variables.
* Apply **Ordinal Encoding** to the `Condition` variable.
* Standardize numerical features using `StandardScaler`.
* Split the dataset into training and testing sets.
* Prevent data leakage by fitting preprocessing transformations only on training data.
* Verify the processed dataset.
* Export the final preprocessed datasets as CSV files.

---

## 📊 Dataset

The dataset contains information about used cars and their resale prices.

### Features

| Column               | Description                         | Preprocessing                  |
| -------------------- | ----------------------------------- | ------------------------------ |
| `Car_ID`             | Unique identifier for each car      | Removed                        |
| `Brand`              | Car manufacturer/brand              | One-Hot Encoding               |
| `Year`               | Manufacturing year                  | Standardization                |
| `Mileage_Km`         | Distance driven in kilometers       | IQR analysis + Standardization |
| `Engine_CC`          | Engine displacement in CC           | IQR analysis + Standardization |
| `Power_BHP`          | Engine power in BHP                 | IQR analysis + Standardization |
| `Fuel_Type`          | Type of fuel used                   | One-Hot Encoding               |
| `Transmission`       | Transmission type                   | One-Hot Encoding               |
| `City`               | City where the car is being sold    | One-Hot Encoding               |
| `Seller_Type`        | Type of seller                      | One-Hot Encoding               |
| `Condition`          | Condition of the vehicle            | Ordinal Encoding               |
| `Previous_Owners`    | Number of previous owners           | IQR analysis + Standardization |
| `Accidents_Reported` | Number of reported accidents        | IQR analysis + Standardization |
| `Service_Score`      | Vehicle service/maintenance score   | Standardization                |
| `Resale_Price_Lakh`  | Resale price of the vehicle in lakh | Target variable                |

---

## 🔄 Preprocessing Workflow

The preprocessing pipeline follows these steps:

```text
Raw Dataset
     │
     ▼
Initial Data Inspection
     │
     ▼
Missing Value Analysis
     │
     ▼
Duplicate Detection
     │
     ▼
Outlier Analysis using IQR
     │
     ▼
Remove Car_ID
     │
     ▼
Separate Features and Target
     │
     ▼
Train-Test Split (80:20)
     │
     ▼
┌──────────────┬──────────────┬───────────────┐
│ Numerical    │ Nominal      │ Ordinal       │
│ Features     │ Features     │ Features      │
└──────┬───────┴──────┬───────┴───────┬───────┘
       │              │               │
       ▼              ▼               ▼
  Imputation      Imputation      Imputation
       │              │               │
       ▼              ▼               ▼
 StandardScaler   One-Hot         Ordinal
                  Encoding        Encoding
       │              │               │
       └──────────────┼───────────────┘
                      ▼
             Fit on Training Data
                      │
                      ▼
              Transform Training
                      │
                      ▼
               Transform Testing
                      │
                      ▼
                 Verification
                      │
                      ▼
                 CSV Export
```

---

## 🔍 Outlier Detection

Potential numerical outliers are identified using the **Interquartile Range (IQR)** method.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The lower and upper boundaries are:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Observations outside these boundaries are considered potential outliers.

However, extreme values are not automatically removed because they may represent legitimate vehicles, such as luxury cars, high-performance vehicles, or vehicles with unusually high mileage.

The target variable `Resale_Price_Lakh` is not used for feature outlier removal.

---

## 🔤 Categorical Encoding

### Nominal Encoding

The following variables do not have an inherent order:

* `Brand`
* `Fuel_Type`
* `Transmission`
* `City`
* `Seller_Type`

Therefore, **One-Hot Encoding** is used.

Example:

```text
Fuel_Type

Petrol → [1, 0, 0]
Diesel → [0, 1, 0]
CNG    → [0, 0, 1]
```

### Ordinal Encoding

`Condition` has a natural order:

```text
Poor < Fair < Average < Good < Very Good < Excellent
```

Therefore, **Ordinal Encoding** is used:

```text
Poor       → 0
Fair       → 1
Average    → 2
Good       → 3
Very Good  → 4
Excellent  → 5
```

---

## ⚖️ Feature Scaling

Numerical features are standardized using `StandardScaler`.

Standardization is calculated as:

```text
z = (x - μ) / σ
```

where:

* `x` = original value
* `μ` = mean calculated from training data
* `σ` = standard deviation calculated from training data

The scaler is fitted **only on the training dataset**.

---

## 🛡️ Data Leakage Prevention

To prevent data leakage, the dataset is first divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The preprocessing pipeline is then fitted only on the training data:

```python
X_train_processed = preprocessor.fit_transform(X_train)
```

The same learned transformations are applied to the test data:

```python
X_test_processed = preprocessor.transform(X_test)
```

The test data is therefore never used to calculate:

* Imputation values
* Scaling parameters
* Encoding mappings
* Category information

This prevents information from the test set from influencing the preprocessing process.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation and analysis
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **Scikit-learn** – Preprocessing, encoding, scaling, and train-test splitting
* **Jupyter Notebook / Google Colab** – Development environment

---

## 📁 Project Structure

```text
used-car-resale-data-preprocessing/
│
├── Used_Car_Resale_Preprocessing.ipynb
│
├── used_car_resale_dataset.csv
│
├── used_car_train_preprocessed.csv
│
├── used_car_test_preprocessed.csv
│
├── used_car_preprocessed.csv
│
└── README.md
```

> The original dataset may be excluded from the repository if it is provided by an LMS or other restricted source.

---

## 📈 Dataset Verification

After preprocessing, the following checks are performed:

* Dataset dimensions
* Missing values
* Duplicate records
* Numerical feature statistics
* Encoded categorical variables
* Standardized numerical features
* Training/testing dimensions
* Processed feature names

The processed datasets are also exported as CSV files for further machine-learning tasks.

---

## 📂 Output Files

### `used_car_train_preprocessed.csv`

Contains the processed training features and corresponding `Resale_Price_Lakh` target values.

### `used_car_test_preprocessed.csv`

Contains the processed testing features and corresponding target values.

### `used_car_preprocessed.csv`

Contains the combined processed dataset, if required for submission.

---

## 🚀 Future Scope

The preprocessed dataset can be used for developing machine-learning models for used-car resale price prediction.

Possible future models include:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting
* XGBoost
* Support Vector Regression

Model performance can be evaluated using metrics such as:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

## 📄 License

This project was created for **educational and academic purposes** as part of a data preprocessing exercise.
