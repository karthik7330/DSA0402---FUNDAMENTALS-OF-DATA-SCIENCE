import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("student_dropout_dataset.csv")

print("=" * 60)
print("        STUDENT DROPOUT RISK ANALYSIS")
print("=" * 60)

# ============================================================
# 2. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET SHAPE")
print("-" * 60)
print("Rows    :", data.shape[0])
print("Columns :", data.shape[1])

print("\n2. FIRST 10 RECORDS")
print("-" * 60)
print(data.head(10))

print("\n3. COLUMN NAMES")
print("-" * 60)
print(data.columns.tolist())

print("\n4. DATA TYPES")
print("-" * 60)
print(data.dtypes)

# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n5. MISSING VALUES")
print("-" * 60)

missing_values = data.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:",
      data.isnull().sum().sum())

# ============================================================
# 4. CHECK DUPLICATES
# ============================================================

print("\n6. DUPLICATE RECORDS")
print("-" * 60)

duplicates = data.duplicated().sum()

print("Number of duplicate rows:", duplicates)

if duplicates > 0:
    data = data.drop_duplicates()
    print("Duplicate rows removed.")
else:
    print("No duplicate rows found.")

# ============================================================
# 5. HANDLE MISSING NUMERICAL VALUES
# ============================================================

print("\n7. HANDLING MISSING VALUES")
print("-" * 60)

# Fill numerical missing values using median
data["Attendance"] = data["Attendance"].fillna(
    data["Attendance"].median()
)

data["Exam_Marks"] = data["Exam_Marks"].fillna(
    data["Exam_Marks"].median()
)

# Fill categorical missing values using mode
data["Digital_Access"] = data["Digital_Access"].fillna(
    data["Digital_Access"].mode()[0]
)

print("Missing numerical values filled using median.")
print("Missing categorical values filled using mode.")

# ============================================================
# 6. VERIFY MISSING VALUES
# ============================================================

print("\n8. MISSING VALUES AFTER CLEANING")
print("-" * 60)

print(data.isnull().sum())

# ============================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================

print("\n9. DESCRIPTIVE STATISTICS")
print("-" * 60)

print(data.describe())

# ============================================================
# 8. TARGET VARIABLE DISTRIBUTION
# ============================================================

print("\n10. DROPOUT RISK DISTRIBUTION")
print("-" * 60)

print(data["Dropout_Risk"].value_counts())

print("\nPercentage Distribution:")

risk_percentage = (
    data["Dropout_Risk"]
    .value_counts(normalize=True)
    * 100
)

print(risk_percentage.round(2))

# ============================================================
# 9. ENCODE CATEGORICAL VARIABLES
# ============================================================

print("\n11. ENCODING CATEGORICAL VARIABLES")
print("-" * 60)

# Create a copy for ML preparation
ml_data = data.copy()

# Label encode binary categorical columns
binary_columns = [
    "Gender",
    "Region",
    "Digital_Access",
    "Scholarship",
    "Previous_Dropout"
]

label_encoders = {}

for column in binary_columns:

    encoder = LabelEncoder()

    ml_data[column] = encoder.fit_transform(
        ml_data[column]
    )

    label_encoders[column] = encoder

# Encode Income Category
income_encoder = LabelEncoder()

ml_data["Income_Category"] = income_encoder.fit_transform(
    ml_data["Income_Category"]
)

label_encoders["Income_Category"] = income_encoder

# Encode target variable
target_encoder = LabelEncoder()

ml_data["Dropout_Risk"] = target_encoder.fit_transform(
    ml_data["Dropout_Risk"]
)

label_encoders["Dropout_Risk"] = target_encoder

print("Categorical variables encoded successfully.")

# ============================================================
# 10. REMOVE STUDENT ID
# ============================================================

ml_data = ml_data.drop(
    columns=["Student_ID"]
)

# ============================================================
# 11. SEPARATE FEATURES AND TARGET
# ============================================================

X = ml_data.drop(
    columns=["Dropout_Risk"]
)

y = ml_data["Dropout_Risk"]

print("\n12. FEATURES AND TARGET")
print("-" * 60)

print("Number of Features:", X.shape[1])
print("Number of Records :", X.shape[0])

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Dropout_Risk")

# ============================================================
# 12. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n13. TRAIN-TEST SPLIT")
print("-" * 60)

print("Training records :", X_train.shape[0])
print("Testing records  :", X_test.shape[0])

# ============================================================
# 13. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\n14. FEATURE SCALING")
print("-" * 60)

print("StandardScaler applied successfully.")

# ============================================================
# 14. FINAL PREPROCESSING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("       PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal Dataset Shape:", data.shape)

print("\nTraining Data Shape:", X_train.shape)

print("Testing Data Shape :", X_test.shape)

print("\nPreprocessing steps completed:")
print("1. Dataset loaded")
print("2. Dataset inspected")
print("3. Missing values identified")
print("4. Missing values handled")
print("5. Duplicate records checked")
print("6. Descriptive statistics generated")
print("7. Categorical variables encoded")
print("8. Target variable separated")
print("9. Train-test split performed")
print("10. Feature scaling performed")