import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

# Number of students
n = 500

# Student IDs
student_id = [f"S{i:03d}" for i in range(1, n + 1)]

# Basic student information
age = np.random.randint(12, 19, n)

gender = np.random.choice(
    ["Male", "Female"],
    n
)

region = np.random.choice(
    ["Rural", "Urban"],
    n,
    p=[0.55, 0.45]
)

# Academic information
attendance = np.clip(
    np.random.normal(75, 15, n),
    35,
    100
).round(1)

exam_marks = np.clip(
    np.random.normal(65, 15, n),
    20,
    100
).round(1)

# Socioeconomic information
income_category = np.random.choice(
    ["Low", "Medium", "High"],
    n,
    p=[0.40, 0.45, 0.15]
)

# Distance from school in kilometers
distance_to_school = np.round(
    np.random.uniform(0.5, 15, n),
    1
)

# Digital access
digital_access = np.random.choice(
    ["Yes", "No"],
    n,
    p=[0.65, 0.35]
)

# Scholarship support
scholarship = np.random.choice(
    ["Yes", "No"],
    n,
    p=[0.45, 0.55]
)

# Teacher-student ratio
teacher_student_ratio = np.round(
    np.random.uniform(20, 50, n),
    1
)

# Previous dropout history
previous_dropout = np.random.choice(
    ["Yes", "No"],
    n,
    p=[0.12, 0.88]
)

# -------------------------------------------------
# Generate dropout risk
# -------------------------------------------------

risk_score = np.zeros(n)

# Low attendance increases risk
risk_score += np.where(attendance < 60, 3, 0)
risk_score += np.where(
    (attendance >= 60) & (attendance < 75),
    1.5,
    0
)

# Low exam marks increase risk
risk_score += np.where(exam_marks < 40, 3, 0)
risk_score += np.where(
    (exam_marks >= 40) & (exam_marks < 60),
    1.5,
    0
)

# Low income increases risk
risk_score += np.where(
    income_category == "Low",
    2,
    0
)

# Long distance increases risk
risk_score += np.where(
    distance_to_school > 10,
    2,
    0
)

# No digital access increases risk
risk_score += np.where(
    digital_access == "No",
    1,
    0
)

# No scholarship support increases risk
risk_score += np.where(
    scholarship == "No",
    1,
    0
)

# Previous dropout increases risk
risk_score += np.where(
    previous_dropout == "Yes",
    3,
    0
)

# Poor teacher-student ratio
risk_score += np.where(
    teacher_student_ratio > 40,
    1,
    0
)

# Add small random variation
risk_score += np.random.normal(0, 1, n)

# Convert score into risk category
dropout_risk = np.where(
    risk_score >= 6,
    "High",
    np.where(
        risk_score >= 3,
        "Medium",
        "Low"
    )
)

# -------------------------------------------------
# Create DataFrame
# -------------------------------------------------

data = pd.DataFrame({
    "Student_ID": student_id,
    "Age": age,
    "Gender": gender,
    "Region": region,
    "Attendance": attendance,
    "Exam_Marks": exam_marks,
    "Income_Category": income_category,
    "Distance_to_School": distance_to_school,
    "Digital_Access": digital_access,
    "Scholarship": scholarship,
    "Teacher_Student_Ratio": teacher_student_ratio,
    "Previous_Dropout": previous_dropout,
    "Dropout_Risk": dropout_risk
})

# -------------------------------------------------
# Add a few missing values
# -------------------------------------------------

missing_indices = np.random.choice(
    n,
    size=15,
    replace=False
)

data.loc[missing_indices[:5], "Attendance"] = np.nan
data.loc[missing_indices[5:10], "Exam_Marks"] = np.nan
data.loc[missing_indices[10:], "Digital_Access"] = np.nan

# -------------------------------------------------
# Save dataset
# -------------------------------------------------

data.to_csv(
    "student_dropout_dataset.csv",
    index=False
)

print("==============================================")
print(" STUDENT DROPOUT DATASET CREATED")
print("==============================================")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 10 Records:")
print(data.head(10))

print("\nDropout Risk Distribution:")
print(data["Dropout_Risk"].value_counts())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDataset saved as:")
print("student_dropout_dataset.csv")