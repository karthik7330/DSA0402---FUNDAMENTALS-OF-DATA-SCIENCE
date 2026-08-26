import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("student_dropout_dataset.csv")

# ============================================================
# 2. HANDLE MISSING VALUES
# ============================================================

data["Attendance"] = data["Attendance"].fillna(
    data["Attendance"].median()
)

data["Exam_Marks"] = data["Exam_Marks"].fillna(
    data["Exam_Marks"].median()
)

data["Digital_Access"] = data["Digital_Access"].fillna(
    data["Digital_Access"].mode()[0]
)

print("=" * 60)
print("       EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ============================================================
# 3. BASIC STATISTICS
# ============================================================

print("\n1. NUMERICAL SUMMARY")
print("-" * 60)

print(
    data[
        [
            "Age",
            "Attendance",
            "Exam_Marks",
            "Distance_to_School",
            "Teacher_Student_Ratio"
        ]
    ].describe()
)

# ============================================================
# 4. DROPOUT RISK DISTRIBUTION
# ============================================================

print("\n2. DROPOUT RISK DISTRIBUTION")
print("-" * 60)

risk_counts = data["Dropout_Risk"].value_counts()

print(risk_counts)

print("\nPercentage:")

risk_percentage = (
    data["Dropout_Risk"]
    .value_counts(normalize=True)
    * 100
)

print(risk_percentage.round(2))

# ============================================================
# GRAPH 1 - DROPOUT RISK DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=data,
    x="Dropout_Risk"
)

plt.title("Student Dropout Risk Distribution")
plt.xlabel("Dropout Risk")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig("01_dropout_risk_distribution.png", dpi=300)
plt.show()

# ============================================================
# 5. ATTENDANCE ANALYSIS
# ============================================================

print("\n3. ATTENDANCE ANALYSIS")
print("-" * 60)

attendance_by_risk = data.groupby(
    "Dropout_Risk"
)["Attendance"].mean()

print(
    attendance_by_risk.round(2)
)

# ============================================================
# GRAPH 2 - ATTENDANCE VS DROPOUT RISK
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Dropout_Risk",
    y="Attendance"
)

plt.title("Attendance Distribution by Dropout Risk")
plt.xlabel("Dropout Risk")
plt.ylabel("Attendance (%)")

plt.tight_layout()
plt.savefig("02_attendance_vs_risk.png", dpi=300)
plt.show()

# ============================================================
# 6. EXAM MARKS ANALYSIS
# ============================================================

print("\n4. EXAM MARKS ANALYSIS")
print("-" * 60)

marks_by_risk = data.groupby(
    "Dropout_Risk"
)["Exam_Marks"].mean()

print(
    marks_by_risk.round(2)
)

# ============================================================
# GRAPH 3 - EXAM MARKS VS DROPOUT RISK
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=data,
    x="Dropout_Risk",
    y="Exam_Marks"
)

plt.title("Exam Marks Distribution by Dropout Risk")
plt.xlabel("Dropout Risk")
plt.ylabel("Exam Marks")

plt.tight_layout()
plt.savefig("03_exam_marks_vs_risk.png", dpi=300)
plt.show()

# ============================================================
# 7. INCOME CATEGORY ANALYSIS
# ============================================================

print("\n5. INCOME CATEGORY VS DROPOUT RISK")
print("-" * 60)

income_risk = pd.crosstab(
    data["Income_Category"],
    data["Dropout_Risk"],
    normalize="index"
) * 100

print(
    income_risk.round(2)
)

# ============================================================
# GRAPH 4 - INCOME VS DROPOUT RISK
# ============================================================

plt.figure(figsize=(9, 5))

income_plot = pd.crosstab(
    data["Income_Category"],
    data["Dropout_Risk"]
)

income_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Income Category vs Dropout Risk")
plt.xlabel("Income Category")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.legend(title="Dropout Risk")

plt.tight_layout()
plt.savefig("04_income_vs_risk.png", dpi=300)
plt.show()

# ============================================================
# 8. DIGITAL ACCESS ANALYSIS
# ============================================================

print("\n6. DIGITAL ACCESS VS DROPOUT RISK")
print("-" * 60)

digital_risk = pd.crosstab(
    data["Digital_Access"],
    data["Dropout_Risk"],
    normalize="index"
) * 100

print(
    digital_risk.round(2)
)

# ============================================================
# GRAPH 5 - DIGITAL ACCESS VS RISK
# ============================================================

plt.figure(figsize=(9, 5))

digital_plot = pd.crosstab(
    data["Digital_Access"],
    data["Dropout_Risk"]
)

digital_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Digital Access vs Dropout Risk")
plt.xlabel("Digital Access")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.legend(title="Dropout Risk")

plt.tight_layout()
plt.savefig("05_digital_access_vs_risk.png", dpi=300)
plt.show()

# ============================================================
# 9. SCHOLARSHIP ANALYSIS
# ============================================================

print("\n7. SCHOLARSHIP VS DROPOUT RISK")
print("-" * 60)

scholarship_risk = pd.crosstab(
    data["Scholarship"],
    data["Dropout_Risk"],
    normalize="index"
) * 100

print(
    scholarship_risk.round(2)
)

# ============================================================
# GRAPH 6 - SCHOLARSHIP VS RISK
# ============================================================

plt.figure(figsize=(9, 5))

scholarship_plot = pd.crosstab(
    data["Scholarship"],
    data["Dropout_Risk"]
)

scholarship_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Scholarship Support vs Dropout Risk")
plt.xlabel("Scholarship Support")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.legend(title="Dropout Risk")

plt.tight_layout()
plt.savefig("06_scholarship_vs_risk.png", dpi=300)
plt.show()

# ============================================================
# 10. PREVIOUS DROPOUT ANALYSIS
# ============================================================

print("\n8. PREVIOUS DROPOUT VS CURRENT RISK")
print("-" * 60)

previous_risk = pd.crosstab(
    data["Previous_Dropout"],
    data["Dropout_Risk"],
    normalize="index"
) * 100

print(
    previous_risk.round(2)
)

# ============================================================
# GRAPH 7 - PREVIOUS DROPOUT VS RISK
# ============================================================

plt.figure(figsize=(9, 5))

previous_plot = pd.crosstab(
    data["Previous_Dropout"],
    data["Dropout_Risk"]
)

previous_plot.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Previous Dropout History vs Current Risk")
plt.xlabel("Previous Dropout")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.legend(title="Dropout Risk")

plt.tight_layout()
plt.savefig("07_previous_dropout_vs_risk.png", dpi=300)
plt.show()

# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

print("\n9. CORRELATION ANALYSIS")
print("-" * 60)

correlation_data = data.copy()

# Convert categorical columns to numerical values
correlation_data["Gender"] = (
    correlation_data["Gender"]
    .map({"Male": 0, "Female": 1})
)

correlation_data["Region"] = (
    correlation_data["Region"]
    .map({"Rural": 0, "Urban": 1})
)

correlation_data["Income_Category"] = (
    correlation_data["Income_Category"]
    .map({
        "Low": 0,
        "Medium": 1,
        "High": 2
    })
)

correlation_data["Digital_Access"] = (
    correlation_data["Digital_Access"]
    .map({"No": 0, "Yes": 1})
)

correlation_data["Scholarship"] = (
    correlation_data["Scholarship"]
    .map({"No": 0, "Yes": 1})
)

correlation_data["Previous_Dropout"] = (
    correlation_data["Previous_Dropout"]
    .map({"No": 0, "Yes": 1})
)

correlation_data["Dropout_Risk"] = (
    correlation_data["Dropout_Risk"]
    .map({
        "Low": 0,
        "Medium": 1,
        "High": 2
    })
)

correlation_matrix = correlation_data.corr(
    numeric_only=True
)

print(
    correlation_matrix.round(2)
)

# ============================================================
# GRAPH 8 - CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Student Factors")

plt.tight_layout()
plt.savefig("08_correlation_heatmap.png", dpi=300)
plt.show()

# ============================================================
# 12. IMPORTANT FACTORS
# ============================================================

print("\n10. FACTORS MOST RELATED TO DROPOUT RISK")
print("-" * 60)

risk_correlation = (
    correlation_matrix["Dropout_Risk"]
    .drop("Dropout_Risk")
    .sort_values(
        key=abs,
        ascending=False
    )
)

print(
    risk_correlation.round(3)
)

# ============================================================
# 13. DISTANCE ANALYSIS
# ============================================================

print("\n11. DISTANCE TO SCHOOL BY DROPOUT RISK")
print("-" * 60)

distance_by_risk = data.groupby(
    "Dropout_Risk"
)["Distance_to_School"].mean()

print(
    distance_by_risk.round(2)
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("             EDA COMPLETED")
print("=" * 60)

print("\nGenerated graphs:")

print("1. 01_dropout_risk_distribution.png")
print("2. 02_attendance_vs_risk.png")
print("3. 03_exam_marks_vs_risk.png")
print("4. 04_income_vs_risk.png")
print("5. 05_digital_access_vs_risk.png")
print("6. 06_scholarship_vs_risk.png")
print("7. 07_previous_dropout_vs_risk.png")
print("8. 08_correlation_heatmap.png")