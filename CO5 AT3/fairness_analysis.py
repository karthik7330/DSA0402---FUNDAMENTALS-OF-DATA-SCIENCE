import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("student_dropout_dataset.csv")

print("=" * 70)
print("       HIGH-RISK RECALL AND FAIRNESS ANALYSIS")
print("=" * 70)

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

# ============================================================
# 3. ENCODE DATA
# ============================================================

ml_data = data.copy()

categorical_columns = [
    "Gender",
    "Region",
    "Income_Category",
    "Digital_Access",
    "Scholarship",
    "Previous_Dropout"
]

encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    ml_data[column] = encoder.fit_transform(
        ml_data[column]
    )

    encoders[column] = encoder

target_encoder = LabelEncoder()

ml_data["Dropout_Risk"] = target_encoder.fit_transform(
    ml_data["Dropout_Risk"]
)

# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = ml_data.drop(
    columns=[
        "Student_ID",
        "Dropout_Risk"
    ]
)

y = ml_data["Dropout_Risk"]

# Save original indices for fairness analysis
indices = np.arange(len(data))

X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
    X,
    y,
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ============================================================
# 6. CREATE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42
        )
}

# ============================================================
# 7. TRAIN MODELS
# ============================================================

predictions = {}

for name, model in models.items():

    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions[name] = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions[name] = model.predict(
            X_test
        )

# ============================================================
# 8. IDENTIFY HIGH-RISK CLASS
# ============================================================

high_risk_class = target_encoder.transform(
    ["High"]
)[0]

print("\nRisk class encoding:")

for class_name, encoded_value in zip(
    target_encoder.classes_,
    target_encoder.transform(
        target_encoder.classes_
    )
):

    print(
        class_name,
        "->",
        encoded_value
    )

print(
    "\nHigh-risk class:",
    high_risk_class
)

# ============================================================
# 9. HIGH-RISK PERFORMANCE
# ============================================================

high_risk_results = []

print("\n" + "=" * 70)
print("                 HIGH-RISK PERFORMANCE")
print("=" * 70)

for name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Calculate recall specifically for HIGH risk
    high_risk_recall = recall_score(
        y_test,
        y_pred,
        labels=[high_risk_class],
        average="macro",
        zero_division=0
    )

    # Convert high-risk into binary problem
    actual_high = (
        y_test == high_risk_class
    ).astype(int)

    predicted_high = (
        y_pred == high_risk_class
    ).astype(int)

    binary_cm = confusion_matrix(
        actual_high,
        predicted_high,
        labels=[0, 1]
    )

    tn, fp, fn, tp = binary_cm.ravel()

    high_risk_precision = precision_score(
        actual_high,
        predicted_high,
        zero_division=0
    )

    high_risk_f1 = f1_score(
        actual_high,
        predicted_high,
        zero_division=0
    )

    high_risk_results.append({
        "Model": name,
        "High_Risk_Precision": high_risk_precision,
        "High_Risk_Recall": high_risk_recall,
        "High_Risk_F1": high_risk_f1,
        "False_Positives": fp,
        "False_Negatives": fn
    })

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    print(
        "High-Risk Precision:",
        round(high_risk_precision, 4)
    )

    print(
        "High-Risk Recall:",
        round(high_risk_recall, 4)
    )

    print(
        "High-Risk F1:",
        round(high_risk_f1, 4)
    )

    print(
        "False Positives:",
        fp
    )

    print(
        "False Negatives:",
        fn
    )

# ============================================================
# 10. HIGH-RISK COMPARISON TABLE
# ============================================================

high_risk_df = pd.DataFrame(
    high_risk_results
)

print("\n" + "=" * 70)
print("             HIGH-RISK MODEL COMPARISON")
print("=" * 70)

print(
    high_risk_df.round(4)
)

high_risk_df.to_csv(
    "high_risk_model_comparison.csv",
    index=False
)

# ============================================================
# 11. SELECT MODEL FOR FAIRNESS ANALYSIS
# ============================================================

best_model_name = high_risk_df.loc[
    high_risk_df["High_Risk_F1"].idxmax(),
    "Model"
]

best_predictions = predictions[
    best_model_name
]

print("\nSelected model for fairness analysis:")
print(best_model_name)

# ============================================================
# 12. CREATE TEST DATAFRAME
# ============================================================

test_data = data.iloc[
    index_test
].copy()

test_data["Predicted_Risk"] = (
    target_encoder.inverse_transform(
        best_predictions
    )
)

# ============================================================
# 13. FAIRNESS FUNCTION
# ============================================================

def calculate_group_metrics(
    dataframe,
    group_column
):

    results = []

    groups = dataframe[
        group_column
    ].dropna().unique()

    for group in groups:

        group_data = dataframe[
            dataframe[group_column] == group
        ]

        actual = (
            group_data["Dropout_Risk"]
            == "High"
        ).astype(int)

        predicted = (
            group_data["Predicted_Risk"]
            == "High"
        ).astype(int)

        if actual.sum() == 0:

            recall = 0

        else:

            recall = recall_score(
                actual,
                predicted,
                zero_division=0
            )

        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )

        results.append({
            group_column: group,
            "Students": len(group_data),
            "High_Risk_Recall": recall,
            "High_Risk_Precision": precision,
            "High_Risk_F1": f1
        })

    return pd.DataFrame(results)

# ============================================================
# 14. GENDER FAIRNESS
# ============================================================

print("\n" + "=" * 70)
print("                 GENDER FAIRNESS")
print("=" * 70)

gender_results = calculate_group_metrics(
    test_data,
    "Gender"
)

print(
    gender_results.round(4)
)

gender_results.to_csv(
    "fairness_gender.csv",
    index=False
)

# ============================================================
# 15. REGION FAIRNESS
# ============================================================

print("\n" + "=" * 70)
print("                 REGION FAIRNESS")
print("=" * 70)

region_results = calculate_group_metrics(
    test_data,
    "Region"
)

print(
    region_results.round(4)
)

region_results.to_csv(
    "fairness_region.csv",
    index=False
)

# ============================================================
# 16. INCOME FAIRNESS
# ============================================================

print("\n" + "=" * 70)
print("              SOCIOECONOMIC FAIRNESS")
print("=" * 70)

income_results = calculate_group_metrics(
    test_data,
    "Income_Category"
)

print(
    income_results.round(4)
)

income_results.to_csv(
    "fairness_income.csv",
    index=False
)

# ============================================================
# 17. FAIRNESS GAP CALCULATION
# ============================================================

def calculate_fairness_gap(
    results,
    metric="High_Risk_Recall"
):

    maximum = results[
        metric
    ].max()

    minimum = results[
        metric
    ].min()

    return maximum - minimum

gender_gap = calculate_fairness_gap(
    gender_results
)

region_gap = calculate_fairness_gap(
    region_results
)

income_gap = calculate_fairness_gap(
    income_results
)

print("\n" + "=" * 70)
print("                 FAIRNESS GAPS")
print("=" * 70)

print(
    "Gender Recall Gap :",
    round(gender_gap, 4)
)

print(
    "Region Recall Gap :",
    round(region_gap, 4)
)

print(
    "Income Recall Gap :",
    round(income_gap, 4)
)

# ============================================================
# 18. FAIRNESS VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    gender_results["Gender"],
    gender_results["High_Risk_Recall"]
)

plt.title(
    "High-Risk Recall Across Gender Groups"
)

plt.xlabel("Gender")

plt.ylabel("High-Risk Recall")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "10_gender_fairness.png",
    dpi=300
)

plt.show()

# ============================================================
# REGION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    region_results["Region"],
    region_results["High_Risk_Recall"]
)

plt.title(
    "High-Risk Recall Across Regions"
)

plt.xlabel("Region")

plt.ylabel("High-Risk Recall")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "11_region_fairness.png",
    dpi=300
)

plt.show()

# ============================================================
# INCOME GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    income_results["Income_Category"],
    income_results["High_Risk_Recall"]
)

plt.title(
    "High-Risk Recall Across Income Categories"
)

plt.xlabel("Income Category")

plt.ylabel("High-Risk Recall")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "12_income_fairness.png",
    dpi=300
)

plt.show()

# ============================================================
# 19. FINAL FAIRNESS SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("                 FAIRNESS SUMMARY")
print("=" * 70)

print(
    "\nGender recall gap:",
    round(gender_gap, 4)
)

print(
    "Region recall gap:",
    round(region_gap, 4)
)

print(
    "Income recall gap:",
    round(income_gap, 4)
)

print("\nInterpretation:")

print(
    "A smaller recall gap indicates more consistent "
    "high-risk detection across groups."
)

print(
    "Large gaps should be investigated before "
    "deploying the model."
)

# ============================================================
# 20. FINAL RECOMMENDATION
# ============================================================

print("\n" + "=" * 70)
print("             DEPLOYMENT RECOMMENDATION")
print("=" * 70)

print(
    "\nRecommended model:",
    best_model_name
)

print(
    "\nThe model should be used as a "
    "decision-support system rather than an "
    "automatic decision-making system."
)

print(
    "\nHigh-risk students should be reviewed "
    "by teachers or counsellors before intervention."
)

print(
    "\nNo student should be denied educational "
    "support solely because of the model prediction."
)

print("\n" + "=" * 70)
print("           FAIRNESS ANALYSIS COMPLETED")
print("=" * 70)