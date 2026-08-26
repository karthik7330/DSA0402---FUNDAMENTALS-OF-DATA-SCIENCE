import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("student_dropout_dataset.csv")

print("=" * 65)
print("        STUDENT DROPOUT RISK - MACHINE LEARNING")
print("=" * 65)

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

print("\nMissing values handled successfully.")

# ============================================================
# 3. ENCODE CATEGORICAL VARIABLES
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

# Encode target
target_encoder = LabelEncoder()

ml_data["Dropout_Risk"] = target_encoder.fit_transform(
    ml_data["Dropout_Risk"]
)

# ============================================================
# 4. REMOVE STUDENT ID
# ============================================================

ml_data = ml_data.drop(
    columns=["Student_ID"]
)

# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

X = ml_data.drop(
    columns=["Dropout_Risk"]
)

y = ml_data["Dropout_Risk"]

print("\nFeatures used for prediction:")
print(X.columns.tolist())

print("\nTarget classes:")
print(
    target_encoder.classes_
)

# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records :", len(X_train))
print("Testing records  :", len(X_test))

# ============================================================
# 7. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# ============================================================
# 8. CREATE MODELS
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
# 9. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

predictions = {}

print("\n" + "=" * 65)
print("                 MODEL PERFORMANCE")
print("=" * 65)

for name, model in models.items():

    # Logistic Regression uses scaled data
    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

    predictions[name] = y_pred

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)

    print(
        "Accuracy :", round(accuracy, 4)
    )

    print(
        "Precision:", round(precision, 4)
    )

    print(
        "Recall   :", round(recall, 4)
    )

    print(
        "F1-Score :", round(f1, 4)
    )

# ============================================================
# 10. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("                 MODEL COMPARISON")
print("=" * 65)

print(
    results_df.round(4)
)

# ============================================================
# 11. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print(
    "\nModel comparison saved as:"
)

print(
    "model_comparison_results.csv"
)

# ============================================================
# 12. MODEL COMPARISON GRAPH
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

x = np.arange(
    len(results_df["Model"])
)

width = 0.18

plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):

    plt.bar(
        x + i * width,
        results_df[metric],
        width,
        label=metric
    )

plt.xticks(
    x + width * 1.5,
    results_df["Model"]
)

plt.ylabel("Score")

plt.xlabel("Machine Learning Model")

plt.title(
    "Comparison of Machine Learning Models"
)

plt.ylim(0, 1.05)

plt.legend()

plt.tight_layout()

plt.savefig(
    "09_model_comparison.png",
    dpi=300
)

plt.show()

# ============================================================
# 13. CONFUSION MATRICES
# ============================================================

class_names = target_encoder.classes_

for name, y_pred in predictions.items():

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        name + " - Confusion Matrix"
    )

    plt.xlabel("Predicted Risk")

    plt.ylabel("Actual Risk")

    plt.tight_layout()

    safe_name = name.lower().replace(
        " ",
        "_"
    )

    plt.savefig(
        f"confusion_matrix_{safe_name}.png",
        dpi=300
    )

    plt.show()

# ============================================================
# 14. DETAILED CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("             CLASSIFICATION REPORTS")
print("=" * 65)

for name, y_pred in predictions.items():

    print("\n")
    print("-" * 65)
    print(name)
    print("-" * 65)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

# ============================================================
# 15. BEST MODEL
# ============================================================

best_model_row = results_df.loc[
    results_df["F1-Score"].idxmax()
]

print("\n" + "=" * 65)
print("                  BEST MODEL")
print("=" * 65)

print(
    "Selected Model:",
    best_model_row["Model"]
)

print(
    "Accuracy:",
    round(best_model_row["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_model_row["Precision"], 4)
)

print(
    "Recall:",
    round(best_model_row["Recall"], 4)
)

print(
    "F1-Score:",
    round(best_model_row["F1-Score"], 4)
)

print("\nReason:")
print(
    "The model with the highest F1-score is selected "
    "as the initial best-performing model because F1-score "
    "balances precision and recall."
)

print("\n" + "=" * 65)
print("             MODEL TRAINING COMPLETED")
print("=" * 65)