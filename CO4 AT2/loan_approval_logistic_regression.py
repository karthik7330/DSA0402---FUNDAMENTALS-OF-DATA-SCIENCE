import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

# 1. Create the dataset
data = {
    "Applicant": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
    "Income": [25000, 30000, 45000, 60000, 70000, 28000, 55000, 35000],
    "CreditScore": [580, 600, 680, 750, 800, 590, 720, 620],
    "ExistingLoan": [200000, 180000, 120000, 80000, 50000, 220000, 100000, 160000],
    "EmploymentYears": [1, 2, 3, 5, 6, 1, 4, 2],
    "Approved": [0, 0, 1, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

print("DATASET")
print(df)

# 2. Identify independent and dependent variables
X = df[["Income", "CreditScore", "ExistingLoan", "EmploymentYears"]]
y = df["Approved"]

# 3. Split the dataset: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTraining rows:", X_train.index.tolist())
print("Testing rows:", X_test.index.tolist())

# 4. Implement Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict approval status for test data
y_pred = model.predict(X_test)
print("\nActual test values:", y_test.values)
print("Predicted test values:", y_pred)

# 6. Predict the new applicant
new_applicant = pd.DataFrame(
    [[50000, 700, 100000, 4]],
    columns=["Income", "CreditScore", "ExistingLoan", "EmploymentYears"]
)

new_prediction = model.predict(new_applicant)
new_probability = model.predict_proba(new_applicant)

print("\nNew applicant prediction:")
print("Status:", "Approved" if new_prediction[0] == 1 else "Rejected")
print("Probability of rejection:", new_probability[0][0])
print("Probability of approval:", new_probability[0][1])

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
