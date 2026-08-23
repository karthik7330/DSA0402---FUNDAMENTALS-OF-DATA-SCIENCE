import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read Dataset
df = pd.read_csv("students.csv")

print("Student Dataset")
print(df)

# -------------------------
# Data Preprocessing
# -------------------------

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Remove Duplicates
df = df.drop_duplicates()

# Add Result Column
df["Result"] = np.where(df["Marks"] >= 50, "Pass", "Fail")

# Statistics
average_marks = np.mean(df["Marks"])
highest_marks = np.max(df["Marks"])
lowest_marks = np.min(df["Marks"])
pass_percentage = (df["Result"]=="Pass").mean()*100

print("\nAverage Marks :", average_marks)
print("Highest Marks :", highest_marks)
print("Lowest Marks :", lowest_marks)
print("Pass Percentage :", pass_percentage,"%")

plt.figure(figsize=(8,5))

plt.bar(df["Name"],df["Marks"])

plt.title("Student Marks")

plt.xlabel("Students")

plt.ylabel("Marks")

plt.xticks(rotation=45)

plt.show()


result = df["Result"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(result,
        labels=result.index,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Pass vs Fail")

plt.show()


plt.figure(figsize=(7,5))

plt.hist(df["Marks"],bins=5)

plt.title("Distribution of Marks")

plt.xlabel("Marks")

plt.ylabel("Frequency")

plt.show()



