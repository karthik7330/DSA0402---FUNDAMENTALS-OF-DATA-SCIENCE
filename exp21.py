import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

age = [
    23, 23, 27, 27, 39, 41, 47, 49, 50,
    52, 54, 54, 56, 57, 58, 58, 60, 61
]

fat = [
    9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2,
    34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7
]

data = pd.DataFrame({
    "Age": age,
    "%Fat": fat
})

# Mean
print("Mean:")
print(data.mean())

# Median
print("\nMedian:")
print(data.median())

# Standard Deviation
print("\nStandard Deviation:")
print(data.std())

# Boxplot for Age and %Fat
plt.figure(figsize=(8, 5))
data.boxplot(column=["Age", "%Fat"])
plt.title("Boxplot of Age and Body Fat")
plt.ylabel("Values")
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(data["Age"], data["%Fat"])
plt.title("Age vs Body Fat Percentage")
plt.xlabel("Age")
plt.ylabel("Body Fat (%)")
plt.grid(True)
plt.show()

# Q-Q Plot for Age
plt.figure(figsize=(7, 5))
stats.probplot(data["Age"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Age")
plt.show()

# Q-Q Plot for Body Fat
plt.figure(figsize=(7, 5))
stats.probplot(data["%Fat"], dist="norm", plot=plt)
plt.title("Q-Q Plot of Body Fat Percentage")
plt.show()