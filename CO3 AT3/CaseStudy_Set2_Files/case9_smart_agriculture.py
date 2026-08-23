import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load Dataset
df = pd.read_csv("case9_smart_agriculture.csv")

# Descriptive Statistics
print("Descriptive Statistics")
print(df.describe())

# Histogram
plt.hist(df["Crop_Yield"], bins=10)
plt.title("Crop Yield Distribution")
plt.xlabel("Crop Yield (kg)")
plt.ylabel("Frequency")
plt.show()

# Box Plot
plt.boxplot(df["Soil_Moisture"])
plt.title("Soil Moisture Box Plot")
plt.ylabel("Soil Moisture (%)")
plt.show()

# Hypothesis Testing
optimal = df[df["Soil_Moisture"] >= 60]["Crop_Yield"]
non_optimal = df[df["Soil_Moisture"] < 60]["Crop_Yield"]

t_stat, p_value = ttest_ind(optimal, non_optimal)

print("\nHypothesis Test")
print("T Statistic =", round(t_stat,3))
print("P Value =", round(p_value,5))

if p_value < 0.05:
    print("Reject the Null Hypothesis")
    print("Optimal soil moisture significantly improves crop yield.")
else:
    print("Fail to Reject the Null Hypothesis")