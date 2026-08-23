import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Clinical trial data
control_group = np.array([
    72, 75, 70, 74, 73,
    76, 71, 75, 72, 74
])

treatment_group = np.array([
    62, 65, 60, 64, 63,
    66, 61, 65, 62, 64
])

# Calculate means
control_mean = np.mean(control_group)
treatment_mean = np.mean(treatment_group)

# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(
    control_group,
    treatment_group
)

# Significance level
alpha = 0.05

print("Medical Treatment Analysis")
print("--------------------------")

print("Control Group Mean:", control_mean)
print("Treatment Group Mean:", treatment_mean)

print("\nT-Statistic:", t_statistic)
print("P-Value:", p_value)

# Hypothesis testing
if p_value < alpha:
    print("\nReject the Null Hypothesis")
    print("The treatment has a statistically significant effect.")
else:
    print("\nFail to Reject the Null Hypothesis")
    print("The treatment does not have a statistically significant effect.")

# Visualization
groups = ["Control", "Treatment"]
means = [control_mean, treatment_mean]

plt.figure(figsize=(7, 5))

plt.bar(groups, means)

plt.title("Control vs Treatment Group")
plt.xlabel("Group")
plt.ylabel("Mean Result")

plt.text(
    0.5,
    max(means),
    "p-value = " + str(round(p_value, 4)),
    ha="center"
)

plt.show()