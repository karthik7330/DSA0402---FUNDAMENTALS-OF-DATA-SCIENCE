import numpy as np
from scipy import stats

design_A = np.array([
    4.2, 4.5, 4.8, 4.1, 4.7,
    4.3, 4.6, 4.4, 4.9, 4.5
])

design_B = np.array([
    5.1, 5.3, 5.0, 5.4, 5.2,
    5.5, 5.1, 5.3, 5.4, 5.2
])

mean_A = np.mean(design_A)
mean_B = np.mean(design_B)

t_statistic, p_value = stats.ttest_ind(
    design_A,
    design_B
)

alpha = 0.05

print("Mean Conversion Rate - Design A:", mean_A)
print("Mean Conversion Rate - Design B:", mean_B)

print("\nT-Statistic:", t_statistic)
print("P-Value:", p_value)

if p_value < alpha:
    print("\nThere is a statistically significant difference")
    print("between Design A and Design B.")
else:
    print("\nThere is no statistically significant difference")
    print("between Design A and Design B.")
