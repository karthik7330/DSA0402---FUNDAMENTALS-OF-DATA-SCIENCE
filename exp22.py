import numpy as np
from scipy import stats

drug = np.array([
    12, 15, 10, 14, 13, 16, 11, 15, 12, 14,
    13, 17, 10, 12, 15, 14, 11, 16, 13, 12,
    14, 15, 13, 11, 16
])

placebo = np.array([
    5, 7, 4, 6, 5, 8, 3, 6, 5, 7,
    4, 6, 5, 7, 3, 5, 6, 4, 7, 5,
    6, 4, 5, 6, 3
])

# Drug group
drug_mean = np.mean(drug)
drug_std = np.std(drug, ddof=1)
drug_n = len(drug)

drug_se = drug_std / np.sqrt(drug_n)

drug_t = stats.t.ppf(0.975, drug_n - 1)

drug_margin = drug_t * drug_se

drug_lower = drug_mean - drug_margin
drug_upper = drug_mean + drug_margin

# Placebo group
placebo_mean = np.mean(placebo)
placebo_std = np.std(placebo, ddof=1)
placebo_n = len(placebo)

placebo_se = placebo_std / np.sqrt(placebo_n)

placebo_t = stats.t.ppf(0.975, placebo_n - 1)

placebo_margin = placebo_t * placebo_se

placebo_lower = placebo_mean - placebo_margin
placebo_upper = placebo_mean + placebo_margin

print("Drug Group")
print("Mean Reduction:", drug_mean)
print("95% Confidence Interval:",
      drug_lower, "to", drug_upper)

print("\nPlacebo Group")
print("Mean Reduction:", placebo_mean)
print("95% Confidence Interval:",
      placebo_lower, "to", placebo_upper)