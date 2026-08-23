import pandas as pd
from scipy import stats
import numpy as np

# Read customer review data
data = pd.read_csv("customer_reviews.csv")

# Select ratings
ratings = data["Rating"].dropna()

# Sample size
n = len(ratings)

# Mean rating
mean_rating = ratings.mean()

# Sample standard deviation
standard_deviation = ratings.std()

# Standard error
standard_error = standard_deviation / np.sqrt(n)

# 95% confidence level
confidence_level = 0.95
alpha = 1 - confidence_level

# Critical t-value
t_value = stats.t.ppf(
    1 - alpha / 2,
    n - 1
)

# Margin of error
margin_of_error = t_value * standard_error

# Confidence interval
lower_limit = mean_rating - margin_of_error
upper_limit = mean_rating + margin_of_error

print("Customer Rating Analysis")
print("------------------------")

print("Number of Ratings:", n)
print("Average Rating:", mean_rating)
print("Standard Deviation:", standard_deviation)

print("\n95% Confidence Interval:")
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

print("\nConfidence Interval:",
      lower_limit, "to", upper_limit)

# Customer satisfaction interpretation
if mean_rating >= 4:
    print("\nCustomer Satisfaction Level: HIGH")
elif mean_rating >= 3:
    print("\nCustomer Satisfaction Level: MODERATE")
else:
    print("\nCustomer Satisfaction Level: LOW")