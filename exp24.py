import numpy as np
from scipy import stats

# Read rare element concentration data
data = np.loadtxt("rare_elements.csv", delimiter=",", skiprows=1)

# User input
sample_size = int(input("Enter sample size: "))
confidence_level = float(input("Enter confidence level (%): "))
precision = float(input("Enter desired precision (margin of error): "))

# Check sample size
if sample_size > len(data):
    print("Sample size is larger than the available data.")
else:

    # Select random sample
    sample = np.random.choice(data, sample_size, replace=False)

    # Point estimate
    sample_mean = np.mean(sample)

    # Sample standard deviation
    sample_std = np.std(sample, ddof=1)

    # Standard error
    standard_error = sample_std / np.sqrt(sample_size)

    # Significance level
    alpha = 1 - (confidence_level / 100)

    # Critical t-value
    t_value = stats.t.ppf(1 - alpha / 2, sample_size - 1)

    # Margin of error
    margin_of_error = t_value * standard_error

    # Confidence interval
    lower_limit = sample_mean - margin_of_error
    upper_limit = sample_mean + margin_of_error

    print("\nPoint Estimate (Sample Mean):", sample_mean)
    print("Sample Standard Deviation:", sample_std)
    print("Margin of Error:", margin_of_error)

    print("\nConfidence Interval:")
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)

    if margin_of_error <= precision:
        print("\nDesired precision is achieved.")
    else:
        print("\nDesired precision is not achieved.")
        print("A larger sample size may be required.")