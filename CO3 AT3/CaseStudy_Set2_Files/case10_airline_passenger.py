import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load Dataset
df = pd.read_csv(case10_airline_passenger.csv)

# Descriptive Statistics
print(Descriptive Statistics)
print(df.describe())

# Histogram
plt.hist(df[Flight_Delay], bins=10)
plt.title(Flight Delay Distribution)
plt.xlabel(Flight Delay (Minutes))
plt.ylabel(Frequency)
plt.show()

# Box Plot
plt.boxplot(df[Passenger_Satisfaction])
plt.title(Passenger Satisfaction Box Plot)
plt.ylabel(Rating)
plt.show()

# Hypothesis Testing
short_delay = df[df[Flight_Delay] = 30][Passenger_Satisfaction]
long_delay = df[df[Flight_Delay]  30][Passenger_Satisfaction]

t_stat, p_value = ttest_ind(short_delay, long_delay)

print(nHypothesis Test)
print(T Statistic =, round(t_stat,3))
print(P Value =, round(p_value,5))

if p_value  0.05
    print(Reject the Null Hypothesis)
    print(Shorter flight delays significantly improve passenger satisfaction.)
else
    print(Fail to Reject the Null Hypothesis)