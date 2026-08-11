import pandas as pd
import numpy as np

temperature_data = pd.read_csv("temperature.csv")

mean_temperature = temperature_data.groupby("City")["Temperature"].mean()
standard_deviation = temperature_data.groupby("City")["Temperature"].std()

maximum_temperature = temperature_data.groupby("City")["Temperature"].max()
minimum_temperature = temperature_data.groupby("City")["Temperature"].min()

temperature_range = maximum_temperature - minimum_temperature

highest_range_city = temperature_range.idxmax()
most_consistent_city = standard_deviation.idxmin()

print("Mean Temperature for Each City")
print(mean_temperature)

print("\nStandard Deviation for Each City")
print(standard_deviation)

print("\nTemperature Range for Each City")
print(temperature_range)

print("\nCity with Highest Temperature Range:", highest_range_city)
print("City with Most Consistent Temperature:", most_consistent_city)
