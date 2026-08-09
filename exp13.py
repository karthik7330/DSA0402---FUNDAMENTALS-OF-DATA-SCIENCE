import pandas as pd
import numpy as np

stock_data = pd.read_csv("stock_data.csv")

closing_prices = stock_data["Close"]

average_price = np.mean(closing_prices)
variability = np.std(closing_prices)

minimum_price = np.min(closing_prices)
maximum_price = np.max(closing_prices)

print("Average Closing Price:", average_price)
print("Standard Deviation:", variability)
print("Minimum Closing Price:", minimum_price)
print("Maximum Closing Price:", maximum_price)