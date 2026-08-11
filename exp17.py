import pandas as pd

sales_data = pd.read_csv("customer_sales.csv")

age_frequency = sales_data["Age"].value_counts().sort_index()

print("Frequency Distribution of Customer Ages:")
print(age_frequency)
