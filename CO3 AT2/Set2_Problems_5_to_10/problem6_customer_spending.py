import pandas as pd
df=pd.read_csv("problem6_customer_spending.csv")
print("Average Spending:",df["Spending"].mean())
print("Standard Deviation:",df["Spending"].std())
