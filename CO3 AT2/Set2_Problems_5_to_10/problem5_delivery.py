import pandas as pd
df=pd.read_csv("problem5_delivery.csv")
print("Mean:",df["Delivery_Time_Minutes"].mean())
print("Variance:",df["Delivery_Time_Minutes"].var())
