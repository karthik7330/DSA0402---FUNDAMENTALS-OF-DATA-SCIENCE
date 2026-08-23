import pandas as pd
df=pd.read_csv("problem2_ecommerce.csv")
print("Mean:",df["Purchase_Amount"].mean())
print("Variance:",df["Purchase_Amount"].var())
print("Std Dev:",df["Purchase_Amount"].std())
print("\nCovariance Matrix")
print(df[["Website_Visits","Time_Spent","Purchase_Amount"]].cov())
print("\nCorrelation Matrix")
print(df[["Website_Visits","Time_Spent","Purchase_Amount"]].corr())
