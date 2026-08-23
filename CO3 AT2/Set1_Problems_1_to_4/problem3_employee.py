import pandas as pd
df=pd.read_csv("problem3_employee.csv")
print("Mean:",df["Performance_Rating"].mean())
print("Variance:",df["Performance_Rating"].var())
print("Std Dev:",df["Performance_Rating"].std())
print("\nCovariance Matrix")
print(df[["Training_Hours","Projects_Completed","Performance_Rating"]].cov())
print("\nCorrelation Matrix")
print(df[["Training_Hours","Projects_Completed","Performance_Rating"]].corr())
