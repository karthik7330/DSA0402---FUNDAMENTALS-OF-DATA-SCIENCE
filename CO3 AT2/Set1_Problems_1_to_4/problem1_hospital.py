import pandas as pd
df=pd.read_csv("problem1_hospital.csv")
print("Mean BP:",df["Systolic_BP"].mean())
print("Variance:",df["Systolic_BP"].var())
print("Std Dev:",df["Systolic_BP"].std())
print("\nCovariance Matrix")
print(df[["Daily_Exercise","BMI","Systolic_BP"]].cov())
print("\nCorrelation Matrix")
print(df[["Daily_Exercise","BMI","Systolic_BP"]].corr())
