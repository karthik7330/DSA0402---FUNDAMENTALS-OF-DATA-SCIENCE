import pandas as pd
import math
df=pd.read_csv("problem20_salary_prediction.csv")
mean=df.loc[0,'Mean_Difference']
sd=df.loc[0,'Standard_Deviation']
n=df.loc[0,'Test_Folds']
t=mean/(sd/math.sqrt(n))
print("t Statistic =",round(t,2))
