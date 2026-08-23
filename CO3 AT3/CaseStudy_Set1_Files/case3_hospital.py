import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case3_hospital.csv')
print(df.describe())
print(ttest_ind(df[df.Treatment=='New']['Recovery_Time'],df[df.Treatment=='Existing']['Recovery_Time']))