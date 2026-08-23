import pandas as pd
from scipy.stats import ttest_1samp
df=pd.read_csv('case2_renewable_energy.csv')
print(df.describe())
print(ttest_1samp(df['Energy_Generation'],500))