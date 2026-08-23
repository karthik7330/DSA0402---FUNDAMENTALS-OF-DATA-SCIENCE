import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case6_online_learning.csv')
print(df.describe())
print(ttest_ind(df[df.Video_Watch_Percentage>70]['Quiz_Score'],df[df.Video_Watch_Percentage<=70]['Quiz_Score']))