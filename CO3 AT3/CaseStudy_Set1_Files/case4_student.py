import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case4_student.csv')
print(df.describe())
print(ttest_ind(df[df.Attendance>75]['Final_Marks'],df[df.Attendance<=75]['Final_Marks']))