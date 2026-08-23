import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case5_banking.csv')
print(df.describe(include='all'))
print(ttest_ind(df[df.Loan_Status=='Approved']['Credit_Score'],df[df.Loan_Status=='Rejected']['Credit_Score']))