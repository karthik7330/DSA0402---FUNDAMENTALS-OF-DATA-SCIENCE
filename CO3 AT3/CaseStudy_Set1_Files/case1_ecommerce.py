import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case1_ecommerce.csv')
print(df.describe())
print(ttest_ind(df[df.Membership=='Premium']['Purchase_Amount'],df[df.Membership=='Regular']['Purchase_Amount']))