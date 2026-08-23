import pandas as pd
from scipy.stats import ttest_rel

df=pd.read_csv("problem12_marketing_campaign.csv")
t,p=ttest_rel(df["Sales_After"],df["Sales_Before"])
print("t =",round(t,2))
print("p =",p)
if p<0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")
