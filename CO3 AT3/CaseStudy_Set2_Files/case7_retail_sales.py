import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case7_retail_sales.csv')
print(df.describe())
print(ttest_ind(df[df.Discount_Percentage>20]['Daily_Sales'],df[df.Discount_Percentage<=20]['Daily_Sales']))