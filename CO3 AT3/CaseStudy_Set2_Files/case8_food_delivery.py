import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv('case8_food_delivery.csv')
print(df.describe())
print(ttest_ind(df[df.Delivery_Time<=30]['Customer_Rating'],df[df.Delivery_Time>30]['Customer_Rating']))