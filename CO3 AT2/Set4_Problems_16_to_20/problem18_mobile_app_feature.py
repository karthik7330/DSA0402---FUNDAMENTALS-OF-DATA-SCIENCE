import pandas as pd
import math
df=pd.read_csv("problem18_mobile_app_feature.csv")
m1,m2=df['Average_Engagement_Minutes']
s1,s2=df['Standard_Deviation']
n1,n2=df['Users']
se=math.sqrt((s1**2/n1)+(s2**2/n2))
z=(m2-m1)/se
print("Z Statistic =",round(z,2))
