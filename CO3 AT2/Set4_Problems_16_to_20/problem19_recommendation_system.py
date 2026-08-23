import pandas as pd
import math
df=pd.read_csv("problem19_recommendation_system.csv")
p1,p2=df['CTR']
n1,n2=df['Users']
pool=((p1*n1)+(p2*n2))/(n1+n2)
se=math.sqrt(pool*(1-pool)*((1/n1)+(1/n2)))
z=(p2-p1)/se
print("Z Statistic =",round(z,2))
