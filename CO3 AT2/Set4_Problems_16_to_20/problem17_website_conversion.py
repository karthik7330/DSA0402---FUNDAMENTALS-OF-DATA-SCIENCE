import pandas as pd
import math
df=pd.read_csv("problem17_website_conversion.csv")
x1,n1=df.loc[0,"Purchases"],df.loc[0,"Visitors"]
x2,n2=df.loc[1,"Purchases"],df.loc[1,"Visitors"]
p1=x1/n1
p2=x2/n2
p=(x1+x2)/(n1+n2)
se=math.sqrt(p*(1-p)*(1/n1+1/n2))
z=(p2-p1)/se
print("Conversion A =",round(p1*100,2),"%")
print("Conversion B =",round(p2*100,2),"%")
print("Z =",round(z,2))
