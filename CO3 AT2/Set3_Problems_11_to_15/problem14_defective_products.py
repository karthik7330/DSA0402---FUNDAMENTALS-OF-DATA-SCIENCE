import math

n=500
x=22
z=1.96
p=x/n
se=math.sqrt((p*(1-p))/n)
m=z*se
print("Defect Rate:",round(p*100,2),"%")
print("95% CI:",(round((p-m)*100,2),round((p+m)*100,2)),"%")
