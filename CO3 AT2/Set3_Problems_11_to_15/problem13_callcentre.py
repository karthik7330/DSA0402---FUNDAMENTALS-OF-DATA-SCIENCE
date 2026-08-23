import math

mean=4.8
std=1.5
n=100
z=1.96
se=std/math.sqrt(n)
m=z*se
print("95% CI:",(round(mean-m,2),round(mean+m,2)))
