import math
mean=498
std=12
n=50
z=1.96
se=std/math.sqrt(n)
me=z*se
print("95% CI:",(mean-me,mean+me))
