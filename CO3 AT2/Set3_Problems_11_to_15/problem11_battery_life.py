import math

mu=10
xbar=9.5
s=1.8
n=36
z=(xbar-mu)/(s/math.sqrt(n))
print("Z statistic:",round(z,2))
if abs(z)>1.96:
    print("Reject H0")
else:
    print("Fail to Reject H0")
