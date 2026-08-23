import pandas as pd
import math
df=pd.read_csv("problem7_product_ratings.csv")
mean=df["Rating"].mean()
var=df["Rating"].var()
std=df["Rating"].std()
se=std/math.sqrt(len(df))
print("Mean:",mean)
print("Variance:",var)
print("Standard Error:",se)
