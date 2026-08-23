import pandas as pd
import math
df=pd.read_csv("problem8_exam_scores.csv")
mean=df["Score"].mean()
std=df["Score"].std()
se=std/math.sqrt(len(df))
print("Mean:",mean)
print("Standard Deviation:",std)
print("Standard Error:",se)
