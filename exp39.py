import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("exp39_retail_customers.csv")
X=d[["Total_Spent","Visit_Frequency"]]; s=StandardScaler(); Z=s.fit_transform(X)
k=int(input("Enter the number of customer segments: "))
m=KMeans(n_clusters=k,random_state=42,n_init=10); m.fit(Z); d["Cluster"]=m.labels_
print(d); print(d.groupby("Cluster")[["Total_Spent","Visit_Frequency"]].mean())
print(d["Cluster"].value_counts().sort_index())
for c in range(k):
    z=d[d["Cluster"]==c]; plt.scatter(z["Total_Spent"],z["Visit_Frequency"],label="Cluster "+str(c))
plt.title("Retail Customer Segmentation"); plt.xlabel("Total Amount Spent"); plt.ylabel("Visit Frequency"); plt.legend(); plt.grid(); plt.show()
