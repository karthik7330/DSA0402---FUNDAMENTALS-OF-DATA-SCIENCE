import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("exp40_ecommerce_transactions.csv")
c=d.groupby("Customer_ID").agg({"Transaction_Amount":"sum","Items_Purchased":"sum"}).reset_index()
c.columns=["Customer_ID","Total_Spent","Total_Items"]
s=StandardScaler(); Z=s.fit_transform(c[["Total_Spent","Total_Items"]])
k=int(input("Enter the number of customer segments: "))
m=KMeans(n_clusters=k,random_state=42,n_init=10); m.fit(Z); c["Cluster"]=m.labels_
print(c); print(c.groupby("Cluster")[["Total_Spent","Total_Items"]].mean())
print(c["Cluster"].value_counts().sort_index())
for x in range(k):
    z=c[c["Cluster"]==x]; plt.scatter(z["Total_Spent"],z["Total_Items"],label="Cluster "+str(x))
plt.title("E-Commerce Customer Segmentation"); plt.xlabel("Total Amount Spent"); plt.ylabel("Total Items Purchased"); plt.legend(); plt.grid(); plt.show()
