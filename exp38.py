import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("exp38_ecommerce_customers.csv")
fs=["Age","Annual_Income","Website_Visits","Purchase_Frequency","Average_Order_Value","Spending_Score"]
s=StandardScaler(); X=s.fit_transform(d[fs])
k=int(input("Enter the number of customer segments: "))
m=KMeans(n_clusters=k,random_state=42,n_init=10); m.fit(X); d["Cluster"]=m.labels_
print(d[["Customer_ID","Cluster"]])
print(d.groupby("Cluster")[fs].mean())
print(d["Cluster"].value_counts().sort_index())
for c in range(k):
    z=d[d["Cluster"]==c]; plt.scatter(z["Annual_Income"],z["Spending_Score"],label="Cluster "+str(c))
plt.title("Customer Segmentation"); plt.xlabel("Annual Income"); plt.ylabel("Spending Score"); plt.legend(); plt.grid(); plt.show()
