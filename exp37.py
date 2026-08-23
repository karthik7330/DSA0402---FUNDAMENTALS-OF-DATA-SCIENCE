import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

d=pd.read_csv("exp37_customer_data.csv")
fs=["Annual_Income","Spending_Score","Purchase_Frequency","Average_Order_Value"]
s=StandardScaler(); X=s.fit_transform(d[fs])
k=int(input("Enter the number of customer segments (K): "))
m=KMeans(n_clusters=k,random_state=42,n_init=10); m.fit(X)
d["Segment"]=m.labels_; print(d[["Customer_ID","Segment"]])
a=float(input("Enter Annual Income: ")); b=float(input("Enter Spending Score: "))
c=float(input("Enter Purchase Frequency: ")); e=float(input("Enter Average Order Value: "))
n=pd.DataFrame({"Annual_Income":[a],"Spending_Score":[b],"Purchase_Frequency":[c],"Average_Order_Value":[e]})
print("New Customer Segment:",m.predict(s.transform(n))[0])
