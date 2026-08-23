import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

d=pd.read_csv("exp36_customer_churn.csv")
fs=["Age","Usage_Minutes","Contract_Months","Monthly_Charges","Support_Calls"]
X=d[fs]; y=d["Churn"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)
m=LogisticRegression(); m.fit(Xtr,ytr)
a=float(input("Enter Age: ")); b=float(input("Enter Usage Minutes: ")); c=float(input("Enter Contract Duration in Months: "))
e=float(input("Enter Monthly Charges: ")); f=float(input("Enter Number of Support Calls: "))
n=pd.DataFrame({"Age":[a],"Usage_Minutes":[b],"Contract_Months":[c],"Monthly_Charges":[e],"Support_Calls":[f]})
ns=s.transform(n); p=m.predict(ns); prob=m.predict_proba(ns)
print("Customer is likely to CHURN." if p[0]==1 else "Customer is likely to NOT CHURN.")
print("Churn Probability:",round(prob[0][1]*100,2),"%")
