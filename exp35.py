import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

d=pd.read_csv("exp35_model_data.csv")
print(d.columns.tolist())
fs=input("Enter feature names separated by comma: ").split(",")
fs=[x.strip() for x in fs]
target=input("Enter target variable: ").strip()
X=d[fs]; y=d[target]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
m=LogisticRegression(max_iter=1000); m.fit(Xtr,ytr); p=m.predict(Xte)
for i in range(len(yte)): print("Actual:",yte.iloc[i],"Predicted:",p[i])
print("Accuracy :",accuracy_score(yte,p))
print("Precision:",precision_score(yte,p,zero_division=0))
print("Recall   :",recall_score(yte,p,zero_division=0))
print("F1-Score :",f1_score(yte,p,zero_division=0))
