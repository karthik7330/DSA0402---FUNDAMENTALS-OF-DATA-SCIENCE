import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

d=pd.read_csv("exp31_treatment_data.csv")
g=LabelEncoder(); d["Gender"]=g.fit_transform(d["Gender"])
o=LabelEncoder(); d["Treatment_Outcome"]=o.fit_transform(d["Treatment_Outcome"])
X=d[["Age","Gender","Blood_Pressure","Cholesterol"]]; y=d["Treatment_Outcome"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)
k=int(input("Enter the value of K: ")); m=KNeighborsClassifier(n_neighbors=k); m.fit(Xtr,ytr)
p=m.predict(Xte)
for i in range(len(yte)):
    print("Actual:",o.inverse_transform([yte.iloc[i]])[0],"Predicted:",o.inverse_transform([p[i]])[0])
print("Accuracy :",accuracy_score(yte,p))
print("Precision:",precision_score(yte,p,zero_division=0))
print("Recall   :",recall_score(yte,p,zero_division=0))
print("F1-Score :",f1_score(yte,p,zero_division=0))
