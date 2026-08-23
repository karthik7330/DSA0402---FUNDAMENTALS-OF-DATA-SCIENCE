import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

d=pd.read_csv("exp30_patients.csv")
X=d[["Fever","Cough","Fatigue","Headache"]]
y=d["Condition"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)

k=int(input("Enter the value of K: "))
m=KNeighborsClassifier(n_neighbors=k)
m.fit(Xtr,ytr)

a=int(input("Fever (0-No, 1-Yes): "))
b=int(input("Cough (0-No, 1-Yes): "))
c=int(input("Fatigue (0-No, 1-Yes): "))
e=int(input("Headache (0-No, 1-Yes): "))

p=m.predict([[a,b,c,e]])
print("Patient has the medical condition." if p[0]==1 else "Patient does not have the medical condition.")
