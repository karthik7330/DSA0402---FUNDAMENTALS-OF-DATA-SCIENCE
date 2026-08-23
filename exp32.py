import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

d=pd.read_csv("exp32_houses.csv")
X=d[["Area","Bedrooms","Bathrooms","Age"]]; y=d["Price"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
m=LinearRegression(); m.fit(Xtr,ytr)

a=float(input("Enter Area in square feet: "))
b=int(input("Enter Number of Bedrooms: "))
c=int(input("Enter Number of Bathrooms: "))
e=int(input("Enter Age of House: "))

n=pd.DataFrame({"Area":[a],"Bedrooms":[b],"Bathrooms":[c],"Age":[e]})
print("Predicted House Price:",round(m.predict(n)[0],2))
