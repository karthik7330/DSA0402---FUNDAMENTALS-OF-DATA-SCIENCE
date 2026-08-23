import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

d=pd.read_csv("exp33_houses.csv"); X=d[["Area"]]; y=d["Price"]
plt.scatter(X["Area"],y); plt.title("House Area vs House Price"); plt.xlabel("Area"); plt.ylabel("Price"); plt.show()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
m=LinearRegression(); m.fit(Xtr,ytr); p=m.predict(Xte)
print("Coefficient:",m.coef_[0],"Intercept:",m.intercept_)
print("MAE:",mean_absolute_error(yte,p)); print("MSE:",mean_squared_error(yte,p)); print("R2:",r2_score(yte,p))
plt.scatter(X,y); plt.plot(X,m.predict(X)); plt.title("Linear Regression"); plt.xlabel("Area"); plt.ylabel("Price"); plt.show()
