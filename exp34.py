import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

d=pd.read_csv("exp34_car_prices.csv")
X=d[["Engine_Size","Horsepower","Fuel_Efficiency","Age"]]; y=d["Price"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
m=LinearRegression(); m.fit(Xtr,ytr); p=m.predict(Xte)
print("Model Coefficients:")
for i in range(len(X.columns)): print(X.columns[i],":",m.coef_[i])
print("Intercept:",m.intercept_)
print("MAE:",mean_absolute_error(yte,p)); print("MSE:",mean_squared_error(yte,p)); print("R2:",r2_score(yte,p))
plt.scatter(yte,p); plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.title("Actual vs Predicted"); plt.grid(); plt.show()
