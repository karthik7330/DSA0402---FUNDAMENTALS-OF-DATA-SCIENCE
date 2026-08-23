from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
model = DecisionTreeClassifier(random_state=42)
model.fit(iris.data, iris.target)

print("Enter Iris Flower Details")
sl=float(input("Sepal Length: "))
sw=float(input("Sepal Width: "))
pl=float(input("Petal Length: "))
pw=float(input("Petal Width: "))

p=model.predict([[sl,sw,pl,pw]])
print("Predicted Species:", iris.target_names[p[0]])
