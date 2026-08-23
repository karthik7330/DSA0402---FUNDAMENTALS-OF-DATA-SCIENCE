import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("cars.csv")

# Create encoders
brand_encoder = LabelEncoder()
engine_encoder = LabelEncoder()

# Encode categorical columns
data["Brand"] = brand_encoder.fit_transform(data["Brand"])
data["Engine_Type"] = engine_encoder.fit_transform(data["Engine_Type"])

# Separate features and target
X = data[["Mileage", "Age", "Brand", "Engine_Type"]]
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create CART regression model
model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Get user input
print("Enter New Car Details")

mileage = float(input("Enter Mileage: "))
age = float(input("Enter Age: "))

brand = input("Enter Brand (Toyota/Honda/Ford/BMW): ")
engine = input("Enter Engine Type (Petrol/Diesel): ")

# Encode user input
brand_code = brand_encoder.transform([brand])[0]
engine_code = engine_encoder.transform([engine])[0]

# Create input DataFrame
new_car = pd.DataFrame({
    "Mileage": [mileage],
    "Age": [age],
    "Brand": [brand_code],
    "Engine_Type": [engine_code]
})

# Predict price
predicted_price = model.predict(new_car)[0]

print("\nPredicted Car Price:", round(predicted_price, 2))

# Display decision path
tree_rules = export_text(
    model,
    feature_names=[
        "Mileage",
        "Age",
        "Brand",
        "Engine_Type"
    ]
)

print("\nDecision Tree Rules:")
print(tree_rules)