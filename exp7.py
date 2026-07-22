import pandas as pd

order_data = pd.read_csv("orders.csv")

order_data["Order Date"] = pd.to_datetime(
    order_data["Order Date"],
    format="%d-%m-%Y"
)

orders_per_customer = order_data.groupby("Customer ID")["Customer ID"].count()

average_quantity = order_data.groupby("Product Name")["Order Quantity"].mean()

earliest_date = order_data["Order Date"].min()
latest_date = order_data["Order Date"].max()

print("Total Orders by Each Customer")
print(orders_per_customer)

print("\nAverage Order Quantity for Each Product")
print(average_quantity)

print("\nEarliest Order Date:", earliest_date.date())
print("Latest Order Date:", latest_date.date())