import pandas as pd

sales_data = pd.read_csv("sales.csv")

top_products = sales_data.groupby("Product Name")["Order Quantity"].sum()

top_products = top_products.sort_values(ascending=False)

top_5 = top_products.head(5)

print("Top 5 Most Sold Products")
print(top_5)