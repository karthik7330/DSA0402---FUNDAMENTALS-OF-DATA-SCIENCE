import pandas as pd

property_data = pd.read_csv("property.csv")

average_price = property_data.groupby("Location")["Listing Price"].mean()

properties_more_than_4 = property_data[property_data["Bedrooms"] > 4]

count_properties = len(properties_more_than_4)

largest_property = property_data.loc[property_data["Area"].idxmax()]

print("Average Listing Price by Location")
print(average_price)

print("\nNumber of Properties with More Than 4 Bedrooms:")
print(count_properties)

print("\nProperty with Largest Area")
print(largest_property)