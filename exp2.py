import numpy as np

sales = np.array([
    [250, 300, 280],
    [450, 500, 480],
    [150, 200, 180]
])

avg = np.mean(sales)

print("Sales Data:")
print(sales)

print("\nAverage Price of All Products Sold:", avg)