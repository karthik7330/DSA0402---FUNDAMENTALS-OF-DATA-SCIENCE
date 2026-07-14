import numpy as np

house_data = np.loadtxt("houses.csv", delimiter=",", skiprows=1)

houses = house_data[house_data[:, 0] > 4]
average_price = np.mean(houses[:, 2])

print("Average Sale Price:", average_price)