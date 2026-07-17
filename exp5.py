import numpy as np

fuel_efficiency = np.array([20, 24, 28, 32])

average_efficiency = np.mean(fuel_efficiency)

percentage_improvement = ((fuel_efficiency[3] - fuel_efficiency[0]) / fuel_efficiency[0]) * 100

print("Fuel Efficiency (MPG):", fuel_efficiency)
print("Average Fuel Efficiency:", average_efficiency)
print("Percentage Improvement from Model 1 to Model 4: {:.2f}%".format(percentage_improvement))