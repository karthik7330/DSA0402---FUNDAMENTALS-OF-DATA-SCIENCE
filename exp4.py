import numpy as np

sales_data = np.array([120000, 150000, 180000, 210000])

total_sales = np.sum(sales_data)

percentage_increase = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100

print("Quarterly Sales:", sales_data)
print("Total Sales for the Year:", total_sales)
print("Percentage Increase from Q1 to Q4: {:.2f}%".format(percentage_increase))