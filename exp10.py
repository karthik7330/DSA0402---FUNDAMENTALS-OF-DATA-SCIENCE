import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
sales = [12000, 15000, 18000, 17000, 21000, 25000]


plt.figure(figsize=(6,4))
plt.plot(months, sales, marker='o')
plt.title("Monthly Sales - Line Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.grid(True)
plt.show()


plt.figure(figsize=(6,4))
plt.bar(months, sales)
plt.title("Monthly Sales - Bar Plot")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.show()