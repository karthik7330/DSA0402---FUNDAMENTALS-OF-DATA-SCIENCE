import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
players = pd.read_csv("soccer_players.csv")

# Top 5 players by goals
top_goals = players.sort_values(
    by="Goals",
    ascending=False
).head(5)

print("Top 5 Players with Highest Goals")
print(top_goals[["Name", "Goals"]])

# Top 5 players by salary
top_salary = players.sort_values(
    by="Weekly Salary",
    ascending=False
).head(5)

print("\nTop 5 Highest Paid Players")
print(top_salary[["Name", "Weekly Salary"]])

# Calculate average age
average_age = players["Age"].mean()

print("\nAverage Age:", average_age)

# Players above average age
above_average = players[players["Age"] > average_age]

print("\nPlayers Above Average Age")
print(above_average[["Name", "Age"]])

# Count players by position
position_count = players["Position"].value_counts()

print("\nPlayers by Position")
print(position_count)

# Bar chart
plt.figure(figsize=(8, 5))

plt.bar(position_count.index, position_count.values)

plt.title("Distribution of Players by Position")
plt.xlabel("Position")
plt.ylabel("Number of Players")

plt.show()
