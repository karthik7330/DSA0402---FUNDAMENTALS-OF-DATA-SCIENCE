import numpy as np

student = np.array([
    [85, 90, 97, 81],
    [90, 81, 81, 92],
    [72, 81, 80, 79],
    [88, 94, 89, 90]
])

subjects = np.array(["Math", "Science", "English", "History"])

avg = np.mean(student, axis=0)

high_avg_i = np.argmax(avg)

print("Average Scores:")

for i in range(4):
    print(subjects[i], ":", avg[i])

print("\nSubject with Highest Average Score:",
      subjects[high_avg_i])
