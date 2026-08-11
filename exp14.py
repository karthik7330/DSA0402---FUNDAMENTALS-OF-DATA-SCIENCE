import numpy as np
import matplotlib.pyplot as plt

study_time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([45, 50, 55, 60, 65, 70, 75, 82, 88, 92])

correlation = np.corrcoef(study_time, exam_scores)[0, 1]

print("Correlation Coefficient:", correlation)

plt.figure(figsize=(7, 4))
plt.scatter(study_time, exam_scores)
plt.title("Study Time vs Exam Scores")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(study_time, exam_scores, marker="o")
plt.title("Study Time and Exam Score Trend")
plt.xlabel("Study Time (Hours)")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()
