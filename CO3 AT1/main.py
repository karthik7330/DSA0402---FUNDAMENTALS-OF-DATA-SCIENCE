import random
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Generate Synthetic LMS Dataset
# -------------------------------

random.seed(42)

student_id = []
lms_hours = []
academic_score = []
attendance = []

for i in range(1, 201):

    r = random.random()

    if r < 0.82:
        hours = max(0.5, min(random.gauss(3.2, 0.9), 6))
    elif r < 0.96:
        hours = random.uniform(6, 10)
    else:
        hours = random.uniform(20, 25)

    score = min(100, max(45, 55 + hours * 2.2 + random.gauss(0, 6)))
    attend = min(100, max(65, 70 + hours * 2 + random.gauss(0, 4)))

    student_id.append(f"S{i:03d}")
    lms_hours.append(round(hours, 2))
    academic_score.append(round(score, 1))
    attendance.append(round(attend, 1))

# -------------------------------
# Create DataFrame
# -------------------------------

data = pd.DataFrame({
    "Student_ID": student_id,
    "LMS_Hours": lms_hours,
    "Academic_Score": academic_score,
    "Attendance": attendance
})

# Save Dataset
data.to_excel("LMS_Synthetic_Dataset.xlsx", index=False)

print("Dataset Created Successfully!")

# -----------------------------------
# Histogram
# -----------------------------------

plt.figure(figsize=(7,5))

plt.hist(data["LMS_Hours"], bins=15, color="skyblue", edgecolor="black")

plt.title("Histogram of LMS Usage")
plt.xlabel("Hours per Week")
plt.ylabel("Number of Students")

plt.savefig("Histogram_LMS.png", dpi=300)

plt.show()

# -----------------------------------
# Box Plot
# -----------------------------------

plt.figure(figsize=(7,4))

plt.boxplot(data["LMS_Hours"], vert=False)

plt.title("Box Plot of LMS Usage")
plt.xlabel("Hours per Week")

plt.savefig("BoxPlot_LMS.png", dpi=300)

plt.show()

# -----------------------------------
# Scatter Plot
# -----------------------------------

plt.figure(figsize=(7,5))

plt.scatter(data["LMS_Hours"], data["Academic_Score"])

plt.title("LMS Usage vs Academic Performance")
plt.xlabel("Hours per Week")
plt.ylabel("Academic Score")

plt.savefig("Scatter_LMS_vs_Score.png", dpi=300)

plt.show()

# -----------------------------------
# Pie Chart
# -----------------------------------

low = len(data[data["LMS_Hours"] < 2])

moderate = len(data[(data["LMS_Hours"] >= 2) &
                    (data["LMS_Hours"] <= 6)])

high = len(data[data["LMS_Hours"] > 6])

labels = ["Low", "Moderate", "High"]

sizes = [low, moderate, high]

plt.figure(figsize=(6,6))

plt.pie(sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Student Engagement Levels")

plt.savefig("Pie_Engagement.png", dpi=300)

plt.show()

print("Charts Generated Successfully!")