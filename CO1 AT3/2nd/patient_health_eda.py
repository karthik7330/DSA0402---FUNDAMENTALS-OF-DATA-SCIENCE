"""
Patient Health Risk EDA - Healthcare Analytics Dataset
Full runnable script - reproduces every table, statistic and chart.
Usage: place patient_health.csv in a ./data/ folder next to this
script, then run:  python patient_health_eda.py
"""

# # Healthcare Analytics Dataset — Patient Health Risk EDA
#
# **Question:** A hospital wants to analyze patient health risk based on age,
# blood sugar, blood pressure, BMI, and disease status. Perform EDA to find
# important health patterns.
#
# **Author:** Data Analytics Assignment
# **Dataset:** `patient_health.csv` (20 patient records, 6 columns)
#

# ## Common EDA Tasks Checklist (per updated assessment rubric)
#
# This notebook explicitly covers the following 8 standard EDA tasks:
#
# 1. **Number of rows and columns** — Task 1 & 2 section
# 2. **Data types of each column** — Task 1 & 2 section
# 3. **Missing values and duplicate values** — Task 3a / 3b / 3c sections
# 4. **Mean, median, minimum, maximum, and standard deviation** — Task 4 section
# 5. **Bar chart, histogram, scatter plot, and box plot** (plus pie chart, line
#    chart, and heatmap as bonus visualizations) — Task 5a–5e sections
# 6. **Important patterns in the dataset** — Task 6 sections (group
#    comparisons and correlation heatmap)
# 7. **4–5 observations based on EDA** — Task 7 section
# 8. **Short conclusion from the analysis** — Task 8 section
#
# ### Rubric Criteria Alignment (Weightage)
#
# | Rubric Criteria | Weightage | Where it's covered |
# |---|---|---|
# | Dataset Understanding | 20 | Section 2 (dataset description, features, purpose) |
# | Data Cleaning | 25 | Tasks 3a–3c (missing values, duplicates, inconsistent/incorrect entries) |
# | Data Analysis and Statistical Summary | 20 | Task 4 (descriptive statistics) + Task 6 (patterns) |
# | Data Visualization | 20 | Task 5a–5e (bar, pie, histogram, box, scatter, line, heatmap — all labeled) |
# | Insight Generation and Reporting | 15 | Task 7 (observations) + Task 8 (conclusion) |
#

# ## 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11

pd.set_option('display.max_columns', None)
print("Libraries imported successfully.")

def show_df(dframe, n=None):
    """Pretty-print a DataFrame (use IPython display() instead in Jupyter)."""
    print((dframe.head(n) if n else dframe).to_string())


# ## 2. Load the Dataset

df = pd.read_csv("data/patient_health.csv")
show_df(df, 20)


# ### Dataset Description
#
# | Item | Detail |
# |---|---|
# | **Records (rows)** | 20 patients |
# | **Features (columns)** | 6 |
# | **Patient_ID** | Object (string) — unique identifier for each patient |
# | **Age** | Integer — patient age in years |
# | **Sugar_Level** | Integer — blood sugar level (mg/dL) |
# | **BP** | Integer — systolic blood pressure (mmHg) |
# | **BMI** | Integer — Body Mass Index (kg/m²) |
# | **Disease_Status** | Object (categorical) — `Yes` / `No`, whether the patient has been diagnosed with the disease being studied |
#
# **Purpose & significance:** This dataset allows the hospital to explore how
# routinely-collected vitals (age, blood sugar, blood pressure, BMI) relate to
# a patient's disease status. Identifying these patterns supports **early
# risk screening**, helps prioritise patients for preventive care, and gives
# clinicians a data-driven basis for flagging high-risk profiles before
# symptoms escalate — a common use case for conditions such as diabetes or
# hypertension-related illness.
#

# ## Task 1 & 2: Number of Rows/Columns and Data Types

print("Shape (rows, columns):", df.shape)
print()
print("Column data types:")
print(df.dtypes)


print("General info:")
df.info()


# ## Task 3a: Missing Values

missing = df.isnull().sum()
print("Missing values per column:")
print(missing)
print()
print("Total missing values in dataset:", df.isnull().sum().sum())


# **Observation:** The dataset has **no missing values** in any column.
# No imputation is required here. (In a real hospital dataset we would
# typically impute missing vitals with the column median rather than drop
# patient records, since every record matters clinically.)
#

# ## Task 3b: Duplicate Values

dup_rows = df.duplicated().sum()
dup_ids = df['Patient_ID'].duplicated().sum()
print("Fully duplicated rows:", dup_rows)
print("Duplicated Patient_IDs:", dup_ids)

before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
print(f"Rows before: {before}, after de-duplication: {after}")


# **Observation:** No duplicate rows or duplicate Patient IDs were found; each patient appears exactly once.

# ## Task 3c: Data Cleaning — Consistency Checks (Incorrect Entries & Range Validation)

# Standardise text fields
df['Patient_ID'] = df['Patient_ID'].str.strip()
df['Disease_Status'] = df['Disease_Status'].str.strip().str.capitalize()

print("Unique values in 'Disease_Status':", df['Disease_Status'].unique().tolist())

# Clinical range/sanity checks on numeric columns
checks = {
    'Age (0-120)':            df['Age'].between(0, 120).all(),
    'Sugar_Level (40-400 mg/dL)': df['Sugar_Level'].between(40, 400).all(),
    'BP (70-220 mmHg)':        df['BP'].between(70, 220).all(),
    'BMI (10-60 kg/m2)':       df['BMI'].between(10, 60).all(),
}
for label, ok in checks.items():
    print(f"{label}: {'OK' if ok else 'OUT OF RANGE VALUES FOUND'}")

# Convert Disease_Status to a proper categorical dtype
df['Disease_Status'] = df['Disease_Status'].astype('category')
print()
print("Disease_Status column dtype now:", df['Disease_Status'].dtype)


# **Observation:** All vitals fall within clinically plausible ranges — no
# negative ages, implausible sugar levels, or impossible BP/BMI values.
# `Disease_Status` contains exactly two clean categories (`Yes`/`No`) and has
# been converted to a `category` dtype.
#

# ## Task 4: Descriptive Statistics — Mean, Median, Min, Max, Std

num_cols = ['Age', 'Sugar_Level', 'BP', 'BMI']
print("Descriptive statistics for numerical columns:")
show_df(df[num_cols].describe().round(2))


print("Additional statistics:")
summary = pd.DataFrame({
    'mean':   df[num_cols].mean(),
    'median': df[num_cols].median(),
    'std':    df[num_cols].std(),
    'skew':   df[num_cols].skew(),
})
show_df(summary.round(2))


# ## Task 5a: Bar Chart & Pie Chart — Categorical Analysis (Disease_Status)

status_counts = df['Disease_Status'].value_counts()
status_pct = df['Disease_Status'].value_counts(normalize=True).mul(100).round(1)

print("Counts:")
print(status_counts)
print()
print("Percentage:")
print(status_pct.astype(str) + '%')


colors = ['#D64550', '#4C956C']  # Yes = red (risk), No = green (healthy)
fig, ax = plt.subplots()
bars = ax.bar(status_counts.index, status_counts.values, color=colors, edgecolor='black')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, str(int(b.get_height())),
            ha='center', va='bottom', fontweight='bold')
ax.set_title('Bar Chart: Number of Patients by Disease Status')
ax.set_xlabel('Disease Status')
ax.set_ylabel('Number of Patients')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90, wedgeprops={'edgecolor':'white'})
ax.set_title('Pie Chart: Proportion of Patients With vs Without Disease')
ax.legend(status_counts.index, title="Disease Status", loc="best")
plt.tight_layout()
plt.show()


# **Observation:** 10 of 20 patients (50%) are diagnosed positive (`Yes`) and 10 (50%) are negative (`No`) — a perfectly balanced sample for comparison.

# ## Task 5b: Histogram & Box Plot — Numerical Distributions

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
colors_hist = ['#4C72B0', '#DD8452', '#55A868', '#8172B2']

for ax, col, c in zip(axes, num_cols, colors_hist):
    ax.hist(df[col], bins=6, color=c, edgecolor='black', alpha=0.85)
    ax.set_title(f'Histogram: {col} Distribution')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, col, c in zip(axes, num_cols, colors_hist):
    ax.boxplot(df[col], vert=True, patch_artist=True,
               boxprops=dict(facecolor=c, alpha=0.6))
    ax.set_title(f'Box Plot: {col}')
    ax.set_ylabel(col)
    ax.set_xticks([])

plt.tight_layout()
plt.show()


# **Observation:** No extreme outliers appear across Age, Sugar_Level, BP or
# BMI — all four vitals show fairly symmetric, moderate spreads consistent
# with a general adult patient population.
#

# ## Task 6a: Important Patterns — Vitals vs Disease Status

fig, axes = plt.subplots(1, 4, figsize=(19, 4.5))
for ax, col, c in zip(axes, num_cols, colors_hist):
    data_yes = df[df['Disease_Status'] == 'Yes'][col]
    data_no = df[df['Disease_Status'] == 'No'][col]
    bp = ax.boxplot([data_yes, data_no], tick_labels=['Yes', 'No'], patch_artist=True)
    for patch, col2 in zip(bp['boxes'], colors):
        patch.set_facecolor(col2)
        patch.set_alpha(0.6)
    ax.set_title(f'{col} by Disease Status')
    ax.set_ylabel(col)

plt.tight_layout()
plt.show()


group_means = df.groupby('Disease_Status', observed=True)[num_cols].mean().round(2)
show_df(group_means)


fig, ax = plt.subplots(figsize=(9,5))
x = np.arange(len(num_cols))
width = 0.35
yes_vals = group_means.loc['Yes']
no_vals = group_means.loc['No']

ax.bar(x - width/2, yes_vals, width, label='Yes (Disease)', color='#D64550', edgecolor='black')
ax.bar(x + width/2, no_vals, width, label='No (Healthy)', color='#4C956C', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(num_cols)
ax.set_ylabel('Average Value')
ax.set_title('Bar Chart: Average Age / Sugar Level / BP / BMI by Disease Status')
ax.legend(title='Disease Status')
plt.tight_layout()
plt.show()


# **Observation:** Patients diagnosed with the disease (`Yes`) show clearly
# higher average Sugar_Level, BP and BMI, and are somewhat older on average,
# compared to patients without the disease (`No`) — all four vitals trend
# in the same direction as disease risk.
#

# ## Task 5c: Scatter Plot — Bivariate Relationships

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
pairs = [('Age', 'Sugar_Level'), ('Sugar_Level', 'BP'), ('BMI', 'BP')]

for ax, (xcol, ycol) in zip(axes, pairs):
    for status, c in zip(['Yes', 'No'], colors):
        subset = df[df['Disease_Status'] == status]
        ax.scatter(subset[xcol], subset[ycol], label=status, color=c,
                   edgecolor='black', s=70)
    ax.set_title(f'{xcol} vs {ycol}')
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.legend(title='Disease Status')

plt.tight_layout()
plt.show()


# **Observation:** Diseased patients (red) cluster toward the higher end of sugar level, blood pressure and BMI in every pairing, while healthy patients (green) cluster toward the lower end — a visibly separable pattern even with simple two-variable views.

# ## Task 5d: Line Chart — Trend Analysis (Bonus Visualization)

trend_df = df.sort_values('Age').reset_index(drop=True)

fig, ax1 = plt.subplots(figsize=(11,5))
ax1.plot(trend_df.index, trend_df['Sugar_Level'], marker='o', color='#DD8452', label='Sugar Level (mg/dL)')
ax1.plot(trend_df.index, trend_df['BP'], marker='s', color='#4C72B0', label='Blood Pressure (mmHg)')
ax1.plot(trend_df.index, trend_df['BMI']*4, marker='^', color='#55A868', label='BMI (x4 for scale)')
ax1.set_xlabel('Patients sorted by ascending Age')
ax1.set_ylabel('Value')
ax1.set_title('Line Chart: Sugar Level, BP & BMI Trend Across Patients (by Age)')
ax1.legend(loc='upper left')
plt.tight_layout()
plt.show()


# **Observation:** As patients are ordered by increasing age, sugar level, blood pressure and BMI all trend broadly upward together — suggesting these risk factors accumulate jointly with age in this sample.

# ## Task 6b: Important Patterns — Correlation Heatmap

df_corr = df.copy()
df_corr['Disease_Num'] = (df_corr['Disease_Status'] == 'Yes').astype(int)
corr_matrix = df_corr[num_cols + ['Disease_Num']].corr().round(2)
show_df(corr_matrix)


fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            square=True, linewidths=0.5, cbar_kws={'label': 'Correlation coefficient'}, ax=ax)
ax.set_title('Heatmap: Correlation Between Vitals and Disease Status')
plt.tight_layout()
plt.show()


# **Observation:** `Sugar_Level`, `BP` and `BMI` are all strongly positively
# correlated with each other and with `Disease_Num` (disease presence).
# `Age` also correlates positively with disease, but somewhat more weakly
# than the three direct clinical measurements — consistent with sugar
# level, blood pressure and BMI being the more proximate risk indicators.
#

# ## Task 7: Observations (4–5 Key Observations)
#
# 1. **Dataset is clean**: no missing values, no duplicate records, and all
#    vitals fall within clinically plausible ranges.
# 2. **Class balance**: exactly 50% of patients are disease-positive and
#    50% are disease-negative — an ideal balance for comparison.
# 3. **Sugar_Level, BP and BMI are all strongly positively correlated**
#    with each other and with Disease_Status.
# 4. Patients diagnosed with the disease show markedly higher average
#    sugar level, blood pressure, and BMI — scatter plots show largely
#    separable clusters between diseased and healthy patients.
# 5. **Age correlates positively but more moderately** with disease status
#    than the three direct clinical measurements.
#
# ## Task 8: Conclusion
#
# Sugar level, blood pressure, and BMI together form a strong composite
# risk signal for disease status in this dataset, while age plays a
# secondary, amplifying role. The hospital can use a combined score across
# these vitals — rather than any single measurement — to triage patients
# for closer monitoring or preventive care. Clinical decision thresholds
# would need validation on a much larger, representative patient
# population before real-world use.
#
