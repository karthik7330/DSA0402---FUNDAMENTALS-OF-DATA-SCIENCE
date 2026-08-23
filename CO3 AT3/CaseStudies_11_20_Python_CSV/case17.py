import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('traffic_data.csv')
print(df.describe())
print(df.quantile([0.25,0.5,0.75]))

num_cols=df.select_dtypes(include='number').columns

plt.hist(df[num_cols[0]])
plt.title(num_cols[0])
plt.show()

if len(num_cols)>1:
    plt.boxplot(df[num_cols[1]])
    plt.title(num_cols[1])
    plt.show()

if len(num_cols)>2:
    plt.scatter(df[num_cols[0]],df[num_cols[1]])
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.show()
