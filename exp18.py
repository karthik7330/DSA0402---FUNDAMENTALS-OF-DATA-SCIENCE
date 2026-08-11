import pandas as pd

social_data = pd.read_csv("social_media.csv")

like_frequency = social_data["Likes"].value_counts().sort_index()

print("Frequency Distribution of Likes:")
print(like_frequency)
