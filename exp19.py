import pandas as pd
import string

review_data = pd.read_csv("customer_reviews.csv")

frequency = {}

for review in review_data["Review"]:
    review = review.lower()
    review = review.translate(str.maketrans("", "", string.punctuation))
    words = review.split()

    for word in words:
        if word in frequency:
            frequency[word] = frequency[word] + 1
        else:
            frequency[word] = 1

print("Word Frequency Distribution:")

for word in frequency:
    print(word, ":", frequency[word])
