import pandas as pd
import matplotlib.pyplot as plt
import string

data = pd.read_csv("data.csv")

stop_words = [
    "the", "and", "is", "a", "an", "of", "to",
    "in", "for", "on", "with", "this", "that",
    "it", "was", "are", "as", "be", "very"
]

frequency = {}

for feedback in data["feedback"]:
    feedback = feedback.lower()
    feedback = feedback.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = feedback.split()

    for word in words:
        if word not in stop_words and word != "":
            if word in frequency:
                frequency[word] = frequency[word] + 1
            else:
                frequency[word] = 1

n = int(input("Enter the number of top words: "))

sorted_words = sorted(
    frequency.items(),
    key=lambda x: x[1],
    reverse=True
)

top_words = sorted_words[:n]

print("\nTop", n, "Most Frequent Words:")

for word, count in top_words:
    print(word, ":", count)

words = []
counts = []

for word, count in top_words:
    words.append(word)
    counts.append(count)

plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.title("Top Most Frequent Words in Customer Feedback")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
