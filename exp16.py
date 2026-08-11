import string

file = open("sample_text.txt", "r")
text = file.read()
file.close()

text = text.lower()
text = text.translate(str.maketrans("", "", string.punctuation))
words = text.split()

frequency = {}

for word in words:
    if word in frequency:
        frequency[word] = frequency[word] + 1
    else:
        frequency[word] = 1

print("Word Frequency Distribution:")

for word in frequency:
    print(word, ":", frequency[word])
