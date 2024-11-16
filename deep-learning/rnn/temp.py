example_reviews = ["The movie was a great waste of time. The plot was boring.",
                "I loved the movie. The plot was amazing."]
words = [review.split() for review in [review for review in example_reviews]]
print(words)
# replace "," and "." with "" in all the words
# convert all the words to lower case
words = [[word.replace(".", "").replace(",", "") for word in sequence] for sequence in words]
print(words)
word_i

#remove all the . and , from words

