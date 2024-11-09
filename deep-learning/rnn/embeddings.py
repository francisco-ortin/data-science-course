import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# We use ELMo (Embeddings from Language Models) [https://tfhub.dev/google/elmo/3]
# to get embeddings for individual words. ELMo is a deep contextualized word
# representation model trained on a large corpus of text. It uses a deep bidirectional LSTM network to learn the
# contextual representations of words in a sentence.
# ELMo embeddings are 1024-dimensional vectors for each word in a sentence.

# To compute the similarity between two words, we can use [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
# between their ELMo embeddings.
# Cosine similarity is a measure of similarity between two non-zero vectors, giving a value between -1 and 1.
# A value of 1 means the vectors are identical, 0 means they are orthogonal, and -1 means they are opposite.
# [Inner product](https://en.wikipedia.org/wiki/Dot_product) between two vectors,
# which also measures similarity, has a scale that depends on the magnitude of the vectors,
# while cosine similarity is normalized by the product of the magnitudes of the vectors.
# Inner product is also called dot product.

# Load the ELMo model from TensorFlow Hub
elmo_model = hub.load("https://tfhub.dev/google/elmo/3")


def word_to_embedding(word: str, elmo_model_p) -> np.ndarray:
    # Get the ELMo embeddings for the word
    word_embeddings = elmo_model_p.signatures['default'](tf.constant([word]))['elmo']
    # the shape of embeddings is (batch_size, num_tokens, embedding_dim) = (1, 1, 1024)
    return word_embeddings[0][0]


word_pairs = [
    # related words
    ("apple", "banana"), ("apple", "orange"), ("apple", "apple"), ("king", "queen"), ("red", "yellow"),
    # synonyms
    ("film", "movie"), ("car", "automobile"), ("big", "large"),
    # unrelated words
    ("king", "eat"), ("drink", "movie"),
]

for word_pair in word_pairs:
    # Get the ELMo embeddings for the words
    embeddings_pair = [word_to_embedding(word, elmo_model) for word in word_pair]
    # Calculate cosine similarity between the words
    similarity_score = cosine_similarity([embeddings_pair[0]], [embeddings_pair[1]])
    print(f"Cosine similarity between '{word_pair[0]}' and '{word_pair[1]}': {similarity_score[0][0]}")


## 2D representation of word embeddings using t-SNE


def visualize_embeddings(embeddings_p: np.array, words_p: list[str]) -> None:
    # visualize the embeddings in 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings_p)
    # Plot the 2D embeddings with scatter plot
    plt.figure(figsize=(10, 10))
    # plot the dots
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue')
    # plot each word close to the dot
    for i, word in enumerate(words_p):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.show()


# we take all the words from the word_pairs list
words = list(set([word for word_pair in word_pairs for word in word_pair]))
# Get the ELMo embeddings for all the words
embeddings = np.array([word_to_embedding(word, elmo_model) for word in words])
visualize_embeddings(embeddings, words)

# Questions:
# 1. Are related words closer in the 2D representation?
# Yes, mostly.
# 2. Are there any words close that you thought they would not be close?
# Yes, fruits and colors seem to be in the same group/cluster.
# 3. Why do you think that is the case?
# That is because orange is both a fruit and a color.

