# -*- coding: utf-8 -*-

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import zipfile

"""## Load the IMDb dataset

 We use the 5,000 most frequent words in the dataset. We cut the reviews to a maximum length of 80 words to speed up training. We use embeddings of 1024 dimensions and a large number of epochs (50) to train the models because we use early stopping. We only consider 1,000 sentences of the IMDb dataset because of memory restrictions (lower this number if you have memory issues).
"""


# Consider all the words with a frequency higher than this value. The higher, the more memory is needed.
vocabulary_size = 100_000
# Compute the maximum length of the reviews (for speeding up the training it is better to cut the reviews)
max_review_length = 80
# Max number of epochs to train the models (we use early stopping)
n_epochs = 50
# Embedding dimensions (ELMo embeddings have 1024 dimensions)
embedding_dim = 1024
# Number of sentences for validation and test, the remaining ones will be used for training
n_sentences_val, n_max_sentences_test = 500, 500

"""We load the dataset from the Keras API. Half of the reviews are used for training, and the other half for validation and testing."""

(X_train, y_train), (X_half, y_half) = keras.datasets.imdb.load_data(num_words=vocabulary_size)
assert n_sentences_val + n_max_sentences_test <= len(X_half), "Not enough sentences for validation and test."
X_test, X_val = X_half[-n_max_sentences_test:], X_half[-(n_sentences_val + n_max_sentences_test):-n_max_sentences_test]
y_test, y_val = y_half[-n_max_sentences_test:], y_half[-(n_sentences_val + n_max_sentences_test):-n_max_sentences_test]
# concat to X_train the remaining samples not used for validation and test
X_train = np.concatenate((X_train, X_half[:-(n_sentences_val + n_max_sentences_test)]), axis=0)
y_train = np.concatenate((y_train, y_half[:-(n_sentences_val + n_max_sentences_test)]), axis=0)
X_half, y_half = None, None  # free memory

print(f"Training sequences: {len(X_train):,}.\nValidation sequences: {len(X_val):,}.\nTesting sequences: {len(X_test):,}.")

"""## Process the dataset

We store all the word indexes returned by the IMDb dataset in a dictionary. An `index_to_word` dictionary is created to convert the token IDs back to words. We reserve the first four indices for special tokens `<PAD>`, `<START>`, `<OOV>`, and `<END>`.
"""

# Let's print some reviews. We need to convert the integers (token ids) back to words.
word_to_index = {word: index+3 for word, index in keras.datasets.imdb.get_word_index().items()}  # word -> integer dictionary
# The IMDB dataset reserves the 4 first indices for special tokens <PAD>, <START>, <OOV>, <END>
index_to_word = {value: key for key, value in word_to_index.items()}  # integer -> word dictionary
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<OOV>"
index_to_word[3] = "<END>"

"""We show the first reviews and their corresponding sentiment."""

def decode_review(encoded_review: list[int]) -> str:
    """Decode a review from a list of integers to a string."""
    return ' '.join(index_to_word.get(word_index, "<OOV>") for word_index in encoded_review)

print("First reviews in training set, with the corresponding labels:")
for (i, (review, label)) in enumerate(zip(X_train[:5], y_train[:5])):
    print(f"Review {i + 1}: {decode_review(review)}.\nLabel: {label}.")

"""We add padding to the reviews to have the same length. We use the `post` mode to pad and truncate at the end of the reviews."""

# pad the reviews to have the same length (padding and truncating at the end with "post")
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length, padding="post", truncating="post")
X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_review_length, padding="post", truncating="post")
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length, padding="post", truncating="post")


"""**Important**: The previous model has a huge number of trainable parameters (5.4M) due to the embedding layer (even though the embedding and the vocabulary sizes are not very large). It consumes lots of memory, takes a long time to train, and it requires lots of data.

We compile, train, and evaluate the model. We use the Adam optimizer and the binary cross-entropy loss function. We use early stopping to avoid overfitting.

We save the model to avoid retraining it every time we run the notebook.
"""


elmo = hub.load("https://tfhub.dev/google/elmo/3")

def get_elmo_embeddings(sentences: list[str]) -> np.array:
    """
    Get ELMo embeddings for a list of sentences.
    """
    # ELMo returns a tensor, but we want to extract the embeddings
    embeddings = elmo.signatures['default'](tf.constant(sentences))['elmo']
    return embeddings.numpy()  # Convert to numpy array for easier manipulation

"""We have to generate the ELMo embeddings from the reviews. We use the `get_elmo_embeddings` function to get the embeddings for the training, validation, and test sets. This might take time and consume a lot of memory resources (change the value of `max_sentences_train`, `vocabulary_size` and `max_review_size` if you get an out-of-memory error)."""

#X_train_elmo = get_elmo_embeddings([decode_review(review) for review in X_train])
X_val_elmo = get_elmo_embeddings([decode_review(review) for review in X_val])
X_test_elmo = get_elmo_embeddings([decode_review(review) for review in X_test])



def generate_data_lazy(X_train_p: np.array, y_train_p: np.array, batch_size: int, n_epochs_p:int) -> np.array:
    """
    Generate training data in a lazy way, batch after batch, to avoid memory issues.
    In this way, all the data is not loaded into memory at once.
    :param X_train_p: The original training data with original shape (max_sentences_train, max_review_length)
    :param y_train_p: The original training labels with original shape (max_sentences_train,)
    :param batch_size: The batch size
    :param n_epochs_p: The number of epochs
    :return: Each batch of data, with ELMo embeddings of shape (batch_size, max_review_length, embedding_dim)
    """
    for epoch in range(n_epochs_p):
        to_index = 0
        for batch_number in range(X_train_p.shape[0] // batch_size):
            from_index, to_index = batch_number*batch_size, (batch_number+1)*batch_size
            X_train_elmo_batch = get_elmo_embeddings([decode_review(review) for review in X_train_p[from_index:to_index]])
            yield X_train_elmo_batch, np.array(y_train_p[from_index:to_index])
        X_train_elmo_batch = get_elmo_embeddings([decode_review(review) for review in X_train_p[to_index:]])
        yield X_train_elmo_batch, np.array(y_train_p[to_index:])


def compile_train_evaluate(model_p: keras.Model, X_train_p: np.array, y_train_p: np.array,
                           x_val_p: np.array, y_val_p: np.array, x_test_p: np.array, y_test_p: np.array,
                           batch_size: int, epochs: int) -> (float, float, keras.Model):
    """
    Compile, train and evaluate the model.
    :param model_p: the model to compile, train and evaluate
    :param X_train_p: train X sequences
    :param y_train_p: train y labels
    :param x_val_p: validation X sequences
    :param y_val_p: validation y labels
    :param x_test_p: test X sequences
    :param y_test_p: test y labels
    :param batch_size: batch size
    :param epochs: number of epochs
    :return: (test_loss, test_accuracy, model)
    """
    # we compile and train the model
    model_p.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    model_p.fit(generate_data_lazy(X_train_p, y_train_p, batch_size, epochs),
                batch_size=batch_size, epochs=epochs,
                steps_per_epoch=X_train_p.shape[0] // batch_size + 1,
                validation_data=(x_val_p, y_val_p),
                callbacks=[early_stopping_callback])
    # Evaluate the model on the test set
    loss, accuracy = model_p.evaluate(x_test_p, y_test_p)
    return loss, accuracy, model_p




"""
## ELMo embeddings

In the following RNN, we use [ELMo embeddings](https://en.wikipedia.org/wiki/ELMo). ELMo embeddings are context-dependent embeddings, pretrained and ready to use. ELMo embeddins have 1024 dimensions. BERT, and GPT are more powerful embeddings, but require more memory an CPU resources. Nowadays, the most powerful embeddings (e.g., `SentencePiece`) are for subword units rather than words; we use word embeddings for simplicity.
"""


"""We have to generate the ELMo embeddings from the reviews. We use the `get_elmo_embeddings` function to get the embeddings for the training, validation, and test sets. This might take time and consume a lot of memory resources (change the value of `max_sentences_train`, `vocabulary_size` and `max_review_size` if you get an out-of-memory error)."""


"""Now, let's create a new RNN. *Notice* that we do not need an embedding layer because we are using ELMo embeddings as an input. This makes the model to have fewer parameters. We can use a more complex model with fewer parameters.

**Important**: the first RNN layer uses *dropout*. We also define a dropout layer after the LSTM layer. Dropout is a regularization technique that helps to avoid overfitting. I works the following way. In each iteration, some neurons are randomly set to zero. This forces the network to learn more robust features. Dropout is only used during training, not during inference. Dropout is a powerful technique to avoid overfitting in deep learning, but it slows down the training process.
"""

inputs = keras.Input(shape=(None, embedding_dim), dtype="float32")
# Add 2 LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(inputs)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Dropout layer with 20% rate
x = layers.Dropout(0.2)(x)
# Add a classifier (sigmoid activation function for binary classification)
outputs = layers.Dense(1, activation="sigmoid")(x)
elmo_model = keras.Model(inputs, outputs)
elmo_model.summary()

"""**Important**: This model has 656K trainable parameters, which is much less than the previous models (11.5% the parameters of the previous RNN network). It consumes less memory, takes less time to train, and requires fewer data.

We compile, train, and evaluate the model.
"""

test_loss, test_accuracy, elmo_model = compile_train_evaluate(elmo_model, X_train, y_train, X_val_elmo, y_val, X_test_elmo, y_test,
                                                              32, n_epochs)
print(f"Test loss: {test_loss:.4f}.\nTest accuracy: {test_accuracy:.4f}.")

"""## Inference

We can see how accurate the mode is by predicting the sentiment of some reviews. What follows are some example reviews. Add more reviews to test the model.
"""

example_reviews = ["The movie was a great waste of time. I is awful and boring.",
                   "I loved the movie. The plot was amazing.",
                   "This movie is not worth watching.",
                   "Although the film is not a masterpiece, you may have a good time if your expectations are not high."]


review_embeddings = get_elmo_embeddings(example_reviews)
predictions = elmo_model.predict(review_embeddings)

for i, prediction in enumerate(predictions):
    print(f"Review {i + 1}: {example_reviews[i]}")
    print(f"Probability of being positive: {prediction[0]:.4f}.\n")

