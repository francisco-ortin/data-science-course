# https://arvindn-iitkgp.medium.com/detailed-analysis-on-bi-directional-lstm-on-imdb-dataset-c2bc4cfa2c2c
import os

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub
import zipfile


# In this notebook, we will perform a sentiment analysis task to classify movie reviews as positive or negative.
# By analyzing the text of the reviews, we will predict the sentiment of the review.

# The [IMDb dataset](https://keras.io/api/datasets/imdb/) is a set of 50,000 movie reviews from
# the [Internet Movie Database](https://en.wikipedia.org/wiki/IMDb) (IMDb).

# <img src="https://en.wikipedia.org/wiki/IMDb#/media/File:IMDB_Logo_2016.svg" width="400"/>

## Load the IMDB dataset ########


# consider all the words with a frequency higher than this value
vocabulary_size = 400_000
# compute the maximum length of the reviews (for speeding up the training it is better to cut the reviews)
max_review_length = 80
# max number of epochs to train the models (we use early stopping)
n_epochs = 50
# Embedding dimensions
embedding_dim = 50


# we train one half for training and the other half for validation and testing
(X_train, y_train), (x_half, y_half) = keras.datasets.imdb.load_data(num_words=vocabulary_size)
X_train = X_train[:500]
y_train = y_train[:500]
x_half = x_half[:100]
y_half = y_half[:100]
# get validation set as half of the test set
X_test, X_val = x_half[:len(x_half) // 2], x_half[len(x_half) // 2:]
y_test, y_val = y_half[:len(y_half) // 2], y_half[len(y_half) // 2:]

print(f"Training sequences: {len(X_train):,}.\nValidation sequences: {len(X_val):,}.\nTesting sequences: {len(X_test):,}.")


## Preprocess the data ########

# Let's print some reviews. We need to convert the integers (token ids) back to words.
word_to_index = {word: index+3 for word, index in keras.datasets.imdb.get_word_index().items()}  # word -> integer dictionary
# The IMDB dataset reserves the 4 first indices for special tokens <PAD>, <START>, <OOV>, <END>
index_to_word = {value: key for key, value in word_to_index.items()}  # integer -> word dictionary
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<OOV>"
index_to_word[3] = "<END>"


def decode_review(encoded_review: list[int]) -> str:
    return ' '.join(index_to_word.get(word_index, "<OOV>") for word_index in encoded_review)

print("First reviews in training set, with the corresponding labels:")
for (i, (review, label)) in enumerate(zip(X_train[:5], y_train[:5])):
    print(f"Review {i + 1}: {decode_review(review)}.\nLabel: {label}.")


## PADDING ########


# pad the reviews to have the same length (padding and truncating at the end with "post")
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length, padding="post", truncating="post")
X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_review_length, padding="post", truncating="post")
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length, padding="post", truncating="post")

"""print("First reviews in training set, after padding:")
for (i, (review, label)) in enumerate(zip(X_train[:5], y_train[:5])):
    print(f"Review {i + 1}: {decode_review(review)}.\nLabel: {label}.")
"""

## LSTM MODEL ########

# We create a model where embeddings are computed as the first layer using the Keras `Embedding` layer.
# We have to define the size of the embedding vectors (hyperparameter).

# variable length input integer sequences
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(vocabulary_size, embedding_dim)(inputs)
# Add 2 LSTM layers
#x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
# Add a classifier (sigmoid activation function for binary classification)
outputs = layers.Dense(1, activation="sigmoid")(x)
one_directional_model = keras.Model(inputs, outputs)
one_directional_model.summary()


def compile_train_evaluate(model_p: keras.Model, x_train_p: np.array, y_train_p: np.array,
                           x_val_p: np.array, y_val_p: np.array, x_test_p: np.array, y_test_p: np.array,
                           batch_size: int, epochs: int, zip_file_name: str, model_file_name: str) -> (float, float, keras.Model):
    """
    Compile, train and evaluate the model.
    :param model_p: the model to compile, train and evaluate
    :param x_train_p: train X sequences
    :param y_train_p: train y labels
    :param x_val_p: validation X sequences
    :param y_val_p: validation y labels
    :param x_test_p: test X sequences
    :param y_test_p: test y labels
    :param batch_size: batch size
    :param epochs: number of epochs
    :param model_file_name: file name to store/load the model
    :return: (test_loss, test_accuracy, model)
    """
    # we compile and train the model if it does not exist (otherwise we load it)
    if not os.path.exists(zip_file_name):
        model_p.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        model_p.fit(x_train_p, y_train_p, batch_size=batch_size, epochs=epochs, validation_data=(x_val_p, y_val_p),
                    callbacks=[early_stopping_callback])
        # save the model
        model_p.save(model_file_name)
        # compress the model file
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(model_file_name, arcname=model_file_name)
        # remove the model file
        os.remove(model_file_name)
    else:
        # load the model; open the zip file and extract the model file
        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(".")
        # load the model
        model_p = keras.models.load_model(model_file_name)
        # remove the model file
        os.remove(model_file_name)
    # Evaluate the model on the test set
    loss, accuracy = model_p.evaluate(x_test_p, y_test_p)
    return loss, accuracy, model_p


test_loss, test_accuracy, one_directional_model = compile_train_evaluate(one_directional_model, X_train, y_train, X_val, y_val, X_test, y_test,
                       32, n_epochs, "data/one_directional_model.zip", 'one_directional_model.keras')
print(f"Test loss: {test_loss:.4f}.\nTest accuracy: {test_accuracy:.4f}.")

## BIDIRECTIONAL MODEL ########

# We create a model where embeddings are computed as the first layer using the Keras `Embedding` layer.
# We have to define the size of the embedding vectors (hyperparameter).

# variable length input integer sequences
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(vocabulary_size, embedding_dim)(inputs)
# Add 2 Bidirectional-LSTM layers
#x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
bi_lstm_model = keras.Model(inputs, outputs)
bi_lstm_model.summary()


test_loss, test_accuracy, bi_lstm_model = compile_train_evaluate(bi_lstm_model, X_train, y_train, X_val, y_val, X_test, y_test,
                                                  32, n_epochs, "data/bi_directional_model.zip", "bi_directional_model.keras")
print(f"Test loss: {test_loss:.4f}.\nTest accuracy: {test_accuracy:.4f}.")



## USING GLOVE EMBEDDINGS ########


def create_glove_embeddings_from_file(zip_file_name: str, txt_file_name: str) -> dict[str, np.array]:
    """
    Create a dictionary of GloVe embeddings from a file.
    :param zip_file_name: the zip file name
    :param txt_file_name: the text file name inside the zip file
    :return: the dictionary of embeddings
    """
    glove_embeddings_loc = {}  # word -> vector(embedding_dim) mapping
    with zipfile.ZipFile(zip_file_name, 'r') as zip_file:
        with zip_file.open(txt_file_name, 'r') as file:
            # load the vocabulary_size most frequent words, including padding, start and OOV tokens
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                glove_embeddings_loc[word] = vector
            glove_embeddings_loc["<PAD>"] = np.zeros(embedding_dim)
            glove_embeddings_loc["<START>"] = np.full(embedding_dim, 0.5)
            glove_embeddings_loc["<OOV>"] = np.ones(embedding_dim)
    return glove_embeddings_loc


glove_embeddings = create_glove_embeddings_from_file("data/glove.6B.50d.zip", "glove.6B.50d.txt")
print(f"Found {len(glove_embeddings):,} word embeddings of {embedding_dim} dimensions.")


def get_glove_word_embedding(word_p: str, glove_embeddings_p: dict[str, np.array]) -> np.array:
    """
    Get the GloVe embedding for a word. It is not found, return the embedding for the OOV token.
    """
    return glove_embeddings.get(word_p, glove_embeddings_p["<OOV>"])


glove_embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
for word_index in range(vocabulary_size):
    word = index_to_word.get(word_index, "<OOV>")
    embedding_vector = get_glove_word_embedding(word, glove_embeddings)
    glove_embedding_matrix[word_index] = embedding_vector



inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(vocabulary_size, embedding_dim, weights=[glove_embedding_matrix], trainable=False)(inputs)
#x = layers.Embedding(vocabulary_size, embedding_dim)(inputs)
# Add 2 Bidirectional-LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
#x = layers.LSTM(32)(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
glove_lstm_model = keras.Model(inputs, outputs)
glove_lstm_model.summary()

# 0.86 en val y test para max_review_length = 160

test_loss, test_accuracy, glove_lstm_model = compile_train_evaluate(glove_lstm_model, X_train, y_train, X_val, y_val, X_test, y_test,
                                                 32, n_epochs, "data/glove_model.zip", "glove_model.keras")
print(f"Test loss: {test_loss:.4f}.\nTest accuracy: {test_accuracy:.4f}.")

## INFERENCE ########

# Some example reviews you can modify to test the model

example_reviews = ["The movie was a great waste of time. The plot was boring.",
                   "I loved the movie. The plot was amazing.",
                   "Would not recommend this movie to anyone."]


def prepare_reviews_for_prediction(reviews_p: list[str], word_to_index_p: dict[str, int], max_review_length_p: int)\
        -> np.array:
    """
    Prepare a list of reviews for prediction: include the <START> token, convert the words to lower case,
    remove punctuation, convert the words to indexes, pad the sequences and truncate them if necessary.
    :param reviews_p: the list of reviews to be prepared
    :param word_to_index_p: a dictionary mapping words to indexes
    :param max_review_length_p: the maximum length of the reviews
    :return: and array of token sequences, one for each review
    """
    sequences_loc = []
    for review in reviews_p:
        words_indexes = [1]  # start token
        for word in review.split():
            for char_to_remove in [".", ",", "!", "?"]:
                word = word.lower().replace(char_to_remove, "")
            words_indexes.append(word_to_index_p[word] if word in word_to_index_p else 2)  # OOV token
        sequences_loc.append(words_indexes)
    # pad the sequences
    sequences_loc = keras.preprocessing.sequence.pad_sequences(sequences_loc, maxlen=max_review_length_p,
                                                               padding="post", truncating="post")
    return sequences_loc


sequences_to_predict = prepare_reviews_for_prediction(example_reviews, word_to_index, max_review_length)
predictions = glove_lstm_model.predict(sequences_to_predict)

for i, prediction in enumerate(predictions):
    print(f"Review {i + 1}: {example_reviews[i]}")
    print(f"Probability of being positive: {prediction[0]:.4f}.\n")

