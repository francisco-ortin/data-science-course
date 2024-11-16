# TODO: Usar word2vec con la misma estructura de la red neuronal recurrente, pero cargando los pesos en la capa de embedding

import keras
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from keras.src.layers import Lambda
from tensorflow.keras import layers
import os
from tqdm import tqdm


vocabulary_size = 20_000


# we train one half for training and the other half for validation and testing
(X_train, y_train), (X_half, y_half) = keras.datasets.imdb.load_data(num_words=vocabulary_size)
# get validation set as half of the test set
X_test, X_val = X_half[:len(X_half) // 2], X_half[len(X_half) // 2:]
y_test, y_val = y_half[:len(y_half) // 2], y_half[len(y_half) // 2:]


print(f"Training sequences: {len(X_train):,}.\nValidation sequences: {len(X_val):,}.\nTesting sequences: {len(X_test):,}.")


## Preprocess the data ########

# Let's print some reviews. We need to convert the integers (token ids) back to words.
word_to_index = keras.datasets.imdb.get_word_index()  # word -> integer dictionary
index_to_word = {value: key for key, value in word_to_index.items()}  # integer -> word dictionary

def word_index_to_word(word_index: int) -> str:
    # The IMDB dataset reserves the 3 first indices for special tokens <PAD>, <START> and <OOV>
    match word_index:
        case 0: return "<PAD>"
        case 1: return "<START>"
        case 2: return "<OOV>"
        case _: return index_to_word.get(word_index - 3, "<OOV>")

def decode_review(encoded_review: list[int]) -> str:
    return ' '.join(word_index_to_word(word_index) for word_index in encoded_review)

print("First reviews in training set, with the corresponding labels:")
for (i, (review, label)) in enumerate(zip(X_train[:5], y_train[:5])):
    print(f"Review {i + 1}: {decode_review(review)}.\nLabel: {label}.")


## PADDING ########

# compute the maximum length of the reviews
max_review_length = max(len(review) for review in X_train)

# pad the reviews to have the same length
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length, padding="post")  # pads at the end (post)
X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_review_length, padding="post")
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length, padding="post")

print("First review in training set, after padding:", decode_review(X_train[0]))



import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the pre-trained Word2Vec model (from Google News)
word2vec_path = 'path_to_your_word2vec_model/GoogleNews-vectors-negative300.bin'  # update the path
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Create a tokenizer to fit on the training data
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(decode_review(review) for review in X_train)  # Ensure the data is preprocessed before tokenizing

# Build the embedding matrix
embedding_dim = 300  # Word2Vec vectors are 300-dimensional
embedding_matrix = np.zeros((vocabulary_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocabulary_size:  # Only consider the most frequent words in the vocabulary
        try:
            embedding_matrix[i] = word2vec_model[word]
        except KeyError:
            embedding_matrix[i] = np.zeros(embedding_dim)  # If word not in Word2Vec model, use zeros





# Define the model with the pre-trained Word2Vec embeddings
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to the Word2Vec vector space using the embedding matrix
x = layers.Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix], trainable=False)(inputs)
# Add 2 Bidirectional-LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
elmo_bi_lstm_model = keras.Model(inputs, outputs)
elmo_bi_lstm_model.summary()


"""inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(vocabulary_size, 128)(inputs)
# Add 2 Bidirectional-LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
elmo_bi_lstm_model = keras.Model(inputs, outputs)
elmo_bi_lstm_model.summary()
"""



def compile_train_evaluate(model_p: keras.Model, x_train_p: np.array, y_train_p: np.array,
                           x_val_p: np.array, y_val_p: np.array, x_test_p: np.array, y_test_p: np.array,
                           batch_size: int, epochs: int, model_file_name: str) -> (float, float):
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
    :return: (test_loss, test_accuracy)
    """
    # we compile and train the model if it does not exist (otherwise we load it)
    if not os.path.exists(model_file_name):
        model_p.compile("adam", "binary_crossentropy", metrics=["accuracy"])
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
        model_p.fit(x_train_p, y_train_p, batch_size=batch_size, epochs=epochs, validation_data=(x_val_p, y_val_p),
                    callbacks=[early_stopping_callback])
        # save the model
        model_p.save(model_file_name)
    else:
        model_p = keras.models.load_model(model_file_name)
    # Evaluate the model on the test set
    return model_p.evaluate(x_test_p, y_test_p)


test_loss, test_accuracy = compile_train_evaluate(elmo_bi_lstm_model, X_train, y_train, X_val, y_val, X_test, y_test,
                                                  32, 100, "data/elmo_bi_lstm_model.keras")
print(f"Test loss: {test_loss:.4f}.\nTest accuracy: {test_accuracy:.4f}.")
