# https://arvindn-iitkgp.medium.com/detailed-analysis-on-bi-directional-lstm-on-imdb-dataset-c2bc4cfa2c2c

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_hub as hub

## Load the IMDB dataset ########

# consider top 20,000 words only
vocabulary_size = 20_000


# we train one half for training and the other half for validation and testing
(x_train, y_train), (x_half, y_half) = keras.datasets.imdb.load_data(num_words=vocabulary_size)
# get validation set as half of the test set
x_test, x_val = x_half[:len(x_half) // 2], x_half[len(x_half) // 2:]
y_test, y_val = y_half[:len(y_half) // 2], y_half[len(y_half) // 2:]

print(f"Training sequences: {len(x_train):,}.\nValidation sequences: {len(x_val):,}.\nTesting sequences: {len(x_test):,}.")


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
for (i, (review, label)) in enumerate(zip(x_train[:5], y_train[:5])):
    print(f"Review {i + 1}: {decode_review(review)}.\nLabel: {label}.")


## PADDING ########

# compute the maximum length of the reviews
max_review_length = max(len(review) for review in x_train)

# pad the reviews to have the same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_length, padding="post")  # pads at the end (post)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_review_length, padding="post")
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_length, padding="post")

print("First reviews in training set, after padding:")
for i, review in enumerate(x_train[:5]):
    print(f"Review {i + 1}: {decode_review(review)}.")


## LSTM MODEL ########

# We create a model where embeddings are computed as the first layer using the Keras `Embedding` layer.
# We have to define the size of the embedding vectors (hyperparameter).

# variable length input integer sequences
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(vocabulary_size, 128)(inputs)
# Add 2 LSTM layers
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.LSTM(64)(x)
# Add a classifier (sigmoid activation function for binary classification)
outputs = layers.Dense(1, activation="sigmoid")(x)
one_directional_model = keras.Model(inputs, outputs)
one_directional_model.summary()


def compile_train_evaluate(model_p: keras.Model, x_train_p: np.array, y_train_p: np.array,
                           x_val_p: np.array, y_val_p: np.array, x_test_p: np.array, y_test_p: np.array,
                           batch_size: int, epochs: int, model_name: str) -> (float, float):
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
    :param model_name: model name to be stored
    :return: (test_loss, test_accuracy)
    """
    model_p.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    model_p.fit(x_train_p, y_train_p, batch_size=batch_size, epochs=epochs, validation_data=(x_val_p, y_val_p),
                callbacks=[early_stopping_callback])
    # save the model
    model_p.save(f"data/{model_name}.keras")
    # Evaluate the model on the test set
    test_loss, test_accuracy = model_p.evaluate(x_test_p, y_test_p)
    return test_loss, test_accuracy


compile_train_evaluate(one_directional_model, x_train, y_train, x_val, y_val, x_test, y_test,
                       32, 100, "one_directional_model")

## BIDIRECTIONAL MODEL ########

# We create a model where embeddings are computed as the first layer using the Keras `Embedding` layer.
# We have to define the size of the embedding vectors (hyperparameter).

# variable length input integer sequences
inputs = keras.Input(shape=(None,), dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(vocabulary_size, 128)(inputs)
# Add 2 Bidirectional-LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
elmo_bi_lstm_model = keras.Model(inputs, outputs)
elmo_bi_lstm_model.summary()


compile_train_evaluate(elmo_bi_lstm_model, x_train, y_train, x_val, y_val, x_test, y_test,
                       32, 100, "bi_directional_model")


## USING ELMO EMBEDDINGS ########

"""# Elmo requires the input to be strings, not integers. We need to convert the integer sequences to strings.
# we use our function `decode_review` to convert the integer sequences to strings.
x_test_strings = [decode_review(review) for review in x_test]
x_val_strings = [decode_review(review) for review in x_val]
x_train_strings = [decode_review(review) for review in x_train]




# Load the ELMO model from TensorFlow Hub
elmo_layer = hub.KerasLayer("https://tfhub.dev/google/elmo/3", trainable=False, signature='tokens',
                            output_key='elmo')

# Define the model with ELMO
inputs = keras.Input(shape=(None,), dtype=tf.string)  # ELMO expects strings as input, not integers
# Use the ELMO layer
x = elmo_layer(inputs)  # This outputs a 3D tensor of shape [batch_size, sequence_length, 1024]
# Add 2 Bidirectional LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
# Define the model
elmo_bi_lstm_model = keras.Model(inputs, outputs)
elmo_bi_lstm_model.summary()


compile_train_evaluate(elmo_bi_lstm_model, x_train_strings, y_train, x_val_strings, y_val, x_test_strings, y_test,
                       32, 2, "bi_directional_model")
"""
