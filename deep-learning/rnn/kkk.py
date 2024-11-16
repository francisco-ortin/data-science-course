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
# TODO: delete the next line
#max_review_length = 20

# pad the reviews to have the same length
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length, padding="post")  # pads at the end (post)
X_val = keras.preprocessing.sequence.pad_sequences(X_val, maxlen=max_review_length, padding="post")
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length, padding="post")

print("First review in training set, after padding:", decode_review(X_train[0]))




## ELMO MODEL ########

elmo_model = hub.load("https://tfhub.dev/google/elmo/3")


def word_to_embedding(word: str) -> np.ndarray:
    word_embeddings = elmo_model.signatures['default'](tf.constant([word]))['elmo']
    return word_embeddings[0][0]

def word_index_to_elmo_embedding(word_index: int) -> np.ndarray:
    word = word_index_to_word(word_index)
    word_embeddings = elmo_model.signatures['default'](tf.constant([word]))['elmo']
    return word_embeddings[0][0]

def create_elmo_embedding_dataset(X_p: np.array, max_review_length_p: int, embedding_size: int, file_name: str) -> np.array:
    """
    Create an ELMO embedding dataset
    :param X_p: the original dataset
    :param max_review_length_p: the maximum length of the reviews
    :param embedding_size: the size of the ELMO embeddings
    :return: the ELMO embedding dataset created
    """
    if not os.path.exists(file_name):
        # the file does not exist, so we create the dataset
        X_elmo_loc = np.zeros((len(X_p), max_review_length_p, embedding_size))
        for seq_idx, sequence in enumerate(tqdm(X_p)):
            for word_idx, word_key in enumerate(sequence):
                embedding = word_index_to_elmo_embedding(word_key)
                X_elmo_loc[seq_idx][word_idx] = embedding
        # save the dataset
        np.save(file_name, X_elmo_loc)
        return X_elmo_loc
    else:
        # the file exists, so we load the dataset
        return np.load(file_name)


print("Creating ELMO embeddings for X_train...")
X_train_elmo = create_elmo_embedding_dataset(X_train, max_review_length, 1024, "data/X_train_elmo.npy")
print("Creating ELMO embeddings for X_val...")
X_val_elmo = create_elmo_embedding_dataset(X_val, max_review_length, 1024, "data/X_val_elmo.npy")
print("Creating ELMO embeddings for X_test...")
X_test_elmo = create_elmo_embedding_dataset(X_test, max_review_length, 1024, "data/X_test_elmo.npy")



# Each review is a sequence of `max_review_length` ELMO word embeddings (each of size 1024)
# we set the first dimension to None (instead of `max_review_length`) to allow for variable length sequences
inputs = keras.Input(shape=(None, 1024), dtype="int32")  # ELMO expects strings as input, not integers
# elmo with lambda layer
#elmo_layer = Lambda(word_index_to_embedding, output_shape=(max_review_length, 1024))(inputs)
# Add 2 Bidirectional LSTM layers
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
# Define the model
elmo_bi_lstm_model = keras.Model(inputs, outputs)
elmo_bi_lstm_model.summary()


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


test_loss, test_accuracy = compile_train_evaluate(elmo_bi_lstm_model, X_train_elmo, y_train, X_val_elmo, y_val, X_test_elmo, y_test,
                                                  32, 100, "data/elmo_bi_lstm_model.keras")
print(f"Test loss: {test_loss:.4f}.\nTest accuracy: {test_accuracy:.4f}.")
