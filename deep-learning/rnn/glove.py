# Importing necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

# Load IMDb dataset
max_words = 10000  # Number of most frequent words to consider
max_len = 200  # Maximum length of review (padded sequences)

print("Loading IMDb data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Pad the sequences to ensure uniform input size
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Load GloVe embeddings
embedding_dim = 100  # GloVe word vectors of 100 dimensions

# Download GloVe embeddings if not already available
#!wget -nc http://nlp.stanford.edu/data/glove.6B.zip
#!unzip -n glove.6B.zip

# Load GloVe embeddings into a dictionary
embeddings_index = {}
with open('glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = vector

print(f"Found {len(embeddings_index)} word vectors.")

# Prepare embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for i in range(1, max_words):
    word = imdb.get_word_index()[i-3]  # Mapping from word index to actual word
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim,
                    weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(SimpleRNN(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping callback to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# Train the model
model.summary()

history = model.fit(x_train, y_train, epochs=5, batch_size=64,
                    validation_data=(x_test, y_test), callbacks=[early_stop])

# Evaluate the model
score = model.evaluate(x_test, y_test, batch_size=64)
print(f"Test accuracy: {score[1] * 100:.2f}%")

# Save the model
model.save('sentiment_rnn_glove.h5')
