import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import tensorflow_hub as hub

# Load IMDB dataset
max_words = 10000  # Limit the vocabulary size
max_len = 100  # Maximum review length (in words)

# Load IMDB data (training and testing)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# Pad sequences to ensure uniform length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Load ELMo embedding from TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Convert words into ELMo embeddings
def elmo_embedding(x):
    # Convert input sequence of word indices into corresponding word vectors using ELMo
    words = tf.gather(tf.constant(imdb.get_word_index()), x) 
    embeddings = elmo(words)
    return embeddings

# Create a Sequential model
model = Sequential()

# Add ELMo embeddings as a layer (we will use `ELMO` embeddings directly from TensorFlow Hub)
model.add(Embedding(input_dim=max_words, output_dim=256, input_length=max_len))

# Adding LSTM layer
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Adding fully connected (Dense) layers
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# Save the trained model
model.save("sentiment_model.h5")
