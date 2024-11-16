# https://arvindn-iitkgp.medium.com/detailed-analysis-on-bi-directional-lstm-on-imdb-dataset-c2bc4cfa2c2c

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# consider top 20000 words only
max_features = 20000
# consider 200 words per review only
maxlen = 200


(x_train, y_train),(x_val,y_val) = keras.datasets.imdb.load_data(num_words=max_features)
print(len(x_train),"Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val,maxlen=maxlen)


# variable length input integer sequences
inputs = keras.Input(shape = (None,),dtype="int32")
# Embed each integer to 128 dimensional vector space
x = layers.Embedding(max_features, 128)(inputs)
# Add 2 Bi-LSTMS
x = layers.Bidirectional(layers.LSTM(64,return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1,activation="sigmoid")(x)
model = keras.Model(inputs,outputs)
model.summary()


model.compile("adam","binary_crossentropy",metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32,epochs=2, validation_data=(x_val,y_val))