# https://github.com/ageron/handson-ml3/blob/main/14_deep_computer_vision_with_cnns.ipynb

# We now see a common architecture followed in Convolutional Neural Networks (CNNs):
# - Different convolutional layers, using ReLU as the activation function.
# - Followed by pooling layers to reduce its memory requirements.
# - To perform the classification, the output of the CNN is passed to a dense layers.

# <img src="img/cnns.png" width="600px"/>

# It is very common that the size of the images / feature maps get smaller and smaller.
# Similarly, feature maps get deeper and deeper.

import tensorflow as tf
import numpy as np

# DATA PREPARATION

# We download fashion MNIST the dataset from keras.
mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = mnist
# We change the shape of X_train_full from (60k, 28, 28) to (60k, 28, 28, 1)
# This is because the images have only one channel (gray color), but we want to apply convolutional
# layers the same way as we do for multichannel images (color images)
X_train_full = np.expand_dims(X_train_full, axis=-1)
# We re-scale the gray color to a number between 0 and 1
X_train_full = X_train_full.astype(np.float32) / 255
# We do the same with the test set
X_test = np.expand_dims(X_test.astype(np.float32), axis=-1) / 255

N_VAL_INSTANCES = 10_000
X_train, X_val = X_train_full[:-N_VAL_INSTANCES], X_train_full[-N_VAL_INSTANCES:]
y_train, y_val = y_train_full[:-N_VAL_INSTANCES], y_train_full[-N_VAL_INSTANCES:]

# Show shapes of all the datasets
print(f"Shape of X_train = {X_train.shape} and y_train = {y_train.shape}.")
print(f"Shape of X_val = {X_val.shape} and y_val = {y_val.shape}.")
print(f"Shape of X_test = {X_test.shape} and y_test = {y_test.shape}.")


# CREATION OF THE MODEL

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=[28, 28, 1], kernel_size=7, filters=64, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(kernel_size=3, filters=128, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(kernel_size=3, filters=128, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding="same", activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu", kernel_initializer="he_normal"),
    # Regularization mechanism to reduce overfitting
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    # We have 10 possible outputs (softmax)
    tf.keras.layers.Dense(units=10, activation="softmax")
])

model.summary()

# MODEL TRAINING AND EVALUATION
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
evaluation_results = model.evaluate(X_test, y_test)
print(f"Test loss: {evaluation_results[0]:.4f}. Test accuracy: {evaluation_results[1]:.4f}.")

# Val loss: 0.2739. Val accuracy: 0.9086.
# Test loss: 0.2980. Test accuracy: 0.9012.

# Significantly higher than MLP (with more parameters):
# Test loss: 0.3540. Test accuracy: 0.8714.

# Questions:
# 1) What do you think it is causing this improvement?
# Answer: Mainly, because it has more parameters than MLP.
#         However, an MLP with that number of parameters would not be that performant (CNNs are
#         more efficient for computer vision).

