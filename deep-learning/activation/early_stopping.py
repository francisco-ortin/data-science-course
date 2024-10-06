# We apply a stop criterion to avoid overfitting

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# DATA PREPARATION


(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

N_VAL_INSTANCES = 10_000

X_train, y_train = X_train_full[:-N_VAL_INSTANCES], y_train_full[:-N_VAL_INSTANCES]
X_val, y_val = X_train_full[-N_VAL_INSTANCES:], y_train_full[-N_VAL_INSTANCES:]

print(f"Shape of X_train = {X_train.shape} and y_train = {y_train.shape}.")
print(f"Shape of X_val = {X_val.shape} and y_val = {y_val.shape}.")
print(f"Shape of X_test = {X_test.shape} and y_test = {y_test.shape}.")

CLASS_LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# We rescale the colors to real numbers between 0 and 1
X_train, X_val, X_test = X_train / 255, X_val / 255, X_test / 255

# CREATION OF THE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

# COMPILATION OF THE MODEL

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# TRAINING THE MODEL WITH OVERFITTING

EPOCHS = 20
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val))

# Let's take a look at how test and train loss and accuracy evolved during training
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, EPOCHS], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])  # red/blue, dashed(--), continuous (-), shape(. or *)
plt.show()

# TRAINING THE MODEL WITH EARLY STOP

# Recompile the model to forget the learned weights
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# We create a callback that stops when the val loss grows after "patience" epochs is not improving.
# restore_best_weights = True makes the model to choose the one with the best weights (forget the last ones)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history_early_stop = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val),
                               callbacks=[early_stopping_callback])

# Let's take a look at how test and train loss and accuracy evolved during training
pd.DataFrame(history_early_stop.history).plot(
    figsize=(8, 5), xlim=[0, EPOCHS], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])  # red/blue, dashed(--), continuous (-), shape(. or *)
plt.show()

# Questions:
# 1) What has happened?
# Answer: The after two consecutive increases of val loss the system stops training.
# 2) Identify the two main benefits?
# Answer: a) we do not keep training when it is no longer necessary (CPU saving)
#         b) we get the model with the best val loss (not if we set a fixed number of epochs)
