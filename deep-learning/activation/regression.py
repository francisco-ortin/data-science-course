# https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb
import keras
import numpy as np
import pandas as pd
# We build a regression model using an MLP.
# We use the california housing dataset (20,640 instances, 8 features, 1 real target).

import tensorflow as tf
from keras.src.layers import Dense
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# DATA PREPARATION

# We download the dataset from sklearn. Split in train, val and test.
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full)  # 75% train, 25% val

# Show shapes of all the datasets
print(f"Feature names: {housing.feature_names}.")
print(f"Target name: '{housing.target_names[0]}'.")
print(f"Shape of X_train = {X_train.shape} and y_train = {y_train.shape}.")
print(f"Shape of X_val = {X_val.shape} and y_val = {y_val.shape}.")
print(f"Shape of X_test = {X_test.shape} and y_test = {y_test.shape}.")

# Input data has to be rescaled. We can use sklearn's StandardScaler or TF's Normalization
norm_layer = tf.keras.layers.Normalization(input_shape=X_train.shape[1:])
norm_layer.adapt(X_train)

# CREATION OF THE MODEL USING THE SEQUENTIAL APPROACH
# ANN creation using the sequential approach
model = tf.keras.Sequential([
    norm_layer,
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(50, activation="relu"),
    tf.keras.layers.Dense(1)  # no activation function (regression)
])
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
# Model training
EPOCHS = 20  # 20
history = model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(X_val, y_val))
# Show training and val errors
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[1, EPOCHS + 1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])  # red/blue, dashed(--), continuous (-), shape(. or *)
plt.show()
# Model evaluation
mse_test, rmse_test = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse_test:.4f}. Test RMSE: {rmse_test:.4f}")

def show_predictions(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    # Add a diagonal line for reference
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',
             label='Perfect Prediction')
    # Add legend
    plt.legend()
    # Show plot
    plt.show()

show_predictions(model, X_test, y_test)

# CREATION OF THE MODEL USING THE FUNCTIONAL APPROACH

# The sequential approach to create ANN is easy but not very flexible.
# The functional approach allows multiple inputs and outputs, layer sharing and branching structures.
# As an example, we will create the following *Wide & Deep* ANN for regression:
# <img src="img/wide-and-deep.png" width="600px"/>
# This architecture makes it possible for the neural network to learn both
# deep patterns (using the deep path) and simple rules (through the short path).

# Input layer
input_layer = tf.keras.layers.Input(shape=X_train.shape[1:])
# Normalized layer is created and the input_layer is passed as an input, returning normalized as output
normalized = (normalization_layer := tf.keras.layers.Normalization())(input_layer)
hidden1 = Dense(50, activation="relu")(normalized)
hidden2 = Dense(50, activation="relu")(hidden1)
hidden3 = Dense(50, activation="relu")(hidden2)
# The normalized input (wide) plus the (deep) output of the MLP are concatenated
concat = tf.keras.layers.Concatenate()([normalized, hidden3])
# Output layer
output = Dense(1)(concat)
# The model is created by specifying the inputs and outputs (there could be many)
model = tf.keras.Model(inputs=[input_layer], outputs=[output])
model.summary()

# Model training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val))
# Show training and val errors
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[1, EPOCHS + 1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])  # red/blue, dashed(--), continuous (-), shape(. or *)
plt.show()
# Model evaluation
mse_test, rmse_test = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse_test:.4f}. Test RMSE: {rmse_test:.4f}")

show_predictions(model, X_test, y_test)


# CREATION OF THE MODEL USING THE SUBCLASSING API APPROACH

# We create a class that derives from `tf.keras.Model` or `tf.keras.Layer`,
# creates the model/layer in its constructor and performs transforms the inputs in outputs
# as described in the `call`method.

# The Subclassing API enables dynamic behavior within the model's call method.
# This allows you to perform dynamic computations, control flow operations, and implement complex
# logic based on input data or model state.
# Moreover, custom layers can be defined, improving re-utilization.

class WideAndDeepModel(tf.keras.Model):
    def __init__(self, units, n_hidden_layers, activation, **kwargs):
        super().__init__(**kwargs)  # needed to support naming the model
        self.norm_layer = tf.keras.layers.Normalization()
        self.hidden_layers = [Dense(units, activation=activation) for _ in range(n_hidden_layers)]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input, **kwargs):
        hidden = (normalized := self.norm_layer(input))
        for hidden_layer in self.hidden_layers:
            hidden = hidden_layer(hidden)
        concatenation = tf.keras.layers.concatenate([normalized, hidden])
        return self.output_layer(concatenation)

model = WideAndDeepModel(units=50, n_hidden_layers=3, activation="relu")

# Model training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss="mse", optimizer=optimizer, metrics=["RootMeanSquaredError"])
model.norm_layer.adapt(X_train)
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val))
# Show training and val errors
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[1, EPOCHS + 1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])  # red/blue, dashed(--), continuous (-), shape(. or *)
plt.show()
# Model evaluation
mse_test, rmse_test = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse_test:.4f}. Test RMSE: {rmse_test:.4f}")

show_predictions(model, X_test, y_test)


