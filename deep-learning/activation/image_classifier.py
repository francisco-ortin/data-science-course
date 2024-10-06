# https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb

# Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples 
# and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a 
# label from 10 classes. Each class represents an article of clothing 
#(0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot).

# img/fashion-minst.webp

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import pandas as pd

# DATA PREPARATION

# We download fashion MNIST the dataset from keras.
# It is already shuffled and split into a training set (60,000 images) and a test set (10,000 images)

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# We take the last 10,000 images from the training set for validation and the rest of them for training
N_VAL_INSTANCES = 10_000

X_train, y_train = X_train_full[:-N_VAL_INSTANCES], y_train_full[:-N_VAL_INSTANCES]
X_val, y_val = X_train_full[-N_VAL_INSTANCES:], y_train_full[-N_VAL_INSTANCES:]

# We show the first image and its class label
CLASS_LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
plt.imshow(X_train[0], cmap="binary")
plt.title(CLASS_LABELS[y_train[0]])
plt.axis('off')
plt.show()

# Show shapes of all the datasets
print(f"Shape of X_train = {X_train.shape} and y_train = {y_train.shape}.")
print(f"Shape of X_val = {X_val.shape} and y_val = {y_val.shape}.")
print(f"Shape of X_test = {X_test.shape} and y_test = {y_test.shape}.")

# X holds gray colors from 0 to 255 (1 byte) for each pixel
# y holds the type of clothing: an int from 0 to 9
print(f"Values of X_train (gray colors): {np.unique(X_train)}")
print(f"Values of Y_train (types of clothes): {np.unique(y_train)}")

# We rescale the colors to real numbers between 0 and 1
X_train, X_val, X_test = X_train / 255, X_val / 255, X_test / 255
print(f"Values of X_train (gray colors): {np.unique(X_train).round(decimals=3)}")

# CREATION OF THE MODEL
# ANN creation using the sequential approach
model = tf.keras.Sequential()
# The Input layer *just* specifies the size of the input (i.e., each individual)
# Thus, the input *layer* is not actually a layer
model.add(tf.keras.layers.InputLayer(input_shape=[28, 28]))
# Since each individual is 2D, we use a Flatten layer to convert it to a 1D vector
model.add(tf.keras.layers.Flatten())
# First hidden layer with 300 neurons and relu activation function
model.add(tf.keras.layers.Dense(300, activation="relu"))
# Second hidden layer with 100 neurons and relu activation function
model.add(tf.keras.layers.Dense(100, activation="relu"))
# Last layer with 10 outputs and softmax activation function, since we have a multiclass classification problem
model.add(tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax"))


tf.keras.backend.clear_session()  # Just resets the internal name of the layers (not necessary)

# The previous sequential construction could be just performed with a single invocation (more common).
# Notice that, with this syntax, the Input *layer* is not passed. On the contrary, the first
# actual layer receives an "input_shape" parameter.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax")
])

# We print the ANN with model.summary()
model.summary()

# 266K parameters! It is a very powerful model. You can see how Dense layers are so expensive.
# You need lots of data to train this model. Otherwise, it could overfit.

# A graphical view of the ANN
# The first "None" value in the shape represents that we do not know the size of the minibatch yet
tf.keras.utils.plot_model(model, to_file='img/MLP-topology.png', show_shapes=True, show_layer_names=True)
plt.imshow(mpimg.imread('img/MLP-topology.png'))
plt.axis('off')
plt.show()

# Let's show some information about the ANN's layers
for i, layer in enumerate(model.layers):
    print(f"Layer number {i}, name '{layer.name}', and class {layer.__class__}")

# We can also show the parameters (W and b) of the last layer
weights, biases = model.layers[3].get_weights()
print("Shape of the weights of the last layer:", weights.shape)
print("Values of the weights of the first row in the last layer:", weights[0].round(decimals=3))
print("Biases of the last layer:", biases)

# Weights are initialized randomly to help backprop know the influence of each neuron in the cost
# Multiple initializers can be used in keras; all of them set random values.
# Depending on the activation function, some of them are more appropriate than others:
# - Glorot/Xavier initializer for linear, tanh, sigmoid and softmax activation functions
# - He initializer for ReLU, Leaky ReLU, ELU, GELU, Swish and Mish activation functions
# - LeCun for SELU activation function

# COMPILATION OF THE MODEL

# Once the ANN has been built, we need to compile it.
# Compilation involves configuring the learning process before training the model.
# This step is essential for setting up various parameters that define how the model will learn. Usually:
# - Optimizer: the algorithm used to update the model parameters (weights and biases) during training.
# - Loss function or cost function used upon training.
# - Metrics are used to evaluate the performance of the model during training and testing.
#   By default, just the loss function is used.

# We compile the model.
# - Loss is sparse categorical cross entropy because the output is an int. If it was one-hot encoding,
#   we would use categorical cross entropy.
# - Adam optimizer, a stochastic gradient descent method that is based on adaptive momentum.
# - We add accuracy as a metric (together with the loss values).
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])


# TRAINING THE MODEL

# We train the model for 6 epochs, with batch_size of 32 instances.
# The validation data is used to validate the model after each epoch, **with a dataset different to train**
EPOCHS = 6
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_val, y_val))

# Let's take a look at how test and train loss and accuracy evolved during training
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[1, EPOCHS+1], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])  # red/blue, dashed(--), continuous (-), shape(. or *)
plt.show()

# You can see how:
# - Train and validation losses decrease as the number of epochs increases.
# - Train and validation accuracies increase as the number of epochs increases.
# - Validation loss is usually higher than train loss (probably not in the first epochs because validation is computed after the whole epoch, while train is averaged after each mini-batch).
# - Validation accuracy is usually lower than train loss (probably not in the first epochs).

# EVALUATING THE MODEL

# Let's evaluate the model. For that purpose, we use the test set. **Never use the val or train sets!**
evaluation_results = model.evaluate(X_test, y_test)
print(f"Test loss: {evaluation_results[0]:.4f}. Test accuracy: {evaluation_results[1]:.4f}.")


# MAKE PREDICTIONS (INFERENCE)

# We use the model to predict the classes of the first 5 clothes
# Remember we used softmax so, for each instance, we'll get 10 probabilities of belonging to each class
SAMPLES_TO_PREDICT = 5
y_proba = model.predict(X_to_predict := X_test[:SAMPLES_TO_PREDICT])
print("Probabilities of belonging to each class:\n", y_proba.round(2))

# We get the class with the maximum probability
y_pred = y_proba.argmax(axis=1)
print(f"Classes predicted for each instance (as number): {y_pred.tolist()}")
print(f"Classes predicted for each instance (as text): {[CLASS_LABELS[class_idx] for class_idx in y_pred]}")

# Let's visualize how good we did
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_to_predict):
    plt.subplot(1, SAMPLES_TO_PREDICT, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(CLASS_LABELS[y_test[index]])
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# We did great!

# STORING AND RESTORING MODELS

# Training could take many CPU time, so it is a good idea to store models after training them.
# Saves the model in 'models/mlp_model' dir.
# It could be stored in one single file with save_format="h5", but is not as widespread as "tf" format.
MODEL_DIR = "models/mlp_model"
model.save(MODEL_DIR, save_format="tf")

# We restore a serialized model from disk
new_model = tf.keras.models.load_model(MODEL_DIR)
print(f"Model loaded from '{MODEL_DIR}'.")
# And evaluate its performance
evaluation_results = new_model.evaluate(X_test, y_test)
print(f"Test loss: {evaluation_results[0]:.4f}. Test accuracy: {evaluation_results[1]:.4f}.")


# Questions:
# 1) What do you think it will happen with a) train loss, b) val loss, c) train accuracy and d) val accuracy
#    if you increase EPOCHS to 20. Write it down and then run it.
#    What is happening?
#    Why?
#    How do you think it should be solved? Name all the approaches you know.
# Answer:
#    a) train loss keeps decreasing
#    b) train accuracy keeps increasing
#    c) val loss starts increasing after epoch 12
#    d) val accuracy starts decreasing after epoch 12
# There is overfitting.
# Mechanisms to avoid overfitting:
# 1. One way of stopping that is detecting it and stopping (early stop).
# 2. Lowering model complexity (simpler ANN).
# 3. Getting more training data.
# 4. Feature selection (not possible in this example).
# 5. With regularization techniques (e.g., l2).
