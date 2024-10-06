# https://github.com/ageron/handson-ml3/blob/main/10_neural_nets_with_keras.ipynb

# We show how to perform hyperparameter tuning/optimization in keras.

# When defining ANNs, there are many hyperparameters to tweak.
# Not only ANN architecture parameters (the number of layers, the
# number of neurons and the type of activation function to use in each layer)
# but also the way they are trained (the initialization logic, the type of optimizer to use,
# its learning rate, the batch size, and more).

# The hyperparameter tuning/optimization problem involves finding the best set of hyperparameters
# for a machine learning model to optimize its performance on a given task or dataset.
# The goal of hyperparameter tuning/optimization is to search the hyperparameter space efficiently
# to find the set of hyperparameters that maximizes the model's performance metric, such as accuracy,
# loss, or some other evaluation metric. This process typically involves conducting multiple experiments
# with different hyperparameter configurations, training and evaluating the model for each configuration,
# and selecting the configuration that yields the best performance.

import tensorflow as tf
import keras_tuner as kt

# ADD TO THE NOTEBOOK %pip install -q -U keras_tuner

# DATA PREPARATION

# We download fashion MNIST the dataset from keras.
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
N_TRAIN_INSTANCES, N_VAL_INSTANCES = 1_000, 1_000
X_train, y_train = X_train_full[:N_TRAIN_INSTANCES], y_train_full[:N_TRAIN_INSTANCES]
X_val, y_val = X_train_full[-N_VAL_INSTANCES:], y_train_full[-N_VAL_INSTANCES:]
CLASS_LABELS = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# Show shapes of all the datasets
print(f"Shape of X_train = {X_train.shape} and y_train = {y_train.shape}.")
print(f"Shape of X_val = {X_val.shape} and y_val = {y_val.shape}.")
print(f"Shape of X_test = {X_test.shape} and y_test = {y_test.shape}.")
# We rescale the colors to real numbers between 0 and 1
X_train, X_val, X_test = X_train / 255, X_val / 255, X_test / 255

# HYPERPARAMETER TUNING

# We create a class derived from `HyperModel` to specify how to search for hyperparameters
# - Its `build` method creates and compiles a model, by specifying the ranges of values of each hyperparameter
# - Its `fil` method fits the model

class MyClassificationHyperModel(kt.HyperModel):
    def build(self, hp):
        # Searches for different number of hidden layers between 0 and 8
        n_hidden = hp.Int("n_hidden", min_value=0, max_value=8)
        n_neurons = hp.Int("n_neurons", min_value=16, max_value=256)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        optimizer = hp.Choice("optimizer", values=["sgd", "adam"])
        if optimizer == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten())
        for _ in range(n_hidden):
            model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax"))
        model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def fit(self, hp, model, X, y, **kwargs):
        return model.fit(X, y, **kwargs)

# There are different ways to search for hyperparameters.
# `RandomSearch` performs a random search.
# `BayesianOptimization` performs a probabilistic search (a Gaussian process) that approximates the objective
#  function (model performance metric) based on the observed evaluations of hyperparameters.

bayesian_opt_tuner = kt.BayesianOptimization(
    MyClassificationHyperModel(),
    objective="val_accuracy",  # objective function of the optimization problem
    max_trials=10,  # max number of executions
    directory="hyperparams",  # output folder
    overwrite=False)  # deletes the existing models in the output folder upon new execution

bayesian_opt_tuner.search(X_train, y_train, epochs=10,
                          validation_data=(X_val, y_val),
                          callbacks=[tf.keras.callbacks.EarlyStopping()])


# BEST MODEL AND HYPERPARAMETERS RETRIEVAL

hyperparameters = bayesian_opt_tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {hyperparameters.values}.")

best_trial = bayesian_opt_tuner.oracle.get_best_trials(num_trials=1)[0]
val_accuracy = best_trial.metrics.get_last_value("val_accuracy")
print(f"Validation accuracy with the best hyperparameters: {val_accuracy:.4f}.")

best_model = bayesian_opt_tuner.get_best_models(num_models=1)[0]
evaluation_results = best_model.evaluate(X_test, y_test)
print(f"Best model's loss: {evaluation_results[0]:.4f}. Best model's accuracy: {evaluation_results[1]:.4f}.")


# Questions:
# 1) Do you think the search will find a better model if it keeps searching?
# Answer: Yes, because it has only performed 10 executions and the search space is huge
# 2) Considering the hyperparameter tuning process as a black box, what sets do we have to pass it
# (train, train + val, train + val + test)? Why?
# Answer: We need train to train the model and val to measure its performance (never measure with samples used
#         in training). Test must never be included; it is only used to evaluate the best model.
# 3) Do you take the best model out of the hyperparameter tuning and use it for inference?
# Answer: No, you can find tune it with more epochs, more patience in early stopping, regularization and decreasing
#         the learning rate when the performance reaches a plateau.

