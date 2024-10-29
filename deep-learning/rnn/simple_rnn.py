# The ridership data comes from Chicago's Transit Authority, and was downloaded from the Chicago's Data Portal at https://data.cityofchicago.org/Transportation/CTA-Ridership-Daily-Boarding-Totals/6iiy-9s97/about_data
# The dataset contains the  daily number of bus and train passengers from 2001 to 2021.

# The dataset contains the following columns:
#     - service_date: the date of the service
#     - day_type: the type of day (W=weekday, A=sAturday, U=sUnday)
#     - bus: the number of bus passengers
#     - rail_boardings: the number of train passengers
#     - total_rides: the total number of passengers

import pandas as pd
import matplotlib.pyplot as plt
from keras import Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np

## LOAD THE DATA

def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load the dataset from the CSV file
    :param file_name: the path to the CSV file
    :return: the dataset
    """
    ds = pd.read_csv(file_name, parse_dates=['service_date'])  # parse_dates converts the column to datetime
    # we will use the number of passengers for the train as the target variable, so bus and total_rides columns are not necessary
    ds = ds.drop(columns=['bus', 'total_rides'])
    # rename the rail_boardings column to rail and the service_date column to date
    ds = ds.rename(columns={"rail_boardings": "rail", "service_date": "date"})
    # remove duplicates
    ds = ds.drop_duplicates()
    # sort by date (ascending) and set the date as the index of the dataframe
    ds = ds.sort_values("date").set_index("date")
    return ds


dataset = load_dataset("data/transit.csv")
# display the first few rows of the dataset
print("First few rows of the dataset:")
print(dataset.head())


# Let's plot the number of train passengers for the first semester months of 2001
dataset["2001-01":"2001-06"].plot(grid=True, marker=".", figsize=(12, 3.5))
plt.show()

# The default activation function for the RNN is the hyperbolic tangent (tanh) function,
# which outputs values between -1 and 1. A simple RNN uses the output as part of the input for the next time step.
# Therefore, we need to scale the data to be between -1 and 1. We use the MinMaxScaler from scikit-learn to scale the data.


def scale_data_and_convert_one_hot(dataset_p: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):
    """
    Scale the data between -1 and 1 and convert the day_type column to one-hot encoding
    :param dataset_p: the dataset
    :return: the scaled dataset
    """
    # Scale the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset_p['rail'] = scaler.fit_transform(dataset_p[['rail']])
    # convert the date_type column to a one-hot encoded column
    dataset_p = pd.get_dummies(dataset_p, columns=["day_type"])
    # map True to 1 and False to 0
    dataset_p = dataset_p.astype({"day_type_A": int, "day_type_U": int, "day_type_W": int})
    return dataset_p, scaler


dataset, scaler = scale_data_and_convert_one_hot(dataset)
print("First few rows of the dataset after scaling and one-hot encoding:")
print(dataset.head())

## SPLIT THE DATA INTO TRAINING, VALIDATION AND TESTING SETS

train_ds = dataset[:"2015-12"]
val_ds = dataset["2016-01":"2018-12"]
test_ds = dataset["2019-01":]

def create_variable_time_series_dataset(dataset_p: pd.DataFrame, length_from_p: int, length_to_p: int) \
        -> (list, list):
    """
    Create a variable length time series dataset
    :param dataset_p: the original dataset with the time series
    :param length_from_p: the minimum length of the time series X set
    :param length_to_p: the maximum length of the time series X set
    :return: (X, y) where X is the time series and y is the target variable
    """
    X, y = [], []
    for i in range(0, len(dataset_p)):
        series_length = np.random.randint(length_from_p, length_to_p + 1)
        if i + series_length < len(dataset_p):
            X.append(np.array(dataset_p.iloc[i:i + series_length][['rail', 'day_type_A', 'day_type_U', 'day_type_W']].values))
            y.append(dataset_p.iloc[i + series_length]['rail'])
        else:
            break
    return X, y


length_from, length_to = 40, 60
X_train, y_train = create_variable_time_series_dataset(train_ds, length_from, length_to)
X_val, y_val = create_variable_time_series_dataset(val_ds, length_from, length_to)
X_test, y_test = create_variable_time_series_dataset(test_ds, length_from, length_to)


def X_y_to_tensor_slices(X_p: list, y_p: list) -> tf.data.Dataset:
    # Convert to TensorFlow Dataset
    return tf.data.Dataset.from_generator(
        lambda: zip(X_p, y_p),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # X has shape [None, 4]: None for variable sequence length, and 4 for the number of features in each timestep
            tf.TensorSpec(shape=(), dtype=tf.float32)          # y is a scalar
        )
    )

train_set_tf = X_y_to_tensor_slices(X_train, y_train)
val_set_tf = X_y_to_tensor_slices(X_val, y_val)
test_set_tf = X_y_to_tensor_slices(X_test, y_test)

# Batch each sequence individually (batch of 1) to allow variable-length input
# (mandatory for training RNNs with variable-length sequences (not very common)).
# At inference, any length can be used, even though the model was trained with a specific length.
train_set_tf = (train_set_tf.batch(1)  # organizes the dataset into batches:
                # adds a new dimension batch-size to the dataset, grouping elements into batches
                # batch size is 1 to allow for variable length sequences
                .prefetch(1))  # prefetches the next batch while training on the current batch (pipeline parallelism)
                        # it is set to 1 to prefetch 1 batch while training on another 1 batch

val_set_tf = val_set_tf.batch(1).prefetch(1)
test_set_tf = test_set_tf.batch(1).prefetch(1)

# Define the model
simple_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 4])  # None = variable length sequence, 4 = number of features in each timestep
])
print(simple_model.summary())

def train_model(model: Model, train_set_p: tf.data.Dataset, val_set_p: tf.data.Dataset, learning_rate: float, epochs: int) \
        -> tf.keras.callbacks.History:
    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    # huber loss is less is useful for time series data
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae', 'mape'])
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=2, restore_best_weights=True)
    history = model.fit(train_set_p, validation_data=val_set_p, epochs=epochs,
                        batch_size=1, callbacks=[early_stopping_cb])
    return history


epochs = 500
history = train_model(simple_model, train_set_tf, val_set_tf, learning_rate=0.01, epochs=epochs)


def plot_history(history_p: tf.keras.callbacks.History) -> None:
    """
    Plot the loss and the mean absolute error (MAE) for the training and validation sets
    :param history_p: the training history
    """
    plt.plot(history_p.history['loss'], label='Training loss')
    plt.plot(history_p.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


plot_history(history)
test_loss, test_mae, test_mape = simple_model.evaluate(test_set_tf)
# The test MAE is scaled, so we need to inverse the scaling to get the actual value.
# Since the output is a 2d array (bath_size, y features), we need to select the first element of the first element of the array.
print(f"Test MAE: {scaler.inverse_transform([[test_mae]])[0][0]:.0f}.")
print(f"Test MAPE: {test_mape:.2f}.")


def plot_predictions(y_test_p: np.array, y_pred_p: np.array, n_instances: int):
    """
    Plot the first 100 instances of y_pred vs y_test
    :param y_test_p: the true values
    :param y_pred_p: the predicted values
    :param n_instances: the number of instances to plot
    """
    plt.figure(figsize=(12, 3.5))
    plt.plot(y_test_p[:n_instances], marker=".", label="True values")
    plt.plot(y_pred_p[:n_instances], marker=".", label="Predicted values")
    plt.legend()
    plt.show()


n_instances_to_plot = 100
y_pred = simple_model.predict(test_set_tf)
plot_predictions(y_test, y_pred, n_instances_to_plot)


## MULTIPLE SIMPLE RECURRENT NEURONS

# The model is not very accurate, but it is a simple model. We can try to improve the model by adding a layer of simple RNNs.
# Instead of one simple recurrent neuron, we can add 32 simple recurrent neurons to the model.
# Since the output has to be a scalar, we need to add a Dense layer with one neuron at the end of the model.

multi_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 4]),
    tf.keras.layers.Dense(1)  # no activation function by default (regression model)
])
print(multi_model.summary())

history = train_model(multi_model, train_set_tf, val_set_tf, learning_rate=0.01, epochs=epochs)
plot_history(history)
test_loss, test_mae, test_mape = multi_model.evaluate(test_set_tf)
print(f"Test MAE: {scaler.inverse_transform([[test_mae]])[0][0]:.0f}.")
print(f"Test MAPE: {test_mape:.2f}.")

y_pred = multi_model.predict(test_set_tf)
plot_predictions(y_test, y_pred, n_instances_to_plot)


## DEEP RNN

# We can also try to improve the model by adding more layers to the model.
# We can add three layers of simple RNNs with 32 neurons each.
# Each layer will return the output of each time step, so they can be stacked.
# To that aim we set return_sequences=True.
# We can also add a Dense layer with one neuron at the end of the model.


deep_model = tf.keras.Sequential([
    # return_sequences=True is used to return the output of each time step, not just the last one
    # in this way, the next layer can process the output of each time step (they can be stacked)
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 4], return_sequences=True),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 4]),
    tf.keras.layers.Dense(1)  # no activation function by default (regression model)
])
print(deep_model.summary())

history = train_model(deep_model, train_set_tf, val_set_tf, learning_rate=0.01, epochs=epochs)
plot_history(history)
test_loss, test_mae, test_mape = deep_model.evaluate(test_set_tf)
print(f"Test MAE: {scaler.inverse_transform([[test_mae]])[0][0]:.0f}.")
print(f"Test MAPE: {test_mape:.2f}.")

y_pred = deep_model.predict(test_set_tf)
plot_predictions(y_test, y_pred, n_instances_to_plot)


## QUESTIONS:
# 1. Are model preformed being improved by adding more complexity/parameters?
# Answer: No, the sencond model performs better, but no the third one..
# EN EL TERCER MODELO EL LOSS DE TRAIN BAJA MUCHÍSIMO, PERO EL DE VALIDACIÓN SUBE TRAS EL PRIMER EPOCH (SOBREAJUSTE)
# 2. Why?
# 3. Do you think that will always be the case?
# No, it depends on the data and the model. In this case, the third model is overfitting the data (too cmplex).

