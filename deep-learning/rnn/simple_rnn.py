# The ridership data comes from Chicago's Transit Authority, and was downloaded from the Chicago's Data Portal at https://data.cityofchicago.org/Transportation/CTA-Ridership-Daily-Boarding-Totals/6iiy-9s97/about_data
import numpy as np
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

## LOAD THE DATA

dataset = pd.read_csv('data/transit.csv', parse_dates=['service_date'])  # parse_dates converts the column to datetime
# we will use the number of passengers for the train as the target variable, so bus and total_rides columns are not necessary
dataset = dataset.drop(columns=['bus', 'total_rides'])
# rename the rail_boardings column to rail and the service_date column to date
dataset = dataset.rename(columns={"rail_boardings": "rail", "service_date": "date"})
# remove duplicates
dataset = dataset.drop_duplicates()
# sort by date (ascending) and set the date as the index of the dataframe
dataset = dataset.sort_values("date").set_index("date")
# display the first few rows of the dataset
print("First few rows of the dataset:")
print(dataset.head())


# Let's plot the number of train passengers for the first semester months of 2001
#dataset["2001-01":"2001-06"].plot(grid=True, marker=".", figsize=(12, 3.5))
#plt.show()

# The default activation function for the RNN is the hyperbolic tangent (tanh) function,
# which outputs values between -1 and 1. A simple RNN uses the output as part of the input for the next time step.
# Therefore, we need to scale the data to be between -1 and 1. We use the MinMaxScaler from scikit-learn to scale the data.

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset['rail'] = scaler.fit_transform(dataset[['rail']])
# convert the date_type column to a one-hot encoded column
dataset = pd.get_dummies(dataset, columns=["day_type"])
# map True to 1 and False to 0
dataset = dataset.astype({"day_type_A": int, "day_type_U": int, "day_type_W": int})
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
    i = 0
    while i < len(dataset_p):
        series_length = np.random.randint(length_from_p, length_to_p + 1)
        if i + series_length < len(dataset_p):
            X.append(np.array(dataset_p.iloc[i:i + series_length][['rail', 'day_type_A', 'day_type_U', 'day_type_W']].values))
            y.append(dataset_p.iloc[i + series_length]['rail'])
            i += series_length
        else:
            break
    return X, y


length_from, length_to = 40, 60
X_train, y_train = create_variable_time_series_dataset(train_ds, length_from, length_to)
X_val, y_val = create_variable_time_series_dataset(val_ds, length_from, length_to)
X_test, y_test = create_variable_time_series_dataset(test_ds, length_from, length_to)
print(f"Length of X: {len(X_train)}")
print(f"Length of y: {len(y_train)}")
print("First few elements of X:")
print(X_train[:1])
print("First few elements of y:")
print(y_train[:5])


# Sample variable-length sequences for X and corresponding scalar targets for y
"""X = [
    np.array([[0.1], [0.2], [0.3]]),  # Sequence of 3 timesteps
    np.array([[0.5], [0.6]]),  # Sequence of 2 timesteps
    np.array([[0.4], [0.8], [0.9], [1.0]]),  # Sequence of 4 timesteps
    np.array([[0.1]]),  # Sequence of 1 timestep
    np.array([[0.7], [0.2], [0.9]])  # Sequence of 3 timesteps
]

y = [
    0.3,  # Target for the first sequence
    0.5,  # Target for the second sequence
    0.7,  # Target for the third sequence
    0.1,  # Target for the fourth sequence
    0.4  # Target for the fifth sequence
]
"""

# Select sequences of random length between 30 and 60 days using tf.keras.utils.timeseries_dataset_from_array

def fit_and_evaluate(model_p: Model, train_set_p: pd.DataFrame, val_ds_p: pd.DataFrame, learning_rate: float, epochs: int=500) -> float:
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=50, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model_p.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    history = model_p.fit(train_set_p, validation_data=val_ds_p, epochs=epochs,
                          callbacks=[early_stopping_cb], batch_size=1)  # batch_size=1 for RNNs to allow for variable length sequences
    valid_loss, valid_mae = model_p.evaluate(val_ds_p)
    return valid_mae * 1e6

def X_y_to_tensor_slices(X_p: list, y_p: list) -> tf.data.Dataset:
    # Convert to TensorFlow Dataset
    return tf.data.Dataset.from_generator(
        lambda: zip(X_p, y_p),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # X has shape [None, 5]: None for variable sequence length, and 5 for the number of features in each timestep
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
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 4])  # None = variable length sequence, 4 = number of features in each timestep
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae', 'mape'])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae", patience=5, restore_best_weights=True)
# Train the model without specifying batch_size (since dataset is already batched)
history = model.fit(train_set_tf, validation_data=val_set_tf, epochs=500,
                    batch_size=1, callbacks=[early_stopping_cb])

test_loss, test_mae, test_mape = model.evaluate(train_set_tf)
print(f"Test MAE: {scaler.inverse_transform([[test_mae]])[0][0]:.0f}.")
print(f"Test MAPE: {test_mape:.2f}.")
