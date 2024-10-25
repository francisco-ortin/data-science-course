# The ridership data comes from Chicago's Transit Authority, and was downloaded from the Chicago's Data Portal at https://data.cityofchicago.org/Transportation/CTA-Ridership-Daily-Boarding-Totals/6iiy-9s97/about_data
import numpy as np
# The dataset contains the  daily number of bus and train passengers from 2001 to 2021.

# The dataset contains the following columns:
#     - service_date: the date of the service
#     - day_type: the type of day (W=weekday, A=sAturday, U=sUnday)
#     - bus: the number of bus passengers
#     - rail: the number of train passengers
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
# remove duplicates
dataset = dataset.drop_duplicates()
# sort by date (ascending) and set the date as the index of the dataframe
dataset = dataset.sort_values("service_date").set_index("service_date")
# display the first few rows of the dataset
print("First few rows of the dataset:")
print(dataset.head())


# Let's plot the number of train passengers for the first semester months of 2001
dataset["2001-01":"2001-06"].plot(grid=True, marker=".", figsize=(12, 3.5))
plt.show()

# The default activation function for the RNN is the hyperbolic tangent (tanh) function,
# which outputs values between -1 and 1. A simple RNN uses the output as part of the input for the next time step.
# Therefore, we need to scale the data to be between -1 and 1. We use the MinMaxScaler from scikit-learn to scale the data.

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset['rail_boardings'] = scaler.fit_transform(dataset[['rail_boardings']])
print("First few rows of the dataset after scaling:")
print(dataset.head())

## SPLIT THE DATA INTO TRAINING, VALIDATION AND TESTING SETS

train_ds = dataset[:"2015-12"]
val_ds = dataset["2016-01":"2018-12"]
test_ds = dataset["2019-01":]

# Select sequences of random length between 30 and 60 days using tf.keras.utils.timeseries_dataset_from_array

model = tf.keras.Sequential([
 tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

print(model.summary())

def fit_and_evaluate(model_p: Model, train_set_p: pd.DataFrame, val_ds_p: pd.DataFrame, learning_rate: float, epochs: int=500) -> float:
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=50, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model_p.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    history = model_p.fit(train_set_p, validation_data=val_ds_p, epochs=epochs,
                          callbacks=[early_stopping_cb], batch_size=1)  # batch_size=1 for RNNs to allow for variable length sequences
    valid_loss, valid_mae = model_p.evaluate(val_ds_p)
    return valid_mae * 1e6

# TODO: CONVERTIR LOS DATOS DEL DATASET A LA ESTRUCTURA DE ABAJO Y MOVERLO A UTILS.PY

# Sample variable-length sequences for X and corresponding scalar targets for y
X = [
    np.array([[0.1], [0.2], [0.3]]),       # Sequence of 3 timesteps
    np.array([[0.5], [0.6]]),              # Sequence of 2 timesteps
    np.array([[0.4], [0.8], [0.9], [1.0]]),# Sequence of 4 timesteps
    np.array([[0.1]]),                     # Sequence of 1 timestep
    np.array([[0.7], [0.2], [0.9]])        # Sequence of 3 timesteps
]

y = [
    0.3,  # Target for the first sequence
    0.5,  # Target for the second sequence
    0.7,  # Target for the third sequence
    0.1,  # Target for the fourth sequence
    0.4   # Target for the fifth sequence
]

# Convert to TensorFlow Dataset
train_set_p = tf.data.Dataset.from_generator(
    lambda: zip(X, y),
    output_signature=(
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),  # X has shape [None, 1] for variable length
        tf.TensorSpec(shape=(), dtype=tf.float32)          # y is a scalar
    )
)

print("First few elements of the dataset:")
print(train_set_p)

# Batch each sequence individually (batch of 1) to allow variable-length input
# (mandatory for training RNNs with variable-length sequences (not very common)).
# At inference, any length can be used, even though the model was trained with a specific length.
train_set_p = (train_set_p.batch(1)  # organizes the dataset into batches:
               # adds a new dimension batch-size to the dataset, grouping elements into batches
        # batch size is 1 to allow for variable length sequences
               .prefetch(1))  # prefetches the next batch while training on the current batch (pipeline parallelism)
                        # it is set to 1 to prefetch 1 batch while training on another 1 batch
val_set_p = train_set_p
test_set_p = train_set_p

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])  # RNN input shape accommodates variable length
])

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae'])

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mae", patience=2, restore_best_weights=True)
# Train the model without specifying batch_size (since dataset is already batched)
history = model.fit(train_set_p, validation_data=val_set_p, epochs=50,
                    batch_size=1, callbacks=[early_stopping_cb])

test_loss, test_mae = model.evaluate(train_set_p)
print(f"Test MAE: {scaler.inverse_transform([[test_mae]])[0][0]:.0f}")
