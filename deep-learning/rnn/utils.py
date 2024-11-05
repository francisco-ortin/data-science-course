import pandas as pd
import numpy as np
from keras import Model
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

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


def X_y_to_tensor_slices(X_p: list, y_p: list) -> tf.data.Dataset:
    """
    Convert the X and y lists to a TensorFlow tf.data.Dataset
    :param X_p: the time series
    :param y_p: the target variable
    :return: the TensorFlow dataset
    """
    return tf.data.Dataset.from_generator(
        lambda: zip(X_p, y_p),
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),  # X has shape [None, 4]: None for variable sequence length, and 4 for the number of features in each timestep
            tf.TensorSpec(shape=(), dtype=tf.float32)          # y is a scalar
        )
    )


def train_model(model: Model, train_set_p: tf.data.Dataset, val_set_p: tf.data.Dataset, learning_rate: float, epochs: int) \
        -> tf.keras.callbacks.History:
    """
    Train the model
    :param model: model to train
    :param train_set_p: the training set
    :param val_set_p: the validation set
    :param learning_rate: the learning rate
    :param epochs: the number of epochs
    :return: the training history
    """
    # Compile the model
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    # huber loss is less is useful for time series data
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=['mae', 'mape'])
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=2, restore_best_weights=True)
    history = model.fit(train_set_p, validation_data=val_set_p, epochs=epochs,
                        batch_size=1, callbacks=[early_stopping_cb])
    return history


def plot_history(history_p: tf.keras.callbacks.History, title: str) -> None:
    """
    Plot the loss and the mean absolute error (MAE) for the training and validation sets
    :param history_p: the training history
    :param title: the title of the plot
    """
    plt.plot(history_p.history['loss'], label='Training loss')
    plt.plot(history_p.history['val_loss'], label='Validation loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_predictions(y_test_p: np.array, y_pred_p: np.array, n_instances: int, title: str) -> None:
    """
    Plot the first 100 instances of y_pred vs y_test
    :param y_test_p: the true values
    :param y_pred_p: the predicted values
    :param n_instances: the number of instances to plot
    """
    plt.figure(figsize=(12, 3.5))
    plt.plot(y_test_p[:n_instances], marker=".", label="True values")
    plt.plot(y_pred_p[:n_instances], marker=".", label="Predicted values")
    plt.title(title)
    plt.legend()
    plt.show()

