import seaborn as sns
import matplotlib.pyplot as plt
from keras.src.callbacks import History
import numpy as np
import pandas as pd

def plot_function(x: np.ndarray, y: np.array, title: str) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


def plot_functions(x: np.ndarray, y: np.array, y_predicted: np.array, title: str) -> None:
    plt.figure(figsize=(5,5))
    plt.plot(x, y, label="Original function")
    plt.plot(x, y_predicted, label="Predicted function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.show()


def show_history(history: History, loss_label: str, accuracy_label: str) -> None:
    """Function that shows the loss and accuracy plots for a given training history"""
    # Plot training loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label=loss_label)
    plt.title(loss_label)
    plt.xlabel('Epoch')
    plt.ylabel(loss_label)

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label=accuracy_label)
    plt.title(accuracy_label)
    plt.xlabel('Epoch')
    plt.ylabel(accuracy_label)

    plt.tight_layout()
    plt.show()


def show_loss(history: History):
    """Function that shows the loss of the model, given the training history"""
    plt.plot(history.history['loss'], label='Training MSE')
    plt.plot(history.history['mean_absolute_error'], label=' Training MAE')
    plt.title('Model MSE and MEA')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def show_distribution(dataframe: pd.DataFrame) -> None:
    """Show the distribution of the features in the iris dataset"""
    plt.figure('Distribution of Iris Dataset Features', figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(dataframe['sepal length (cm)'], kde=True, color='blue')
    plt.title('Distribution of Sepal Length')

    plt.subplot(2, 2, 2)
    sns.histplot(dataframe['sepal width (cm)'], kde=True, color='orange')
    plt.title('Distribution of Sepal Width')

    plt.subplot(2, 2, 3)
    sns.histplot(dataframe['petal length (cm)'], kde=True, color='green')
    plt.title('Distribution of Petal Length')

    plt.subplot(2, 2, 4)
    sns.histplot(dataframe['petal width (cm)'], kde=True, color='red')
    plt.title('Distribution of Petal Width')

    plt.tight_layout()
    plt.show()
