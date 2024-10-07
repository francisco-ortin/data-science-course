import matplotlib.pyplot as plt
from keras.src.callbacks import History


def show_history(history: History, loss_label: str, accuracy_label: str) -> None:
    """
    Function that shows the loss and accuracy plots for a given training history
    :param history: Training history
    :param loss_label: Label for the loss plot
    :param accuracy_label: Label for the accuracy plot
    """
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


